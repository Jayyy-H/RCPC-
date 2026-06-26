import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import ImageFile
from tqdm import tqdm
from transformers import AutoConfig

from inference_ipr_metrics_jsonl import (
    append_token_ids,
    build_inputs,
    choose_model_class,
    collate_model_inputs,
    count_jsonl_rows,
    env_bool,
    infer_label,
    iter_jsonl,
    normalize_extra_info,
    resolve_model_path,
    single_token_id,
)
from verl.utils import get_processor, get_tokenizer


def resolve_binary_model_path(args: argparse.Namespace) -> str:
    try:
        return resolve_model_path(args.model_path, args.auto_merge_checkpoint)
    except RuntimeError as exc:
        raw_path = Path(args.model_path).expanduser()
        final_candidates = []
        if raw_path.name.startswith("checkpoint-"):
            final_candidates.append(raw_path.parent / "final")
        final_candidates.append(raw_path / "final")

        resolved_final = None
        for final_candidate in final_candidates:
            if not final_candidate.exists():
                continue
            try:
                resolved_final = resolve_model_path(str(final_candidate), args.auto_merge_checkpoint)
                break
            except Exception:
                continue

        if args.fallback_to_final and resolved_final is not None:
            print(
                f"[WARN] MODEL_PATH={raw_path} is not a loadable HuggingFace model directory; "
                f"falling back to final checkpoint: {resolved_final}",
                flush=True,
            )
            return resolved_final

        message = [
            str(exc),
            "",
            "This path looks like a Transformers Trainer/DeepSpeed intermediate checkpoint, "
            "not a complete HuggingFace model directory. This binary SFT inference script loads "
            "model weights directly with from_pretrained(), so MODEL_PATH must contain config.json "
            "and model weights such as model.safetensors or pytorch_model.bin.",
        ]
        if resolved_final is not None:
            message.append(f"Found a loadable final model nearby. Use: MODEL_PATH={resolved_final}")
            message.append(
                "If you intentionally want the script to use that nearby final model when a "
                "checkpoint-* path is passed, set FALLBACK_TO_FINAL=true."
            )
        else:
            message.append(
                "If you need to evaluate this exact intermediate checkpoint, first export or "
                "convert it into a HuggingFace loadable directory, or save a full HF snapshot at "
                "that step during SFT."
            )
        raise RuntimeError("\n".join(message)) from exc


def score_binary_batch(
    *,
    items: List[Dict[str, Any]],
    model: Any,
    tokenizer: Any,
    yes_id: int,
    no_id: int,
) -> List[Dict[str, Any]]:
    scoring_inputs_list = [append_token_ids(item["model_inputs"], [yes_id]) for item in items]
    scoring_inputs = collate_model_inputs(scoring_inputs_list, tokenizer)
    with torch.inference_mode():
        outputs = model(**scoring_inputs, use_cache=False, return_dict=True)

    logits_for_answer = outputs.logits[:, -2, :]
    yes_logits = logits_for_answer[:, yes_id].float()
    no_logits = logits_for_answer[:, no_id].float()
    binary_logits = torch.stack([no_logits, yes_logits], dim=-1)
    yes_probs = torch.softmax(binary_logits, dim=-1)[:, 1]

    results = []
    for yes_logit, no_logit, yes_prob in zip(yes_logits, no_logits, yes_probs):
        yes_probability = float(yes_prob.detach().cpu().item())
        yes_logit_value = float(yes_logit.detach().cpu().item())
        no_logit_value = float(no_logit.detach().cpu().item())
        pred_answer = "Yes" if yes_probability >= 0.5 else "No"
        results.append(
            {
                "pred_answer": pred_answer,
                "response": pred_answer,
                "yes_probability": yes_probability,
                "yes_logit": yes_probability,
                "yes_token_logit": yes_logit_value,
                "no_token_logit": no_logit_value,
                "binary_logit_margin": yes_logit_value - no_logit_value,
                "score_source": "next_token_binary_yes_no_logits",
                "format_ok": True,
            }
        )
    return results


def run_inference(args: argparse.Namespace) -> None:
    ImageFile.LOAD_TRUNCATED_IMAGES = args.load_truncated_images

    model_path = resolve_binary_model_path(args)
    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"input file does not exist: {args.input_file}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    tokenizer = get_tokenizer(model_path, trust_remote_code=args.trust_remote_code)
    processor = get_processor(
        model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )
    if processor is None:
        raise RuntimeError(f"processor not found in {model_path}")

    yes_id = single_token_id(tokenizer, args.yes_token_text)
    no_id = single_token_id(tokenizer, args.no_token_text)
    print(
        json.dumps(
            {
                "binary_answer_tokens": {
                    "class_0": {"label": "No", "text": args.no_token_text, "token_id": no_id},
                    "class_1": {"label": "Yes", "text": args.yes_token_text, "token_id": yes_id},
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    model_cls = choose_model_class(config)
    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval().to(device)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = count_jsonl_rows(args.input_file)
    shard_rows = len(range(args.shard_rank, total_rows, args.num_shards))
    if args.limit is not None:
        shard_rows = min(shard_rows, args.limit)

    processed = 0
    written = 0
    skipped = 0
    missing_label = 0
    pending_items: List[Dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8", buffering=1) as fout:
        progress = tqdm(
            total=shard_rows,
            desc=f"binary infer shard {args.shard_rank + 1}/{args.num_shards}",
            position=args.shard_rank if args.num_shards > 1 else 0,
        )

        def write_result(item: Dict[str, Any], inference: Dict[str, Any]) -> None:
            nonlocal written, missing_label

            row = item["row"]
            result = dict(row)
            result["id"] = result.get("id", str(item["line_no"]))
            result["response"] = inference["response"]
            result["pred_answer"] = inference["pred_answer"]
            result["format_ok"] = inference["format_ok"]
            result["yes_logit"] = format(inference["yes_logit"], ".8f")
            result["yes_probability"] = format(inference["yes_probability"], ".8f")
            result["yes_token_logit"] = format(inference["yes_token_logit"], ".8f")
            result["no_token_logit"] = format(inference["no_token_logit"], ".8f")
            result["binary_logit_margin"] = format(inference["binary_logit_margin"], ".8f")
            result["yes_logit_source"] = inference["score_source"]
            result["image_count"] = len(item["image_paths"])

            label = infer_label(result, args.target_key)
            if label is None:
                missing_label += 1
            elif "label" not in result:
                result["label"] = label

            extra_info = normalize_extra_info(result)
            fusion_model_res = extra_info.get("fusion_model_res")
            if not isinstance(fusion_model_res, dict):
                fusion_model_res = {}
                extra_info["fusion_model_res"] = fusion_model_res
            if args.baseline_score_fallback == "yes_logit" and fusion_model_res.get("valley_score") in (None, ""):
                fusion_model_res["valley_score"] = result["yes_logit"]
                fusion_model_res["valley_score_source"] = "fallback_from_binary_yes_logit"
            result["extra_info"] = extra_info

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1

        def process_batch(items: List[Dict[str, Any]]) -> None:
            nonlocal skipped
            if not items:
                return
            try:
                inferences = score_binary_batch(
                    items=items,
                    model=model,
                    tokenizer=tokenizer,
                    yes_id=yes_id,
                    no_id=no_id,
                )
            except Exception as exc:
                if len(items) > 1:
                    print(
                        f"[WARN] batch_size={len(items)} failed with {type(exc).__name__}: {exc}; "
                        "retrying one sample at a time",
                        flush=True,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for item in items:
                        process_batch([item])
                    return
                skipped += 1
                if not args.skip_errors:
                    raise
                print(f"[WARN] skip line {items[0]['line_no']}: {type(exc).__name__}: {exc}", flush=True)
                progress.update(1)
                return

            for item, inference in zip(items, inferences):
                write_result(item, inference)
            progress.update(len(items))
            progress.set_postfix({"written": written, "skipped": skipped, "batch": len(items)})

        for raw_index, (line_no, row) in enumerate(iter_jsonl(args.input_file)):
            if raw_index % args.num_shards != args.shard_rank:
                continue
            if args.limit is not None and processed >= args.limit:
                break
            processed += 1

            try:
                if args.prompt_key not in row:
                    raise KeyError(f"missing prompt key: {args.prompt_key}")
                model_inputs, _, image_paths = build_inputs(
                    row,
                    tokenizer,
                    processor,
                    config,
                    device,
                    dtype,
                    args.prompt_key,
                    args.max_pixels,
                    args.min_pixels,
                )
                pending_items.append(
                    {
                        "line_no": line_no,
                        "row": row,
                        "model_inputs": model_inputs,
                        "image_paths": image_paths,
                    }
                )
                if len(pending_items) >= args.batch_size:
                    process_batch(pending_items)
                    pending_items = []
            except Exception:
                skipped += 1
                if not args.skip_errors:
                    raise
                print(f"[WARN] skip line {line_no}", flush=True)
                traceback.print_exc()
                progress.update(1)

        process_batch(pending_items)
        progress.close()

    summary = {
        "input_file": args.input_file,
        "output_file": args.output_file,
        "processed": processed,
        "written": written,
        "skipped": skipped,
        "missing_label": missing_label,
        "score_source": "next_token_binary_yes_no_logits",
        "batch_size": args.batch_size,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--prompt_key", default="problem")
    parser.add_argument("--target_key", default="solution")
    parser.add_argument("--yes_token_text", default="Yes")
    parser.add_argument("--no_token_text", default="No")
    parser.add_argument("--max_pixels", type=int, default=100352)
    parser.add_argument("--min_pixels", type=int, default=50176)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--trust_remote_code", action="store_true", default=env_bool("TRUST_REMOTE_CODE", True))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--skip_errors", action="store_true")
    parser.add_argument("--auto_merge_checkpoint", action="store_true")
    parser.add_argument("--fallback_to_final", action="store_true", default=env_bool("FALLBACK_TO_FINAL", False))
    parser.add_argument("--load_truncated_images", action="store_true", default=env_bool("LOAD_TRUNCATED_IMAGES", True))
    parser.add_argument("--baseline_score_fallback", choices=("none", "yes_logit"), default="none")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be greater than 0")
    if args.num_shards <= 0:
        parser.error("--num_shards must be greater than 0")
    if args.shard_rank < 0 or args.shard_rank >= args.num_shards:
        parser.error("--shard_rank must be in [0, num_shards)")
    return args


if __name__ == "__main__":
    run_inference(parse_args())
