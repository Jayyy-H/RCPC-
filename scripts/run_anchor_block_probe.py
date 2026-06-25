#!/usr/bin/env python3
"""Run RCPC candidate action anchoring and local block aggregation.

This script is intentionally dataset-agnostic. It reads JSONL/JSON/CSV
records, runs a Qwen/Qwen3-style causal LM, records generated-token entropy,
splits the response into micro-actions, selects Top-K candidate actions, and
constructs peak-centered local candidate blocks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm


@dataclass
class ReasoningSample:
    sample_id: str
    source: str
    question: str
    answer: str = ""


@dataclass
class TokenTrace:
    token_index: int
    token_id: int
    text: str
    entropy: float
    char_start: int
    char_end: int


@dataclass
class ActionSpan:
    action_index: int
    text: str
    char_start: int
    char_end: int
    token_indices: List[int]
    token_start: Optional[int]
    token_end: Optional[int]
    token_count: int
    entropy_topr_mean: float
    entropy_robust_z: float


@dataclass
class CandidateBlock:
    block_index: int
    anchor_action_id: int
    action_ids: List[int]
    action_start: int
    action_end: int
    text: str
    char_start: int
    char_end: int
    token_start: Optional[int]
    token_end: Optional[int]
    token_count: int
    anchor_robust_E: float
    block_mean_robust_E: float
    block_max_robust_E: float


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        obj = json.loads(text)
        if not isinstance(obj, list):
            raise ValueError(f"JSON file must contain a list: {path}")
        return [dict(x) for x in obj]
    rows = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_samples(
    data_path: Path,
    num_samples: int,
    seed: int,
    id_field: str,
    source_field: str,
    question_field: str,
    answer_field: str,
    shuffle: bool,
) -> List[ReasoningSample]:
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        rows = read_csv(data_path)
    elif suffix in {".json", ".jsonl"}:
        rows = read_json_or_jsonl(data_path)
    else:
        raise ValueError(f"Unsupported data format: {data_path}. Use JSONL, JSON, or CSV.")

    samples: List[ReasoningSample] = []
    for idx, row in enumerate(rows):
        question = str(row.get(question_field, "")).strip()
        if not question:
            continue
        sample_id = str(row.get(id_field, f"sample-{idx:06d}"))
        source = str(row.get(source_field, data_path.stem))
        answer = str(row.get(answer_field, ""))
        samples.append(ReasoningSample(sample_id=sample_id, source=source, question=question, answer=answer))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)
    if num_samples > 0:
        samples = samples[:num_samples]
    return samples


def build_prompt(sample: ReasoningSample, template: str) -> str:
    return template.format(question=sample.question, answer=sample.answer, source=sample.source, sample_id=sample.sample_id)


def load_model(model_path: str, device: torch.device, dtype_arg: str, trust_remote_code: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype_arg == "auto":
        dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    elif dtype_arg == "bf16":
        dtype = torch.bfloat16
    elif dtype_arg == "fp16":
        dtype = torch.float16
    elif dtype_arg == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype_arg}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def prepare_inputs(prompt: str, tokenizer: Any, device: torch.device) -> Dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt
    inputs = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}


def generate_with_entropy(
    model: Any,
    tokenizer: Any,
    inputs: Dict[str, torch.Tensor],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> Tuple[str, List[TokenTrace], List[int]]:
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "return_dict_in_generate": True,
        "output_scores": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    prompt_len = int(inputs["input_ids"].shape[-1])
    full_ids = outputs.sequences[0].detach().cpu().tolist()
    gen_ids = full_ids[prompt_len:]
    scores = list(outputs.scores or [])
    if len(scores) < len(gen_ids):
        gen_ids = gen_ids[: len(scores)]

    response = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    traces: List[TokenTrace] = []
    prev_text = ""
    for i, token_id in enumerate(gen_ids):
        prefix_text = tokenizer.decode(gen_ids[: i + 1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        piece = prefix_text[len(prev_text) :]
        char_start = len(prev_text)
        char_end = len(prefix_text)
        prev_text = prefix_text

        logits = scores[i][0].detach().float()
        finite_mask = torch.isfinite(logits)
        if finite_mask.any():
            logits = logits[finite_mask]
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        entropy = float(-(probs * log_probs).sum().detach().cpu().item())
        traces.append(
            TokenTrace(
                token_index=i,
                token_id=int(token_id),
                text=piece,
                entropy=entropy,
                char_start=char_start,
                char_end=char_end,
            )
        )
    return response, traces, gen_ids


def split_micro_actions(text: str, min_chars: int, max_chars: int) -> List[Tuple[int, int, str]]:
    """Split response text into comma-level reasoning action spans."""
    spans: List[Tuple[int, int, str]] = []
    boundaries = set()
    start = 0

    boundary_patterns = [
        r"(?<!\d)[,，](?!\d)",
        r"[\n;；]",
        r"(?<!\d\.)(?<=[.!?。！？])\s+",
        r"(?=\n?\s*(?:Step\s*)?\d+[.)]\s+)",
        r"(?i)(?=\b(?:therefore|however|so|if|then|because|but|thus|hence|next|now|check|conclude|finally|let's|let us)\b)",
    ]
    for pattern in boundary_patterns:
        for match in re.finditer(pattern, text):
            boundary = match.end() if match.end() > match.start() else match.start()
            if 0 < boundary < len(text):
                boundaries.add(boundary)

    for boundary in sorted(boundaries | {len(text)}):
        chunk_start, chunk_end = start, boundary
        if chunk_end <= chunk_start:
            continue
        raw = text[chunk_start:chunk_end]
        stripped = raw.strip()
        if stripped:
            lead = len(raw) - len(raw.lstrip())
            trail = len(raw.rstrip())
            display_text = stripped.rstrip(",，;；").strip() or stripped
            spans.append((chunk_start + lead, chunk_start + trail, display_text))
        start = boundary

    singleton_connectives = {
        "so",
        "then",
        "but",
        "however",
        "therefore",
        "thus",
        "hence",
        "if",
        "because",
        "next",
        "now",
        "check",
        "conclude",
        "finally",
    }
    connective_merged: List[Tuple[int, int, str]] = []
    idx = 0
    while idx < len(spans):
        s, e, span_text = spans[idx]
        normalized = span_text.strip().lower().rstrip(",，;；")
        if normalized in singleton_connectives and idx + 1 < len(spans):
            _, ne, _ = spans[idx + 1]
            merged_text = text[s:ne].strip().rstrip(",，;；").strip()
            connective_merged.append((s, ne, merged_text or text[s:ne].strip()))
            idx += 2
        else:
            connective_merged.append((s, e, span_text))
            idx += 1
    spans = connective_merged

    protected_short_prefixes = (
        "so",
        "then",
        "but",
        "however",
        "therefore",
        "thus",
        "hence",
        "if",
        "because",
        "next",
        "now",
        "check",
        "conclude",
        "finally",
    )

    def is_protected_short_span(span_text: str) -> bool:
        normalized = span_text.strip().lower()
        return any(normalized == prefix or normalized.startswith(prefix + " ") for prefix in protected_short_prefixes)

    merged: List[Tuple[int, int, str]] = []
    for span in spans:
        span_text = span[2]
        is_answer_span = "answer:" in span_text.lower()
        if merged and len(span_text) < min_chars and not is_answer_span and not is_protected_short_span(span_text):
            ps, _, _ = merged[-1]
            _, ne, _ = span
            merged_text = text[ps:ne].strip().rstrip(",，;；").strip()
            merged[-1] = (ps, ne, merged_text or text[ps:ne].strip())
        else:
            merged.append(span)

    final_spans: List[Tuple[int, int, str]] = []
    for s, e, t in merged:
        if len(t) <= max_chars:
            final_spans.append((s, e, t.rstrip(",，;；").strip() or t))
            continue
        local_start = s
        for match in re.finditer(r"(?<=[.!?。！？])\s+", text[s:e]):
            local_end = s + match.end()
            piece = text[local_start:local_end].strip()
            if piece:
                final_spans.append((local_start, local_end, piece.rstrip(",，;；").strip() or piece))
            local_start = local_end
        if local_start < e:
            piece = text[local_start:e].strip()
            if piece:
                final_spans.append((local_start, e, piece.rstrip(",，;；").strip() or piece))
    return final_spans


def apply_token_length_cap(
    spans: List[Tuple[int, int, str]],
    traces: List[TokenTrace],
    max_action_tokens: int,
) -> List[Tuple[int, int, str]]:
    if max_action_tokens <= 0:
        return spans

    capped: List[Tuple[int, int, str]] = []
    for char_start, char_end, action_text in spans:
        token_indices = [
            tr.token_index
            for tr in traces
            if tr.char_end > char_start and tr.char_start < char_end and tr.char_end > tr.char_start
        ]
        if len(token_indices) <= max_action_tokens:
            capped.append((char_start, char_end, action_text))
            continue
        for offset in range(0, len(token_indices), max_action_tokens):
            chunk = token_indices[offset : offset + max_action_tokens]
            if not chunk:
                continue
            s = max(char_start, traces[chunk[0]].char_start)
            e = min(char_end, traces[chunk[-1]].char_end)
            piece = "".join(traces[i].text for i in chunk).strip()
            if piece:
                capped.append((s, e, piece.rstrip(",，;；").strip() or piece))
    return capped


def median_abs_deviation(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    med = statistics.median(values)
    return statistics.median([abs(v - med) for v in values])


def score_actions(
    text: str,
    traces: List[TokenTrace],
    min_chars: int,
    max_chars: int,
    max_action_tokens: int,
    eps: float,
    min_robust_denom: float,
) -> List[ActionSpan]:
    spans = split_micro_actions(text, min_chars=min_chars, max_chars=max_chars)
    spans = apply_token_length_cap(spans, traces=traces, max_action_tokens=max_action_tokens)

    actions: List[ActionSpan] = []
    for action_index, (char_start, char_end, action_text) in enumerate(spans):
        token_indices = [
            tr.token_index
            for tr in traces
            if tr.char_end > char_start and tr.char_start < char_end and tr.char_end > tr.char_start
        ]
        entropies = [traces[i].entropy for i in token_indices]
        if entropies:
            r = max(1, math.ceil(math.sqrt(len(entropies))))
            top_vals = sorted(entropies, reverse=True)[:r]
            top_r_mean = sum(top_vals) / len(top_vals)
            token_start = min(token_indices)
            token_end = max(token_indices)
        else:
            top_r_mean = 0.0
            token_start = None
            token_end = None
        actions.append(
            ActionSpan(
                action_index=action_index,
                text=action_text,
                char_start=char_start,
                char_end=char_end,
                token_indices=token_indices,
                token_start=token_start,
                token_end=token_end,
                token_count=len(token_indices),
                entropy_topr_mean=top_r_mean,
                entropy_robust_z=0.0,
            )
        )

    values = [a.entropy_topr_mean for a in actions]
    med = statistics.median(values) if values else 0.0
    mad = median_abs_deviation(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    denom = max(mad, std if mad < min_robust_denom else 0.0, min_robust_denom, eps)
    for action in actions:
        action.entropy_robust_z = (action.entropy_topr_mean - med) / denom
    return actions


def local_peak_indices(actions: Sequence[ActionSpan], min_anchor_z: float) -> List[int]:
    peaks: List[int] = []
    for i, action in enumerate(actions):
        z = action.entropy_robust_z
        if z < min_anchor_z:
            continue
        left = actions[i - 1].entropy_robust_z if i > 0 else float("-inf")
        right = actions[i + 1].entropy_robust_z if i + 1 < len(actions) else float("-inf")
        if z >= left and z > right:
            peaks.append(i)
    return peaks


def aggregate_candidate_blocks(
    actions: Sequence[ActionSpan],
    candidate_action_ids: Sequence[int],
    top_blocks: int,
    min_anchor_z: float,
) -> List[CandidateBlock]:
    candidate_set = set(candidate_action_ids)
    peaks = [
        idx
        for idx in local_peak_indices(actions, min_anchor_z=min_anchor_z)
        if actions[idx].action_index in candidate_set
    ]
    peaks = sorted(peaks, key=lambda idx: actions[idx].entropy_robust_z, reverse=True)

    blocks: List[CandidateBlock] = []
    used_action_ids = set()
    for anchor in peaks:
        left = anchor
        right = anchor
        if anchor - 1 >= 0 and actions[anchor - 1].action_index in candidate_set:
            left = anchor - 1
        if anchor + 1 < len(actions) and actions[anchor + 1].action_index in candidate_set:
            right = anchor + 1

        block_actions = list(actions[left : right + 1])
        block_ids = [a.action_index for a in block_actions]
        if any(action_id in used_action_ids for action_id in block_ids):
            continue

        z_values = [a.entropy_robust_z for a in block_actions]
        token_indices = [idx for action in block_actions for idx in action.token_indices]
        token_start = min(token_indices) if token_indices else None
        token_end = max(token_indices) if token_indices else None
        char_start = min(a.char_start for a in block_actions)
        char_end = max(a.char_end for a in block_actions)
        text = " ".join(a.text.strip() for a in block_actions if a.text.strip())
        blocks.append(
            CandidateBlock(
                block_index=len(blocks),
                anchor_action_id=actions[anchor].action_index,
                action_ids=block_ids,
                action_start=block_ids[0],
                action_end=block_ids[-1],
                text=text,
                char_start=char_start,
                char_end=char_end,
                token_start=token_start,
                token_end=token_end,
                token_count=len(token_indices),
                anchor_robust_E=actions[anchor].entropy_robust_z,
                block_mean_robust_E=statistics.mean(z_values) if z_values else 0.0,
                block_max_robust_E=max(z_values) if z_values else 0.0,
            )
        )
        used_action_ids.update(block_ids)
        if len(blocks) >= top_blocks:
            break
    return blocks


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_review_md(out_dir: Path, records: List[Dict[str, Any]], limit: int) -> None:
    def ff(value: Any) -> str:
        try:
            return f"{float(value):.3f}"
        except Exception:
            return str(value)

    lines = [
        "# RCPC Anchor and Block Probe Review",
        "",
        "This file is generated by `scripts/run_anchor_block_probe.py`.",
        "",
    ]
    shown = 0
    for rec in records:
        if shown >= limit:
            break
        if "error" in rec:
            continue
        shown += 1
        lines += [f"## Sample {shown}: `{rec['sample_id']}` ({rec['source']})", ""]
        question = (rec.get("question") or "").strip()
        if len(question) > 1200:
            question = question[:1200].rstrip() + " ..."
        lines += ["**Question**", "", "```text", question, "```", ""]
        if rec.get("gold_answer"):
            lines += [f"**Gold answer:** `{rec['gold_answer']}`", ""]
        lines += ["**Generated reasoning**", "", "```text", (rec.get("response") or "").strip(), "```", ""]
        lines += ["**Top Candidate Actions**", ""]
        for rank, action in enumerate(rec.get("top_actions", []), 1):
            text = (action.get("text") or "").strip().replace("\n", " / ")
            lines.append(
                f"{rank}. `action_id={action.get('action_index')}` · "
                f"`robust_E={ff(action.get('entropy_robust_z'))}` · "
                f"`raw_E={ff(action.get('entropy_topr_mean'))}` · "
                f"`tokens={action.get('token_count')}` · "
                f"`token_span={action.get('token_start')}:{action.get('token_end')}`"
            )
            lines.append(f"   - {text}")
        lines.append("")
        lines += ["**Peak-Centered Candidate Blocks**", ""]
        blocks = rec.get("candidate_blocks", [])
        if not blocks:
            lines.append("_No block selected._")
        for rank, block in enumerate(blocks, 1):
            text = (block.get("text") or "").strip().replace("\n", " / ")
            lines.append(
                f"{rank}. `anchor_action={block.get('anchor_action_id')}` · "
                f"`actions={block.get('action_ids')}` · "
                f"`anchor_robust_E={ff(block.get('anchor_robust_E'))}` · "
                f"`mean_robust_E={ff(block.get('block_mean_robust_E'))}` · "
                f"`tokens={block.get('token_count')}` · "
                f"`token_span={block.get('token_start')}:{block.get('token_end')}`"
            )
            lines.append(f"   - {text}")
        lines.append("")
    (out_dir / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(out_dir: Path, records: List[Dict[str, Any]], top_actions: int, review_limit: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "results.jsonl", records)

    action_rows: List[Dict[str, Any]] = []
    block_rows: List[Dict[str, Any]] = []
    for rec in records:
        if "error" in rec:
            continue
        for rank, action in enumerate(rec.get("top_actions", [])[:top_actions], 1):
            row = {
                "sample_id": rec["sample_id"],
                "source": rec["source"],
                "rank": rank,
                **action,
            }
            action_rows.append(row)
        for rank, block in enumerate(rec.get("candidate_blocks", []), 1):
            row = {
                "sample_id": rec["sample_id"],
                "source": rec["source"],
                "rank": rank,
                **block,
            }
            block_rows.append(row)

    write_jsonl(out_dir / "top_actions.jsonl", action_rows)
    write_jsonl(out_dir / "candidate_blocks.jsonl", block_rows)
    write_csv(
        out_dir / "top_actions.csv",
        action_rows,
        [
            "sample_id",
            "source",
            "rank",
            "action_index",
            "text",
            "char_start",
            "char_end",
            "token_start",
            "token_end",
            "token_count",
            "entropy_topr_mean",
            "entropy_robust_z",
            "token_indices",
        ],
    )
    write_csv(
        out_dir / "candidate_blocks.csv",
        block_rows,
        [
            "sample_id",
            "source",
            "rank",
            "block_index",
            "anchor_action_id",
            "action_ids",
            "action_start",
            "action_end",
            "text",
            "char_start",
            "char_end",
            "token_start",
            "token_end",
            "token_count",
            "anchor_robust_E",
            "block_mean_robust_E",
            "block_max_robust_E",
        ],
    )
    write_review_md(out_dir, records, limit=review_limit)

    stats = {
        "records": len(records),
        "errors": sum(1 for rec in records if "error" in rec),
        "top_action_rows": len(action_rows),
        "candidate_block_rows": len(block_rows),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local or Hugging Face model path, for example Qwen/Qwen3-0.6B.")
    parser.add_argument("--data-path", required=True, help="JSONL/JSON/CSV data path.")
    parser.add_argument("--out-dir", default="runs/anchor_block_probe")
    parser.add_argument("--num-samples", type=int, default=50, help="Use <=0 to run all samples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")

    parser.add_argument("--id-field", default="id")
    parser.add_argument("--source-field", default="source")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")

    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true")

    parser.add_argument("--top-actions", type=int, default=12)
    parser.add_argument("--top-blocks", type=int, default=6)
    parser.add_argument("--min-action-chars", type=int, default=12)
    parser.add_argument("--max-action-chars", type=int, default=260)
    parser.add_argument("--max-action-tokens", type=int, default=24)
    parser.add_argument("--min-robust-denom", type=float, default=0.05)
    parser.add_argument("--min-anchor-z", type=float, default=0.5)
    parser.add_argument("--review-limit", type=int, default=30)
    parser.add_argument(
        "--prompt-template",
        default=(
            "Solve the following reasoning problem. Provide concise step-by-step reasoning. "
            "Each step should contain one evidence link, condition check, option elimination, "
            "or causal/logical decision when possible. End with a final line 'Answer: ...'.\n\n"
            "Problem:\n{question}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)
    samples = load_samples(
        data_path=data_path,
        num_samples=args.num_samples,
        seed=args.seed,
        id_field=args.id_field,
        source_field=args.source_field,
        question_field=args.question_field,
        answer_field=args.answer_field,
        shuffle=args.shuffle,
    )
    if not samples:
        raise RuntimeError(f"No samples loaded from {data_path}")

    device = choose_device(args.device)
    print(f"[data] loaded {len(samples)} samples from {data_path}")
    print(f"[model] loading {args.model_path} on {device}")
    model, tokenizer = load_model(args.model_path, device, args.dtype, args.trust_remote_code)

    records: List[Dict[str, Any]] = []
    for sample in tqdm(samples, desc="RCPC probe"):
        prompt = build_prompt(sample, args.prompt_template)
        try:
            inputs = prepare_inputs(prompt, tokenizer, device)
            response, traces, generated_token_ids = generate_with_entropy(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=not args.greedy,
            )
            actions = score_actions(
                response,
                traces,
                min_chars=args.min_action_chars,
                max_chars=args.max_action_chars,
                max_action_tokens=args.max_action_tokens,
                eps=1e-6,
                min_robust_denom=args.min_robust_denom,
            )
            top_actions = sorted(actions, key=lambda a: a.entropy_robust_z, reverse=True)[: args.top_actions]
            blocks = aggregate_candidate_blocks(
                actions,
                candidate_action_ids=[a.action_index for a in top_actions],
                top_blocks=args.top_blocks,
                min_anchor_z=args.min_anchor_z,
            )
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "source": sample.source,
                    "question": sample.question,
                    "gold_answer": sample.answer,
                    "prompt": prompt,
                    "response": response,
                    "generated_token_ids": generated_token_ids,
                    "tokens": [asdict(x) for x in traces],
                    "actions": [asdict(x) for x in actions],
                    "top_actions": [asdict(x) for x in top_actions],
                    "candidate_blocks": [asdict(x) for x in blocks],
                }
            )
        except Exception as exc:
            print(f"[warn] failed sample {sample.sample_id}: {type(exc).__name__}: {exc}")
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "source": sample.source,
                    "question": sample.question,
                    "gold_answer": sample.answer,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    write_outputs(out_dir, records, top_actions=args.top_actions, review_limit=args.review_limit)
    print(f"[out] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
