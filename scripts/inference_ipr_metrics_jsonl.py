import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForVision2Seq

from verl.utils import get_processor, get_tokenizer


ANSWER_RE = re.compile(r"<answer>\s*(Yes|No)\s*</answer>", flags=re.IGNORECASE)
ANSWER_OPEN_RE = re.compile(r"<answer[^>]*>", flags=re.IGNORECASE)
STEP_SECTION_PATTERNS = (
    ("step_1", re.compile(r"Step\s*1\s*:\s*Preconditions", flags=re.IGNORECASE)),
    ("step_2", re.compile(r"Step\s*2\s*:\s*Violation\s+Conditions", flags=re.IGNORECASE)),
    ("step_3", re.compile(r"Step\s*3\s*:\s*Exemptions", flags=re.IGNORECASE)),
    ("step_4", re.compile(r"\bConclusion\b", flags=re.IGNORECASE)),
)
FSDP_RANK0_RE = re.compile(r"model_world_size_(\d+)_rank_0\.pt")
HF_WEIGHT_PATTERNS = (
    "pytorch_model.bin",
    "model.safetensors",
    "tf_model.h5",
    "flax_model.msgpack",
    "model-*.safetensors",
    "pytorch_model-*.bin",
)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def maybe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except Exception:
        return value


def move_to_device(value: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Any:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and dtype is not None:
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    if isinstance(value, list):
        return [move_to_device(item, device, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device, dtype) for item in value)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device, dtype) for key, item in value.items()}
    return value


def process_image(image: Image.Image, max_pixels: int, min_pixels: int) -> Image.Image:
    if image.width * image.height > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize(
            (int(image.width * resize_factor), int(image.height * resize_factor)),
            resample=Image.Resampling.NEAREST,
        )
    if image.width * image.height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        image = image.resize(
            (int(image.width * resize_factor), int(image.height * resize_factor)),
            resample=Image.Resampling.NEAREST,
        )
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def choose_model_class(config: Any) -> Any:
    if getattr(config, "model_type", None) == "valley":
        return AutoModel
    if type(config) in AutoModelForVision2Seq._model_mapping.keys():
        return AutoModelForVision2Seq
    return AutoModelForCausalLM


def has_config(path: Path) -> bool:
    return (path / "config.json").is_file()


def has_hf_weights(path: Path) -> bool:
    return any(glob.glob(str(path / pattern)) for pattern in HF_WEIGHT_PATTERNS)


def has_fsdp_shards(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(FSDP_RANK0_RE.match(item.name) for item in path.iterdir() if item.is_file())


def find_actor_dir(path: Path) -> Optional[Path]:
    if path.name == "huggingface" and path.parent.is_dir():
        parent = path.parent
        if has_fsdp_shards(parent):
            return parent
    if path.name == "actor" and path.is_dir():
        return path
    actor_dir = path / "actor"
    if actor_dir.is_dir():
        return actor_dir
    return None


def run_model_merger(actor_dir: Path, hf_dir: Path) -> None:
    merger_script = Path(__file__).resolve().parent / "model_merger.py"
    if not merger_script.is_file():
        raise FileNotFoundError(f"model merger script not found: {merger_script}")

    cmd = [
        sys.executable,
        str(merger_script),
        "--local_dir",
        str(actor_dir),
        "--output_dir",
        str(hf_dir),
    ]
    print(f"[INFO] running model merger: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(merger_script.parent.parent))


def maybe_merge_actor_checkpoint(actor_dir: Path, hf_dir: Path, auto_merge: bool) -> None:
    if not has_fsdp_shards(actor_dir):
        return

    lock_dir = actor_dir / ".merge_huggingface.lock"
    acquired_lock = False

    while True:
        if lock_dir.exists():
            try:
                lock_age = time.time() - lock_dir.stat().st_mtime
            except FileNotFoundError:
                continue
            if lock_age > 6 * 3600:
                print(f"[WARN] removing stale merge lock: {lock_dir}", flush=True)
                try:
                    lock_dir.rmdir()
                except OSError:
                    pass
                continue
            print(f"[INFO] waiting for another process to finish checkpoint merge: {lock_dir}", flush=True)
            time.sleep(10)
            continue

        if has_hf_weights(hf_dir):
            return

        if not auto_merge:
            command = f"python3 scripts/model_merger.py --local_dir {actor_dir} --output_dir {hf_dir}"
            raise RuntimeError(
                "Found a verl/FSDP actor checkpoint, but its HuggingFace directory has no merged model weights. "
                "Run the merger first, or rerun this script with --auto_merge_checkpoint.\n"
                f"Suggested command:\n{command}"
            )

        try:
            lock_dir.mkdir()
            acquired_lock = True
            break
        except FileExistsError:
            continue

    if not acquired_lock:
        return

    try:
        if not has_hf_weights(hf_dir):
            print(f"[INFO] merging FSDP actor shards from {actor_dir} into {hf_dir}", flush=True)
            run_model_merger(actor_dir, hf_dir)
        if not has_hf_weights(hf_dir):
            raise RuntimeError(f"merge finished but no HuggingFace model weights were found in {hf_dir}")
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass


def resolve_model_path(model_path: str, auto_merge: bool) -> str:
    raw_path = Path(model_path).expanduser()
    if not raw_path.exists():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")

    candidates = []
    actor_dir = find_actor_dir(raw_path)
    if actor_dir is not None:
        candidates.append(actor_dir / "huggingface")
        candidates.append(actor_dir)
    candidates.extend([raw_path / "huggingface", raw_path])

    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered_candidates.append(candidate)

    for candidate in ordered_candidates:
        if not candidate.is_dir() or not has_config(candidate):
            continue
        candidate_actor_dir = find_actor_dir(candidate) or actor_dir
        if candidate_actor_dir is not None:
            maybe_merge_actor_checkpoint(candidate_actor_dir, candidate, auto_merge)
        if has_hf_weights(candidate):
            if candidate != raw_path:
                print(f"[INFO] resolved MODEL_PATH {raw_path} -> {candidate}", flush=True)
            return str(candidate)

    inspected = "\n".join(f"- {candidate}" for candidate in ordered_candidates)
    raise RuntimeError(
        "Could not resolve MODEL_PATH to a loadable HuggingFace model directory. "
        "For verl checkpoints, pass either global_step_xxx, global_step_xxx/actor, "
        "or global_step_xxx/actor/huggingface after merging shards.\n"
        f"Inspected candidates:\n{inspected}"
    )


def decode_text(decoder: Any, tokenizer: Any, token_ids: torch.Tensor) -> str:
    decode_owner = decoder if hasattr(decoder, "batch_decode") else tokenizer
    try:
        return decode_owner.batch_decode(token_ids, skip_special_tokens=False)[0]
    except TypeError:
        return decode_owner.batch_decode(token_ids)[0]


def build_inputs(
    row: Dict[str, Any],
    tokenizer: Any,
    processor: Any,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
    prompt_key: str,
    max_pixels: int,
    min_pixels: int,
) -> Tuple[Dict[str, Any], int, List[str]]:
    images = row.get("image", row.get("images"))
    image_paths = [item for item in as_list(images) if isinstance(item, str) and item.strip()]
    prompt_text = str(row[prompt_key])

    if getattr(config, "model_type", None) == "valley":
        processed = processor(
            {
                "conversations": [{"role": "user", "content": prompt_text}],
                "images": image_paths if image_paths else None,
            }
        )
        input_ids = processed["input_ids"].to(device)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.long, device=device),
            "images": move_to_device(processed.get("images"), device, dtype),
            "image_sizes": processed.get("image_sizes"),
            "pixel_values": move_to_device(processed.get("pixel_values"), device, dtype),
            "pixel_values_videos": move_to_device(processed.get("pixel_values_videos"), device, dtype),
            "image_grid_thw": move_to_device(processed.get("image_grid_thw"), device),
            "video_grid_thw": move_to_device(processed.get("video_grid_thw"), device),
        }
        model_inputs = {key: value for key, value in model_inputs.items() if value is not None}
        return model_inputs, input_ids.shape[-1], image_paths

    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    model_inputs: Dict[str, Any] = {}
    if image_paths:
        pil_images = [process_image(Image.open(path), max_pixels, min_pixels) for path in image_paths]
        image_inputs = processor.image_processor(pil_images, return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        merge_length = processor.image_processor.merge_size**2
        image_index = 0
        while "<image>" in prompt and image_index < len(image_grid_thw):
            prompt = prompt.replace(
                "<image>",
                "<|vision_start|>"
                + "<|placeholder|>" * int(image_grid_thw[image_index].prod().item() // merge_length)
                + "<|vision_end|>",
                1,
            )
            image_index += 1
        prompt = prompt.replace("<|placeholder|>", processor.image_token)
        model_inputs.update(move_to_device(dict(image_inputs), device, dtype))

    input_ids = torch.tensor(
        tokenizer.encode(prompt, add_special_tokens=False),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    model_inputs["input_ids"] = input_ids
    model_inputs["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return model_inputs, input_ids.shape[-1], image_paths


def _squeeze_input_sequence(value: torch.Tensor, key: str) -> torch.Tensor:
    if value.dim() == 2 and value.size(0) == 1:
        value = value[0]
    if value.dim() != 1:
        raise ValueError(f"{key} must have shape [seq_len] or [1, seq_len], got {tuple(value.shape)}")
    return value


def _flatten_tensor_values(values: List[Any], key: str) -> torch.Tensor:
    tensors = []
    for value in values:
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, (list, tuple)):
            tensors.extend(item for item in value if isinstance(item, torch.Tensor))
        else:
            raise TypeError(f"cannot batch {key}: unsupported value type {type(value).__name__}")
    if not tensors:
        raise ValueError(f"cannot batch {key}: no tensors found")
    return torch.cat(tensors, dim=0)


def _unwrap_singleton(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def collate_model_inputs(model_inputs_list: List[Dict[str, Any]], tokenizer: Any) -> Dict[str, Any]:
    if not model_inputs_list:
        raise ValueError("cannot collate an empty input batch")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id for batched inference")

    input_ids_list = [
        _squeeze_input_sequence(model_inputs["input_ids"], "input_ids") for model_inputs in model_inputs_list
    ]
    attention_mask_list = []
    for model_inputs, input_ids in zip(model_inputs_list, input_ids_list):
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        else:
            attention_mask = _squeeze_input_sequence(attention_mask, "attention_mask")
        attention_mask_list.append(attention_mask)

    max_length = max(input_ids.size(0) for input_ids in input_ids_list)
    batch_input_ids = []
    batch_attention_mask = []
    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        pad_length = max_length - input_ids.size(0)
        batch_input_ids.append(
            torch.cat(
                [
                    torch.full(
                        (pad_length,),
                        pad_token_id,
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    ),
                    input_ids,
                ],
                dim=0,
            )
        )
        batch_attention_mask.append(
            torch.cat(
                [
                    torch.zeros(
                        (pad_length,),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                    attention_mask,
                ],
                dim=0,
            )
        )

    batch: Dict[str, Any] = {
        "input_ids": torch.stack(batch_input_ids, dim=0),
        "attention_mask": torch.stack(batch_attention_mask, dim=0),
    }

    concat_keys = ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw")
    list_keys = ("images", "image_sizes")
    for key in concat_keys + list_keys:
        values = [model_inputs.get(key) for model_inputs in model_inputs_list]
        present = [value is not None for value in values]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(f"cannot batch mixed samples where only some contain {key}")
        if len(model_inputs_list) == 1:
            batch[key] = values[0]
            continue
        if key in concat_keys:
            batch[key] = _flatten_tensor_values(values, key)
        else:
            batch[key] = [_unwrap_singleton(value) for value in values]

    return batch


def append_token_ids(model_inputs: Dict[str, Any], token_ids: List[int]) -> Dict[str, Any]:
    input_ids = _squeeze_input_sequence(model_inputs["input_ids"], "input_ids")
    suffix = torch.tensor(token_ids, dtype=input_ids.dtype, device=input_ids.device)

    appended = dict(model_inputs)
    appended["input_ids"] = torch.cat([input_ids, suffix], dim=0).unsqueeze(0)

    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    else:
        attention_mask = _squeeze_input_sequence(attention_mask, "attention_mask")
    suffix_mask = torch.ones(suffix.size(0), dtype=attention_mask.dtype, device=attention_mask.device)
    appended["attention_mask"] = torch.cat([attention_mask, suffix_mask], dim=0).unsqueeze(0)
    return appended


def normalize_yes_no(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        if int(value) == 1:
            return "Yes"
        if int(value) == 0:
            return "No"
        return None

    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"1", "yes", "true", "reject", "violation"}:
        return "Yes"
    if lowered in {"0", "no", "false", "approve", "no violation"}:
        return "No"

    match = ANSWER_RE.search(text)
    if match:
        return "Yes" if match.group(1).lower() == "yes" else "No"

    head = text.split("<", 1)[0].strip().lower()
    if head == "yes":
        return "Yes"
    if head == "no":
        return "No"
    return None


def infer_label(row: Dict[str, Any], target_key: str) -> Optional[str]:
    for key in ("label", "gt_label", "answer", target_key):
        label = normalize_yes_no(row.get(key))
        if label is not None:
            return label

    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict):
        label = normalize_yes_no(extra_info.get("final_label"))
        if label is not None:
            return label
    return None


def normalize_extra_info(row: Dict[str, Any]) -> Dict[str, Any]:
    extra_info = maybe_json_loads(row.get("extra_info", {}))
    if not isinstance(extra_info, dict):
        extra_info = {}

    for key in (
        "country",
        "reject_reason",
        "reject_reason_code",
        "product_audit_status",
        "train_data_type",
        "final_label",
        "human_audit_result",
        "machine_audit_result",
        "machine_agent_res",
        "fusion_model_res",
        "llm_studio_agent_result_map",
    ):
        if key in row and key not in extra_info:
            extra_info[key] = maybe_json_loads(row[key])

    if isinstance(extra_info.get("fusion_model_res"), str):
        extra_info["fusion_model_res"] = maybe_json_loads(extra_info["fusion_model_res"])
    if extra_info.get("fusion_model_res") is None:
        extra_info["fusion_model_res"] = {}

    return extra_info


def build_yes_no_pairs(tokenizer: Any) -> List[Dict[str, Any]]:
    text_pairs = [
        ("Yes", "No"),
        ("Yes</answer>", "No</answer>"),
        ("<answer>Yes", "<answer>No"),
        ("<answer> Yes", "<answer> No"),
        (">Yes", ">No"),
        (" yes", " no"),
        (" Yes", " No"),
        (" Yes</answer>", " No</answer>"),
        ("YES", "NO"),
        ("yes", "no"),
        ("\nYes", "\nNo"),
    ]
    pairs = []
    seen = set()
    for yes_text, no_text in text_pairs:
        yes_ids = tokenizer.encode(yes_text, add_special_tokens=False)
        no_ids = tokenizer.encode(no_text, add_special_tokens=False)
        if not yes_ids or not no_ids:
            continue
        pair = (yes_ids[0], no_ids[0])
        if pair[0] == pair[1]:
            continue
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(
            {
                "yes_id": pair[0],
                "no_id": pair[1],
                "yes_ids": yes_ids,
                "no_ids": no_ids,
                "yes_text": yes_text,
                "no_text": no_text,
            }
        )
    return pairs


def single_token_id(tokenizer: Any, text: str) -> int:
    tokenized = tokenizer.tokenize(text)
    if len(tokenized) == 1:
        return tokenizer.convert_tokens_to_ids(tokenized)[0]

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) == 1:
        return token_ids[0]

    raise ValueError(f"{text!r} is not a single token for this tokenizer: tokens={tokenized}, ids={token_ids}")


def answer_from_output(text: str) -> Optional[str]:
    match = ANSWER_RE.search(text)
    if match:
        return "Yes" if match.group(1).lower() == "yes" else "No"
    return normalize_yes_no(text)


def answer_prefix_before_label(text: str) -> Optional[str]:
    match = ANSWER_RE.search(text)
    if not match:
        return None
    return text[: match.start(1)]


def answer_prefix_yes_probability(
    *,
    model: Any,
    tokenizer: Any,
    model_inputs: Dict[str, Any],
    output_text: str,
    yes_id: int,
    no_id: int,
) -> Optional[float]:
    prefix_text = answer_prefix_before_label(output_text)
    if prefix_text is None:
        return None

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    if not prefix_ids:
        return None

    input_ids = model_inputs["input_ids"]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    batch_size = input_ids.size(0)
    prefix_tensor = torch.tensor(prefix_ids, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(0)
    if batch_size != 1:
        prefix_tensor = prefix_tensor.expand(batch_size, -1)

    scoring_inputs = dict(model_inputs)
    scoring_inputs["input_ids"] = torch.cat([input_ids, prefix_tensor], dim=1)

    attention_mask = scoring_inputs.get("attention_mask")
    if attention_mask is not None:
        prefix_mask = torch.ones(
            (batch_size, prefix_tensor.size(1)),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        scoring_inputs["attention_mask"] = torch.cat([attention_mask, prefix_mask], dim=1)
    else:
        scoring_inputs["attention_mask"] = torch.ones_like(scoring_inputs["input_ids"], dtype=torch.long)

    outputs = model(
        **scoring_inputs,
        use_cache=False,
        return_dict=True,
    )
    yes_no_logits = outputs.logits[:, -1, :][:, [yes_id, no_id]]
    yes_no_probs = torch.softmax(yes_no_logits.float(), dim=-1)
    return yes_no_probs[:, 0].detach().cpu().item()


def answer_prefix_yes_probabilities(
    *,
    model: Any,
    tokenizer: Any,
    model_inputs_list: List[Dict[str, Any]],
    output_texts: List[str],
    yes_id: int,
    no_id: int,
) -> List[Optional[float]]:
    probabilities: List[Optional[float]] = [None] * len(model_inputs_list)
    scoring_inputs_list = []
    scoring_indices = []

    for index, (model_inputs, output_text) in enumerate(zip(model_inputs_list, output_texts)):
        prefix_text = answer_prefix_before_label(output_text)
        if prefix_text is None:
            continue
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        if not prefix_ids:
            continue
        scoring_inputs_list.append(append_token_ids(model_inputs, prefix_ids))
        scoring_indices.append(index)

    if not scoring_inputs_list:
        return probabilities

    scoring_inputs = collate_model_inputs(scoring_inputs_list, tokenizer)
    outputs = model(
        **scoring_inputs,
        use_cache=False,
        return_dict=True,
    )
    yes_no_logits = outputs.logits[:, -1, :][:, [yes_id, no_id]]
    yes_no_probs = torch.softmax(yes_no_logits.float(), dim=-1)[:, 0].detach().cpu().tolist()
    for index, probability in zip(scoring_indices, yes_no_probs):
        probabilities[index] = float(probability)
    return probabilities


def next_token_yes_probability(
    *,
    model: Any,
    model_inputs: Dict[str, Any],
    yes_id: int,
    no_id: int,
) -> float:
    input_ids = model_inputs["input_ids"]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    batch_size = input_ids.size(0)
    yes_col = torch.full((batch_size, 1), yes_id, dtype=input_ids.dtype, device=input_ids.device)
    scoring_inputs = dict(model_inputs)
    scoring_inputs["input_ids"] = torch.cat([input_ids, yes_col], dim=1)

    attention_mask = scoring_inputs.get("attention_mask")
    if attention_mask is not None:
        one_col = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        scoring_inputs["attention_mask"] = torch.cat([attention_mask, one_col], dim=1)
    else:
        scoring_inputs["attention_mask"] = torch.ones_like(scoring_inputs["input_ids"], dtype=torch.long)

    outputs = model(
        **scoring_inputs,
        use_cache=False,
        return_dict=True,
    )
    logits_for_yes_position = outputs.logits[:, -2, :]
    yes_no_logits = logits_for_yes_position[:, [yes_id, no_id]]
    yes_no_probs = torch.softmax(yes_no_logits.float(), dim=-1)
    return yes_no_probs[:, 0].detach().cpu().item()


def next_token_yes_probabilities(
    *,
    model: Any,
    tokenizer: Any,
    model_inputs_list: List[Dict[str, Any]],
    yes_id: int,
    no_id: int,
) -> List[float]:
    scoring_inputs_list = [append_token_ids(model_inputs, [yes_id]) for model_inputs in model_inputs_list]
    scoring_inputs = collate_model_inputs(scoring_inputs_list, tokenizer)
    outputs = model(
        **scoring_inputs,
        use_cache=False,
        return_dict=True,
    )
    logits_for_yes_position = outputs.logits[:, -2, :]
    yes_no_logits = logits_for_yes_position[:, [yes_id, no_id]]
    yes_no_probs = torch.softmax(yes_no_logits.float(), dim=-1)[:, 0]
    return [float(value) for value in yes_no_probs.detach().cpu().tolist()]


def score_yes_no_from_step(step_scores: torch.Tensor, yes_no_pairs: List[Dict[str, Any]]) -> Optional[float]:
    if not yes_no_pairs:
        return None
    if step_scores.dim() == 2:
        step_scores = step_scores[0]

    yes_logits = []
    no_logits = []
    vocab_size = step_scores.shape[-1]
    for pair in yes_no_pairs:
        yes_id = pair["yes_id"]
        no_id = pair["no_id"]
        if yes_id >= vocab_size or no_id >= vocab_size:
            continue
        yes_logits.append(step_scores[yes_id])
        no_logits.append(step_scores[no_id])
    if not yes_logits or not no_logits:
        return None

    pair_logits = torch.stack([torch.stack(yes_logits).max(), torch.stack(no_logits).max()])
    return torch.softmax(pair_logits.float(), dim=-1)[0].item()


def pairs_matching_token(token_id: int, yes_no_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs = [pair for pair in yes_no_pairs if token_id in {pair["yes_id"], pair["no_id"]}]
    return pairs or yes_no_pairs


def text_after_answer_open(text: str) -> Optional[str]:
    match = ANSWER_OPEN_RE.search(text)
    if not match:
        return None
    return text[match.end() :]


def find_answer_token_index(tokenizer: Any, generated_ids_list: List[int]) -> Optional[int]:
    for token_index in range(len(generated_ids_list)):
        prefix_text = tokenizer.decode(generated_ids_list[:token_index], skip_special_tokens=False)
        current_text = tokenizer.decode(generated_ids_list[: token_index + 1], skip_special_tokens=False)

        prefix_tail = text_after_answer_open(prefix_text)
        current_tail = text_after_answer_open(current_text)
        if current_tail is None:
            continue

        if prefix_tail is None:
            if current_tail.strip():
                return token_index
            continue

        if current_tail != prefix_tail and current_tail.strip():
            return token_index
    return None


def yes_probability_from_generation(
    *,
    tokenizer: Any,
    generated_ids: torch.Tensor,
    scores: Any,
    output_text: str,
    parsed_answer: Optional[str],
    yes_no_pairs: List[Dict[str, Any]],
) -> Tuple[float, str]:
    if not scores or generated_ids.numel() == 0 or not yes_no_pairs:
        return fallback_yes_probability(parsed_answer)

    generated_ids_list = generated_ids.detach().cpu().tolist()
    lower_output = output_text.lower()
    has_answer_tag = "<answer" in lower_output

    answer_token_index = find_answer_token_index(tokenizer, generated_ids_list) if has_answer_tag else None
    if answer_token_index is not None and answer_token_index < len(scores):
        token_id = generated_ids_list[answer_token_index]
        yes_prob = score_yes_no_from_step(scores[answer_token_index], pairs_matching_token(token_id, yes_no_pairs))
        if yes_prob is not None:
            return yes_prob, "answer_token_score"

    for token_index, token_id in enumerate(generated_ids_list):
        matching_pair = None
        for pair in yes_no_pairs:
            if token_id in {pair["yes_id"], pair["no_id"]}:
                matching_pair = pair
                break
        if matching_pair is None:
            continue

        prefix_text = tokenizer.decode(generated_ids_list[:token_index], skip_special_tokens=False)
        after_answer_open = "<answer" in prefix_text.lower()
        if has_answer_tag and not after_answer_open:
            continue

        if parsed_answer == "Yes" and token_id != matching_pair["yes_id"]:
            continue
        if parsed_answer == "No" and token_id != matching_pair["no_id"]:
            continue

        yes_prob = score_yes_no_from_step(scores[token_index], [matching_pair])
        if yes_prob is not None:
            return yes_prob, "generation_score"

    return fallback_yes_probability(parsed_answer)


def fallback_yes_probability(parsed_answer: Optional[str]) -> Tuple[float, str]:
    if parsed_answer == "Yes":
        return 1.0, "parsed_answer_fallback"
    if parsed_answer == "No":
        return 0.0, "parsed_answer_fallback"
    return 0.5, "missing_answer_fallback"


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def count_jsonl_rows(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def trim_generated_ids_after_eos(generated_ids: torch.Tensor, eos_token_id: Any) -> torch.Tensor:
    eos_ids = as_list(eos_token_id)
    if not eos_ids:
        return generated_ids
    eos_set = {int(token_id) for token_id in eos_ids}
    for index, token_id in enumerate(generated_ids.detach().cpu().tolist()):
        if token_id in eos_set:
            return generated_ids[: index + 1]
    return generated_ids


def percentile(values: List[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def token_index_for_text_prefix(tokenizer: Any, text: str, char_index: int, max_tokens: int) -> int:
    prefix_ids = tokenizer.encode(text[:char_index], add_special_tokens=False)
    return min(max(len(prefix_ids), 0), max_tokens)


def parse_step_token_spans(tokenizer: Any, generated_ids: torch.Tensor, output_text: str) -> Dict[str, Tuple[int, int]]:
    matches = []
    search_start = 0
    for name, pattern in STEP_SECTION_PATTERNS:
        match = pattern.search(output_text, search_start)
        if match is None:
            return {}
        matches.append((name, match))
        search_start = match.end()

    answer_match = ANSWER_OPEN_RE.search(output_text, matches[-1][1].end())
    final_char_end = answer_match.start() if answer_match is not None else len(output_text)
    max_tokens = generated_ids.numel()
    spans = {}
    for index, (name, match) in enumerate(matches):
        char_start = match.end()
        char_end = matches[index + 1][1].start() if index + 1 < len(matches) else final_char_end
        token_start = token_index_for_text_prefix(tokenizer, output_text, char_start, max_tokens)
        token_end = token_index_for_text_prefix(tokenizer, output_text, char_end, max_tokens)
        spans[name] = (min(token_start, token_end), max(token_start, token_end))
    return spans


def compute_token_uncertainty_metrics(
    *,
    generated_ids: torch.Tensor,
    scores: Any,
    chunk_size: int,
) -> Dict[str, List[float]]:
    usable_tokens = min(generated_ids.numel(), len(scores))
    metrics = {"entropy": [], "chosen_nll": [], "logit_margin": []}
    if usable_tokens == 0:
        return metrics

    token_ids = generated_ids[:usable_tokens]
    for chunk_start in range(0, usable_tokens, chunk_size):
        chunk_end = min(chunk_start + chunk_size, usable_tokens)
        logits = torch.stack(
            [
                scores[index][0] if scores[index].dim() == 2 else scores[index]
                for index in range(chunk_start, chunk_end)
            ],
            dim=0,
        ).float()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1) / math.log(logits.size(-1))
        chosen_ids = token_ids[chunk_start:chunk_end].to(device=logits.device, dtype=torch.long).unsqueeze(-1)
        chosen_nll = -log_probs.gather(dim=-1, index=chosen_ids).squeeze(-1)
        top_two = torch.topk(logits, k=2, dim=-1).values
        logit_margin = top_two[:, 0] - top_two[:, 1]

        metrics["entropy"].extend(float(value) for value in entropy.detach().cpu().tolist())
        metrics["chosen_nll"].extend(float(value) for value in chosen_nll.detach().cpu().tolist())
        metrics["logit_margin"].extend(float(value) for value in logit_margin.detach().cpu().tolist())
        del logits, log_probs, probs, entropy, chosen_nll, top_two, logit_margin
    return metrics


def repeated_ngram_ratio(token_ids: List[int], ngram_size: int) -> float:
    if ngram_size <= 0 or len(token_ids) < ngram_size:
        return 0.0
    ngrams = [tuple(token_ids[index : index + ngram_size]) for index in range(len(token_ids) - ngram_size + 1)]
    return float((len(ngrams) - len(set(ngrams))) / len(ngrams))


def contains_eos_token(generated_ids: torch.Tensor, eos_token_id: Any) -> bool:
    eos_ids = {int(token_id) for token_id in as_list(eos_token_id)}
    if not eos_ids:
        return False
    return any(int(token_id) in eos_ids for token_id in generated_ids.detach().cpu().tolist())


def extract_confidence_features(
    *,
    tokenizer: Any,
    generated_ids: torch.Tensor,
    raw_generated_ids: torch.Tensor,
    scores: Any,
    output_text: str,
    max_new_tokens: int,
    low_margin_threshold: float,
    repeated_ngram_size: int,
    logit_chunk_size: int,
) -> Dict[str, Any]:
    spans = parse_step_token_spans(tokenizer, generated_ids, output_text)
    token_metrics = compute_token_uncertainty_metrics(
        generated_ids=generated_ids,
        scores=scores,
        chunk_size=logit_chunk_size,
    )
    usable_tokens = len(token_metrics["entropy"])

    features: Dict[str, Any] = {
        "schema_version": 1,
        "step_parse_ok": len(spans) == len(STEP_SECTION_PATTERNS),
        "low_margin_threshold": float(low_margin_threshold),
        "repeated_ngram_size": int(repeated_ngram_size),
    }
    step_entropy_means = {}
    step_margin_means = {}
    for step_name, _ in STEP_SECTION_PATTERNS:
        token_start, token_end = spans.get(step_name, (0, 0))
        token_start = min(token_start, usable_tokens)
        token_end = min(token_end, usable_tokens)
        entropy_values = token_metrics["entropy"][token_start:token_end]
        chosen_nll_values = token_metrics["chosen_nll"][token_start:token_end]
        margin_values = token_metrics["logit_margin"][token_start:token_end]

        entropy_mean = mean_or_none(entropy_values)
        margin_mean = mean_or_none(margin_values)
        step_entropy_means[step_name] = entropy_mean
        step_margin_means[step_name] = margin_mean
        features[f"{step_name}_entropy_mean"] = entropy_mean
        features[f"{step_name}_entropy_p90"] = percentile(entropy_values, 0.90)
        features[f"{step_name}_low_margin_ratio"] = (
            float(sum(value < low_margin_threshold for value in margin_values) / len(margin_values))
            if margin_values
            else None
        )
        features[f"{step_name}_chosen_nll_mean"] = mean_or_none(chosen_nll_values)
        features[f"{step_name}_chosen_nll_p90"] = percentile(chosen_nll_values, 0.90)
        features[f"{step_name}_logit_margin_p10"] = percentile(margin_values, 0.10)
        features[f"{step_name}_step_token_count"] = int(max(token_end - token_start, 0))

    first_entropy = step_entropy_means.get("step_1")
    final_entropy = step_entropy_means.get("step_4")
    first_margin = step_margin_means.get("step_1")
    final_margin = step_margin_means.get("step_4")
    features["step_entropy_change"] = (
        float(final_entropy - first_entropy) if first_entropy is not None and final_entropy is not None else None
    )
    features["step_margin_change"] = (
        float(final_margin - first_margin) if first_margin is not None and final_margin is not None else None
    )

    answer_match = ANSWER_OPEN_RE.search(output_text)
    reasoning_token_end = (
        token_index_for_text_prefix(tokenizer, output_text, answer_match.start(), generated_ids.numel())
        if answer_match is not None
        else generated_ids.numel()
    )
    reasoning_ids = generated_ids[:reasoning_token_end].detach().cpu().tolist()
    features["repeated_ngram_ratio"] = repeated_ngram_ratio(reasoning_ids, repeated_ngram_size)
    features["generation_truncated"] = bool(
        raw_generated_ids.numel() >= max_new_tokens
        and not contains_eos_token(raw_generated_ids, tokenizer.eos_token_id)
    )
    return features


def infer_prepared_batch(
    *,
    items: List[Dict[str, Any]],
    model: Any,
    tokenizer: Any,
    processor: Any,
    score_mode: str,
    yes_id: int,
    no_id: int,
    yes_no_pairs: List[Dict[str, Any]],
    repetition_penalty: float,
    max_new_tokens: int,
    collect_confidence_features: bool,
    low_margin_threshold: float,
    repeated_ngram_size: int,
    feature_logit_chunk_size: int,
) -> List[Dict[str, Any]]:
    model_inputs_list = [item["model_inputs"] for item in items]
    batched_inputs = collate_model_inputs(model_inputs_list, tokenizer)

    with torch.inference_mode():
        if score_mode == "next_token":
            yes_probabilities: List[Optional[float]] = next_token_yes_probabilities(
                model=model,
                tokenizer=tokenizer,
                model_inputs_list=model_inputs_list,
                yes_id=yes_id,
                no_id=no_id,
            )
        else:
            yes_probabilities = [None] * len(items)

        generation = model.generate(
            **batched_inputs,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=score_mode == "answer_token" or collect_confidence_features,
            use_cache=True,
        )

    sequences = generation.sequences if hasattr(generation, "sequences") else generation
    prompt_width = batched_inputs["input_ids"].shape[1]
    raw_generated_ids_list = [sequences[index, prompt_width:] for index in range(len(items))]
    generated_ids_list = [
        trim_generated_ids_after_eos(raw_generated_ids, tokenizer.eos_token_id)
        for raw_generated_ids in raw_generated_ids_list
    ]
    output_texts = []
    parsed_answers = []
    format_flags = []
    for generated_ids in generated_ids_list:
        output_text = decode_text(processor, tokenizer, generated_ids.unsqueeze(0))
        output_text = output_text.replace("<|im_end|>", "").strip()
        parsed_answer = answer_from_output(output_text)
        output_texts.append(output_text)
        parsed_answers.append(parsed_answer)
        format_flags.append(parsed_answer is not None and ANSWER_RE.search(output_text) is not None)

    score_sources = ["next_token_yes_no_score"] * len(items)
    generation_scores = getattr(generation, "scores", ())
    if score_mode == "answer_token":
        with torch.inference_mode():
            prefix_scores = answer_prefix_yes_probabilities(
                model=model,
                tokenizer=tokenizer,
                model_inputs_list=model_inputs_list,
                output_texts=output_texts,
                yes_id=yes_id,
                no_id=no_id,
            )

        for index, prefix_score in enumerate(prefix_scores):
            if prefix_score is not None:
                yes_probabilities[index] = prefix_score
                score_sources[index] = "answer_prefix_yes_no_score"
                continue

            sample_scores = tuple(step_scores[index] for step_scores in generation_scores)
            yes_probability, score_source = yes_probability_from_generation(
                tokenizer=tokenizer,
                generated_ids=generated_ids_list[index],
                scores=sample_scores,
                output_text=output_texts[index],
                parsed_answer=parsed_answers[index],
                yes_no_pairs=yes_no_pairs,
            )
            yes_probabilities[index] = yes_probability
            score_sources[index] = score_source

    confidence_features_list: List[Optional[Dict[str, Any]]] = [None] * len(items)
    if collect_confidence_features:
        if not generation_scores:
            raise RuntimeError("confidence feature extraction requires generation scores, but model.generate returned none")
        for index in range(len(items)):
            sample_scores = tuple(step_scores[index] for step_scores in generation_scores)
            confidence_features_list[index] = extract_confidence_features(
                tokenizer=tokenizer,
                generated_ids=generated_ids_list[index],
                raw_generated_ids=raw_generated_ids_list[index],
                scores=sample_scores,
                output_text=output_texts[index],
                max_new_tokens=max_new_tokens,
                low_margin_threshold=low_margin_threshold,
                repeated_ngram_size=repeated_ngram_size,
                logit_chunk_size=feature_logit_chunk_size,
            )

    results = []
    for output_text, parsed_answer, format_ok, yes_probability, score_source, confidence_features in zip(
        output_texts,
        parsed_answers,
        format_flags,
        yes_probabilities,
        score_sources,
        confidence_features_list,
    ):
        results.append(
            {
                "output_text": output_text,
                "parsed_answer": parsed_answer,
                "format_ok": bool(format_ok),
                "yes_probability": float(yes_probability),
                "score_source": score_source,
                "confidence_features": confidence_features,
            }
        )
    return results


def run_inference(args: argparse.Namespace) -> None:
    model_path = resolve_model_path(args.model_path, args.auto_merge_checkpoint)
    input_file = args.input_file
    output_file = args.output_file

    if not Path(input_file).exists():
        raise FileNotFoundError(f"input file does not exist: {input_file}")

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

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    model_cls = choose_model_class(config)
    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval().to(device)

    yes_id = single_token_id(tokenizer, "Yes")
    no_id = single_token_id(tokenizer, "No")
    print(f"[INFO] Yes token id={yes_id}, No token id={no_id}", flush=True)
    print(f"[INFO] inference batch_size={args.batch_size}", flush=True)
    print(
        f"[INFO] confidence features={'enabled' if args.collect_confidence_features else 'disabled'}, "
        f"low_margin_threshold={args.low_margin_threshold}, repeated_ngram_size={args.repeated_ngram_size}",
        flush=True,
    )
    yes_no_pairs = build_yes_no_pairs(tokenizer) if args.score_mode == "answer_token" else []
    if args.score_mode == "answer_token" and not yes_no_pairs:
        print("[WARN] failed to find single-token Yes/No ids; yes_logit will fall back to parsed answers")

    total_rows = count_jsonl_rows(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    written = 0
    skipped = 0
    missing_label = 0
    missing_baseline = 0
    score_sources: Dict[str, int] = {}

    with open(output_path, "w", encoding="utf-8", buffering=1) as fout:
        shard_rows = len(range(args.shard_rank, total_rows, args.num_shards))
        if args.limit is not None:
            shard_rows = min(shard_rows, args.limit)
        progress = tqdm(
            total=shard_rows,
            desc=f"infer shard {args.shard_rank + 1}/{args.num_shards}",
            position=args.shard_rank if args.num_shards > 1 else 0,
        )

        def write_result(item: Dict[str, Any], inference: Dict[str, Any]) -> None:
            nonlocal written, missing_label, missing_baseline

            row = item["row"]
            line_no = item["line_no"]
            score_source = inference["score_source"]
            score_sources[score_source] = score_sources.get(score_source, 0) + 1

            result = dict(row)
            result["id"] = result.get("id", str(line_no))
            result["response"] = inference["output_text"]
            result["pred_answer"] = inference["parsed_answer"]
            result["format_ok"] = inference["format_ok"]
            result["yes_logit"] = format(inference["yes_probability"], ".8f")
            result["yes_logit_source"] = score_source
            result["image_count"] = len(item["image_paths"])
            if inference["confidence_features"] is not None:
                result["confidence_features"] = inference["confidence_features"]

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

            valley_score = fusion_model_res.get("valley_score")
            if valley_score in (None, ""):
                missing_baseline += 1
                if args.baseline_score_fallback == "yes_logit":
                    fusion_model_res["valley_score"] = result["yes_logit"]
                    fusion_model_res["valley_score_source"] = "fallback_from_yes_logit"
                elif args.baseline_score_fallback == "constant_0_5":
                    fusion_model_res["valley_score"] = "0.50000000"
                    fusion_model_res["valley_score_source"] = "fallback_constant_0_5"

            result["extra_info"] = extra_info
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1

        def process_batch(items: List[Dict[str, Any]]) -> None:
            nonlocal skipped
            if not items:
                return

            batch_error = None
            try:
                inferences = infer_prepared_batch(
                    items=items,
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    score_mode=args.score_mode,
                    yes_id=yes_id,
                    no_id=no_id,
                    yes_no_pairs=yes_no_pairs,
                    repetition_penalty=args.repetition_penalty,
                    max_new_tokens=args.max_new_tokens,
                    collect_confidence_features=args.collect_confidence_features,
                    low_margin_threshold=args.low_margin_threshold,
                    repeated_ngram_size=args.repeated_ngram_size,
                    feature_logit_chunk_size=args.feature_logit_chunk_size,
                )
            except Exception as exc:
                batch_error = exc

            if batch_error is not None:
                if len(items) > 1:
                    print(
                        f"[WARN] batch_size={len(items)} failed with {type(batch_error).__name__}: {batch_error}; "
                        "retrying this batch one sample at a time",
                        flush=True,
                    )
                    batch_error.__traceback__ = None
                    batch_error = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for item in items:
                        process_batch([item])
                    return

                skipped += 1
                line_no = items[0]["line_no"]
                if not args.skip_errors:
                    raise batch_error
                print(f"[WARN] skip line {line_no}", flush=True)
                print(f"{type(batch_error).__name__}: {batch_error}", flush=True)
                progress.update(1)
                progress.set_postfix({"written": written, "skipped": skipped, "batch": args.batch_size})
                return

            for item, inference in zip(items, inferences):
                write_result(item, inference)
            progress.update(len(items))
            progress.set_postfix({"written": written, "skipped": skipped, "batch": len(items)})

        pending_items = []
        for raw_index, (line_no, row) in enumerate(iter_jsonl(input_file)):
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
                progress.set_postfix({"written": written, "skipped": skipped, "batch": args.batch_size})

        process_batch(pending_items)
        progress.close()

    summary = {
        "input_file": input_file,
        "output_file": output_file,
        "processed": processed,
        "written": written,
        "skipped": skipped,
        "missing_label": missing_label,
        "missing_or_fallback_baseline_valley_score": missing_baseline,
        "score_sources": score_sources,
        "baseline_score_fallback": args.baseline_score_fallback,
        "batch_size": args.batch_size,
        "collect_confidence_features": args.collect_confidence_features,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--prompt_key", default="problem")
    parser.add_argument("--target_key", default="solution")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_pixels", type=int, default=100352)
    parser.add_argument("--min_pixels", type=int, default=50176)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--trust_remote_code", action="store_true", default=env_bool("TRUST_REMOTE_CODE", True))
    parser.add_argument("--device", default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples generated together on each GPU worker. Failed batches automatically retry one sample at a time.",
    )
    parser.add_argument(
        "--collect_confidence_features",
        dest="collect_confidence_features",
        action="store_true",
        default=env_bool("COLLECT_CONFIDENCE_FEATURES", True),
        help="Extract step-level uncertainty and trajectory confidence features from generation logits.",
    )
    parser.add_argument(
        "--no_collect_confidence_features",
        dest="collect_confidence_features",
        action="store_false",
        help="Disable trajectory confidence feature extraction.",
    )
    parser.add_argument(
        "--low_margin_threshold",
        type=float,
        default=1.0,
        help="A generated token is counted as low-margin when top1_logit - top2_logit is below this value.",
    )
    parser.add_argument(
        "--repeated_ngram_size",
        type=int,
        default=3,
        help="N-gram size used by repeated_ngram_ratio.",
    )
    parser.add_argument(
        "--feature_logit_chunk_size",
        type=int,
        default=32,
        help="Number of generated-token logits processed together when extracting confidence features.",
    )
    parser.add_argument(
        "--score_mode",
        choices=("next_token", "answer_token"),
        default="answer_token",
        help="answer_token scores Yes/No at the generated <answer> position after CoT; next_token is a prompt-level diagnostic probe.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--skip_errors", action="store_true")
    parser.add_argument(
        "--auto_merge_checkpoint",
        action="store_true",
        help="If MODEL_PATH points to a verl/FSDP checkpoint, merge actor shards into actor/huggingface before inference.",
    )
    parser.add_argument(
        "--baseline_score_fallback",
        choices=("none", "yes_logit", "constant_0_5"),
        default="none",
        help="How to fill extra_info.fusion_model_res.valley_score when the input data does not provide it.",
    )
    args = parser.parse_args()
    if args.num_shards <= 0:
        parser.error("--num_shards must be greater than 0")
    if args.shard_rank < 0 or args.shard_rank >= args.num_shards:
        parser.error("--shard_rank must be in [0, num_shards)")
    if args.batch_size <= 0:
        parser.error("--batch_size must be greater than 0")
    if args.repeated_ngram_size <= 0:
        parser.error("--repeated_ngram_size must be greater than 0")
    if args.feature_logit_chunk_size <= 0:
        parser.error("--feature_logit_chunk_size must be greater than 0")
    return args


if __name__ == "__main__":
    run_inference(parse_args())
