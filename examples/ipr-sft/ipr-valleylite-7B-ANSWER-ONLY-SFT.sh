#!/usr/bin/env bash
set -euo pipefail

# Standalone answer-binary SFT script.
# The input JSONL can still contain CoT in `solution`; this script only
# trains a binary Yes/No objective from the final answer label.
# It intentionally does not touch the existing SFT+GRPO/ROPD training entrypoints.

set -x

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/mnt/bn/yangmin-priv/czh/checkpoints/valley/valleylite_ecom_vl_7B_Gthinker_omni_data_v121_150k_opensourse_150k/}"
#MODEL_PATH="${MODEL_PATH:-/mnt/bn/chenhaobo-va-data/lrj/checkpoints/valleylite-7b-ipr-cot-sft-0524/final/}"
#TRAIN_FILE="${TRAIN_FILE:-/mnt/bn/fengshuyang-bytenas-10t/jiangzishang/mllm/data_flywheel/result/grpo_data/train_clean.jsonl}"
TRAIN_FILE="${TRAIN_FILE:-/mnt/bn/chenhaobo-va-data/lrj/data/train_ropd/train_sft/gpt5.4_4600.jsonl}"
#VAL_FILE="${VAL_FILE:-/mnt/bn/fengshuyang-bytenas-10t/jiangzishang/mllm/data_flywheel/result/grpo_data/test_clean.jsonl}"
VAL_FILE="${VAL_FILE:-/mnt/bn/chenhaobo-va-data/lrj/data/train_ropd/cot_6k_train.jsonl}"

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/bn/chenhaobo-va-data/lrj/checkpoints/valleylite-7b-answer-binary-sft}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${OUTPUT_DIR}/deepspeed_zero1_offload.json}"

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

PROMPT_KEY="${PROMPT_KEY:-problem}"
TARGET_KEY="${TARGET_KEY:-solution}"
ANSWER_KEY="${ANSWER_KEY:-answer}"
YES_TOKEN_TEXT="${YES_TOKEN_TEXT:-Yes}"
NO_TOKEN_TEXT="${NO_TOKEN_TEXT:-No}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_PIXELS="${MAX_PIXELS:-100352}"
MIN_PIXELS="${MIN_PIXELS:-50176}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
EVAL_STEPS="${EVAL_STEPS:-200}"
EVAL_ON_START="${EVAL_ON_START:-true}"
SAVE_STEPS="${SAVE_STEPS:-20}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-ipr-answer-binary-sft}"
RUN_NAME="${RUN_NAME:-valleylite-7b-ipr-answer-binary-sft}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-true}"
FREEZE_MM_PROJECTOR="${FREEZE_MM_PROJECTOR:-false}"
SFT_GRADIENT_CHECKPOINTING="${SFT_GRADIENT_CHECKPOINTING:-true}"
SFT_FSDP="${SFT_FSDP:-}"
SFT_FSDP_ACTIVATION_CHECKPOINTING="${SFT_FSDP_ACTIVATION_CHECKPOINTING:-true}"
SFT_FSDP_USE_ORIG_PARAMS="${SFT_FSDP_USE_ORIG_PARAMS:-true}"
SFT_FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP="${SFT_FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP:-}"
SFT_SKIP_BAD_SAMPLES="${SFT_SKIP_BAD_SAMPLES:-true}"
SFT_EVAL_SKIP_BAD_SAMPLES="${SFT_EVAL_SKIP_BAD_SAMPLES:-true}"
SFT_MAX_SAMPLE_RETRIES="${SFT_MAX_SAMPLE_RETRIES:-16}"
SFT_AUDIT_DATASETS="${SFT_AUDIT_DATASETS:-true}"
SFT_LOAD_TRUNCATED_IMAGES="${SFT_LOAD_TRUNCATED_IMAGES:-true}"

export MODEL_PATH TRAIN_FILE VAL_FILE OUTPUT_DIR
export PROMPT_KEY TARGET_KEY ANSWER_KEY YES_TOKEN_TEXT NO_TOKEN_TEXT MAX_PROMPT_LENGTH MAX_PIXELS MIN_PIXELS
export NUM_TRAIN_EPOCHS MAX_STEPS PER_DEVICE_TRAIN_BATCH_SIZE PER_DEVICE_EVAL_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS
export LEARNING_RATE WEIGHT_DECAY WARMUP_RATIO LOGGING_STEPS EVAL_STEPS EVAL_ON_START SAVE_STEPS SAVE_TOTAL_LIMIT MAX_GRAD_NORM MAX_SAMPLES MAX_EVAL_SAMPLES
export TRUST_REMOTE_CODE ATTN_IMPLEMENTATION REPORT_TO WANDB_PROJECT RUN_NAME
export FREEZE_VISION_TOWER FREEZE_MM_PROJECTOR SFT_GRADIENT_CHECKPOINTING SFT_FSDP
export SFT_FSDP_ACTIVATION_CHECKPOINTING SFT_FSDP_USE_ORIG_PARAMS SFT_FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP
export SFT_SKIP_BAD_SAMPLES SFT_EVAL_SKIP_BAD_SAMPLES SFT_MAX_SAMPLE_RETRIES SFT_AUDIT_DATASETS SFT_LOAD_TRUNCATED_IMAGES
export DEEPSPEED_CONFIG

mkdir -p "${OUTPUT_DIR}"
if [[ -n "${DEEPSPEED_CONFIG}" && ! -f "${DEEPSPEED_CONFIG}" ]]; then
  cat > "${DEEPSPEED_CONFIG}" <<'JSON'
{
  "fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "bf16": {
    "enabled": "auto"
  },
  "train_micro_batch_size_per_gpu": "auto",
  "train_batch_size": "auto",
  "gradient_accumulation_steps": "auto",
  "zero_optimization": {
    "stage": 1,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1000000000.0,
    "reduce_bucket_size": "auto",
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
JSON
fi

cd "${PROJECT_ROOT}"

if [[ -z "${MASTER_ADDR:-}" ]]; then
  export MASTER_ADDR="127.0.0.1"
fi
if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="$(python3 - <<'PY'
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("", 0))
print(sock.getsockname()[1])
sock.close()
PY
)"
  export MASTER_PORT
fi
echo "[answer-binary sft] torchrun rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"

SFT_TRAIN_PY="${TMPDIR:-/tmp}/ipr_answer_only_sft_train_${USER:-user}_$$.py"
trap 'rm -f "${SFT_TRAIN_PY}"' EXIT

cat > "${SFT_TRAIN_PY}" <<'PY'
import json
import math
import os
import inspect
import random
import hashlib
import re
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image, ImageFile
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from verl.utils import get_processor, get_tokenizer


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def normalize_answer_label(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^<answer>\s*|\s*</answer>$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    normalized = text.upper()
    if normalized in {"YES", "Y", "TRUE", "1"}:
        return "Yes"
    if normalized in {"NO", "N", "FALSE", "0"}:
        return "No"
    raise ValueError(f"Cannot normalize answer label from {value!r}")


def extract_answer_label(row: Dict[str, Any], target_key: str) -> str:
    answer_key = os.getenv("ANSWER_KEY", "answer")
    if answer_key in row and str(row.get(answer_key, "")).strip():
        return normalize_answer_label(row[answer_key])

    target_text = str(row.get(target_key, ""))
    matches = re.findall(r"<answer>(.*?)</answer>", target_text, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        return normalize_answer_label(matches[-1])

    stripped = target_text.strip()
    if stripped.upper() in {"YES", "NO"}:
        return normalize_answer_label(stripped)

    raise ValueError(
        f"Cannot find answer label from answer_key={answer_key!r} or target_key={target_key!r}"
    )


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


def load_and_process_image(path: str, max_pixels: int, min_pixels: int) -> Image.Image:
    try:
        with Image.open(path) as image:
            return process_image(image.copy(), max_pixels, min_pixels)
    except Exception as exc:
        raise OSError(f"failed to load image {path}: {type(exc).__name__}: {exc}") from exc


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def squeeze_1d(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.long)
    if tensor.dim() == 2 and tensor.size(0) == 1:
        tensor = tensor[0]
    return tensor.long()


def to_list_ids(value: Any) -> List[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


def parameter_name_matches(name: str, keywords: List[str]) -> bool:
    return any(name == keyword or name.startswith(keyword + ".") or f".{keyword}." in name for keyword in keywords)


def maybe_freeze_modules(model, *, freeze_vision: bool, freeze_projector: bool) -> None:
    vision_keywords = ["visual", "vision_tower", "vision_model", "vision_encoder"]
    projector_keywords = ["mm_projector", "multi_modal_projector", "vision_projector", "visual_projector", "merger"]
    frozen_vision = 0
    frozen_projector = 0
    for name, param in model.named_parameters():
        if freeze_vision and parameter_name_matches(name, vision_keywords):
            param.requires_grad_(False)
            frozen_vision += param.numel()
        elif freeze_projector and parameter_name_matches(name, projector_keywords):
            param.requires_grad_(False)
            frozen_projector += param.numel()

    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(
        json.dumps(
            {
                "sft_trainable_params": trainable,
                "sft_total_params": total,
                "sft_trainable_ratio": trainable / total if total else 0.0,
                "sft_frozen_vision_params": frozen_vision,
                "sft_frozen_projector_params": frozen_projector,
            },
            ensure_ascii=False,
        )
    )


def collate_tensor_or_list(values: List[Any], *, cat: bool = False):
    if not values or any(value is None for value in values):
        return None
    if all(isinstance(value, torch.Tensor) for value in values):
        if cat:
            return torch.cat(values, dim=0)
        try:
            return torch.stack(values, dim=0)
        except RuntimeError:
            return values
    return values


class IPRCotSFTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        data_path: str,
        model_path: str,
        tokenizer,
        processor,
        prompt_key: str,
        target_key: str,
        max_prompt_length: int,
        max_pixels: int,
        min_pixels: int,
        trust_remote_code: bool,
        max_samples: int,
        split_name: str,
        skip_bad_samples: bool,
    ) -> None:
        self.dataset = load_dataset("json", data_files=data_path, split="train")
        if max_samples and max_samples > 0:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.target_key = target_key
        self.max_prompt_length = max_prompt_length
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.is_valley = getattr(self.config, "model_type", None) == "valley"
        self.split_name = split_name
        self.skip_bad_samples = skip_bad_samples
        self.max_sample_retries = max(1, env_int("SFT_MAX_SAMPLE_RETRIES", 16))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._getitem_with_retry(index, retry_depth=0)

    def _getitem_with_retry(self, index: int, retry_depth: int) -> Dict[str, Any]:
        row = dict(self.dataset[index])
        try:
            return self._build_item_from_row(row, index=index)
        except Exception as exc:
            if not self.skip_bad_samples or retry_depth >= self.max_sample_retries:
                raise
            traceback.print_exc()
            print(f" >>> failed to load {self.split_name} sample index={index}: {row}", flush=True)
            print(f" >>> reason {type(exc).__name__}: {exc}", flush=True)
            return self._getitem_with_retry(
                random.randint(0, len(self.dataset) - 1),
                retry_depth=retry_depth + 1,
            )

    def _build_item_from_row(self, row: Dict[str, Any], *, index: int) -> Dict[str, Any]:
        if "image" in row and "images" not in row:
            row["images"] = row["image"]
        images = as_list(row.get("images"))
        image_paths = [item for item in images if isinstance(item, str) and item.strip()]
        missing_image_paths = [path for path in image_paths if not Path(path).exists()]
        if missing_image_paths:
            preview = ", ".join(missing_image_paths[:3])
            suffix = "" if len(missing_image_paths) <= 3 else f", ... (+{len(missing_image_paths) - 3} more)"
            raise FileNotFoundError(f"missing image file(s) for sample index={index}: {preview}{suffix}")
        pil_images = [load_and_process_image(path, self.max_pixels, self.min_pixels) for path in image_paths]

        prompt_text = str(row[self.prompt_key])
        answer_label = extract_answer_label(row, self.target_key)
        class_label = 1 if answer_label == "Yes" else 0

        if self.is_valley:
            processed = self.processor(
                {
                    "conversations": [{"role": "user", "content": prompt_text}],
                    "images": image_paths if image_paths else None,
                }
            )
            prompt_ids = squeeze_1d(processed["input_ids"])
            if prompt_ids.numel() > self.max_prompt_length:
                raise ValueError(
                    f"prompt too long after multimodal processing: "
                    f"{prompt_ids.numel()} > {self.max_prompt_length}; image_count={len(image_paths)}"
                )
            item = {
                "input_ids": prompt_ids,
                "images": processed.get("images"),
                "image_sizes": processed.get("image_sizes"),
                "pixel_values": processed.get("pixel_values"),
                "image_grid_thw": processed.get("image_grid_thw"),
            }
        else:
            messages = [{"role": "user", "content": prompt_text}]
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            item = {}
            if pil_images:
                qwen_raw_prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
                image_inputs = self.processor.image_processor(pil_images, return_tensors="pt")
                image_grid_thw = image_inputs["image_grid_thw"]
                merge_length = self.processor.image_processor.merge_size**2
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
                if image_index != len(image_grid_thw):
                    raise ValueError(
                        f"image placeholder count mismatch: used {image_index} placeholders "
                        f"for {len(image_grid_thw)} image(s)"
                    )
                prompt = prompt.replace("<|placeholder|>", self.processor.image_token)
                item.update(image_inputs)
                _ = qwen_raw_prompt
            prompt_ids = torch.tensor(
                self.tokenizer.encode(prompt, add_special_tokens=False),
                dtype=torch.long,
            )
            if prompt_ids.numel() > self.max_prompt_length:
                raise ValueError(
                    f"prompt too long after image token expansion: "
                    f"{prompt_ids.numel()} > {self.max_prompt_length}; image_count={len(image_paths)}"
                )
            if pil_images:
                image_token_id = self.tokenizer.convert_tokens_to_ids(self.processor.image_token)
                if image_token_id is not None and image_token_id >= 0:
                    expected_image_tokens = sum(
                        int(grid.prod().item() // merge_length) for grid in image_grid_thw
                    )
                    actual_image_tokens = int((prompt_ids == image_token_id).sum().item())
                    if actual_image_tokens != expected_image_tokens:
                        raise ValueError(
                            f"image token count mismatch: actual={actual_image_tokens}, "
                            f"expected={expected_image_tokens}, image_count={len(image_paths)}"
                        )
            item["input_ids"] = prompt_ids

        item["class_labels"] = torch.tensor(class_label, dtype=torch.long)
        return item


def stable_digest(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def dataset_relationship_stats(dataset: IPRCotSFTDataset):
    prompt_targets = defaultdict(set)
    prompt_target_pairs = set()
    answer_counts = defaultdict(int)
    valid_rows = 0
    invalid_rows = 0
    for row in dataset.dataset:
        try:
            prompt_digest = stable_digest(row[dataset.prompt_key])
            answer_label = extract_answer_label(row, dataset.target_key)
            target_digest = stable_digest(answer_label)
        except Exception:
            invalid_rows += 1
            continue
        valid_rows += 1
        prompt_targets[prompt_digest].add(target_digest)
        prompt_target_pairs.add((prompt_digest, target_digest))
        answer_counts[answer_label] += 1

    stats = {
        "rows": len(dataset.dataset),
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "unique_prompts": len(prompt_targets),
        "yes_count": int(answer_counts["Yes"]),
        "no_count": int(answer_counts["No"]),
        "yes_ratio": float(answer_counts["Yes"] / valid_rows) if valid_rows else 0.0,
    }
    return stats, prompt_targets, prompt_target_pairs


def audit_train_eval_relationship(train_dataset: IPRCotSFTDataset, eval_dataset: IPRCotSFTDataset) -> None:
    if not env_bool("SFT_AUDIT_DATASETS", True):
        return

    train_stats, train_prompt_targets, train_pairs = dataset_relationship_stats(train_dataset)
    eval_stats, eval_prompt_targets, eval_pairs = dataset_relationship_stats(eval_dataset)
    overlapping_prompts = set(train_prompt_targets).intersection(eval_prompt_targets)
    conflicting_eval_pairs = sum(
        1
        for prompt_digest, target_digest in eval_pairs
        if prompt_digest in train_prompt_targets and target_digest not in train_prompt_targets[prompt_digest]
    )
    overlapping_eval_pairs = sum(1 for prompt_digest, _ in eval_pairs if prompt_digest in overlapping_prompts)

    audit = {
        "sft_dataset_audit": {
            "train": train_stats,
            "eval": eval_stats,
            "overlapping_unique_prompts": len(overlapping_prompts),
            "eval_prompt_overlap_rate": (
                len(overlapping_prompts) / len(eval_prompt_targets) if eval_prompt_targets else 0.0
            ),
            "exact_prompt_target_pair_overlap": len(train_pairs.intersection(eval_pairs)),
            "conflicting_eval_target_pairs_on_overlapping_prompts": conflicting_eval_pairs,
            "conflicting_target_rate_on_overlapping_eval_pairs": (
                conflicting_eval_pairs / overlapping_eval_pairs if overlapping_eval_pairs else 0.0
            ),
        }
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if conflicting_eval_pairs:
        print(
            "[sft warning] Train/eval contain overlapping prompts with different targets. "
            "Eval loss/accuracy may be noisy if the same prompt has different Yes/No labels.",
            flush=True,
        )


class IPRBinaryAnswerCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids_list = []
        attention_mask_list = []
        class_labels = []

        for feature in features:
            prompt_ids = squeeze_1d(feature["input_ids"])
            attention_mask = torch.ones_like(prompt_ids)
            input_ids_list.append(prompt_ids)
            attention_mask_list.append(attention_mask)
            class_labels.append(int(feature["class_labels"]))

        max_length = max(item.size(0) for item in input_ids_list)
        batch_input_ids = []
        batch_attention_mask = []
        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            pad_length = max_length - input_ids.size(0)
            batch_input_ids.append(
                torch.cat([input_ids, torch.full((pad_length,), self.pad_token_id, dtype=torch.long)], dim=0)
            )
            batch_attention_mask.append(
                torch.cat([attention_mask, torch.zeros((pad_length,), dtype=torch.long)], dim=0)
            )

        batch: Dict[str, Any] = {
            "input_ids": torch.stack(batch_input_ids, dim=0),
            "attention_mask": torch.stack(batch_attention_mask, dim=0),
            "class_labels": torch.tensor(class_labels, dtype=torch.long),
        }

        optional_keys = ("pixel_values", "image_grid_thw", "images", "image_sizes")
        grouped = defaultdict(list)
        for feature in features:
            for key in optional_keys:
                if key in feature and feature[key] is not None:
                    value = feature[key]
                    if isinstance(value, list) and len(value) == 1:
                        value = value[0]
                    grouped[key].append(value)

        if len(grouped.get("pixel_values", [])) == len(features):
            batch["pixel_values"] = collate_tensor_or_list(grouped["pixel_values"], cat=True)
        if len(grouped.get("image_grid_thw", [])) == len(features):
            batch["image_grid_thw"] = collate_tensor_or_list(grouped["image_grid_thw"], cat=True)
        if len(grouped.get("images", [])) == len(features):
            batch["images"] = grouped["images"]
        if len(grouped.get("image_sizes", [])) == len(features):
            batch["image_sizes"] = grouped["image_sizes"]

        return batch


def get_single_token_id(tokenizer, text: str, name: str) -> int:
    token_ids = to_list_ids(tokenizer.encode(text, add_special_tokens=False))
    if len(token_ids) != 1:
        raise ValueError(
            f"{name}={text!r} must tokenize to exactly one token for binary loss, "
            f"but got token_ids={token_ids}. Set {name}_TOKEN_TEXT to a single-token form."
        )
    return int(token_ids[0])


class IPRBinaryAnswerTrainer(Trainer):
    def __init__(self, *args, yes_token_id: int, no_token_id: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.yes_token_id = int(yes_token_id)
        self.no_token_id = int(no_token_id)
        self.label_names = ["class_labels"]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        class_labels = inputs.pop("class_labels").long()
        attention_mask = inputs.get("attention_mask")
        outputs = model(**inputs)
        logits = outputs.logits
        last_prompt_indices = attention_mask.long().sum(dim=1) - 1
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        next_token_logits = logits[batch_indices, last_prompt_indices, :]
        binary_logits = torch.stack(
            [
                next_token_logits[:, self.no_token_id],
                next_token_logits[:, self.yes_token_id],
            ],
            dim=-1,
        )
        loss = F.cross_entropy(binary_logits.float(), class_labels.to(binary_logits.device))
        if return_outputs:
            return loss, {"logits": binary_logits}
        return loss


def compute_binary_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(labels, tuple):
        labels = labels[0]
    pred_labels = np.asarray(predictions).argmax(axis=-1)
    labels = np.asarray(labels)
    return {
        "accuracy": float((pred_labels == labels).mean()) if labels.size else 0.0,
        "pred_yes_ratio": float((pred_labels == 1).mean()) if pred_labels.size else 0.0,
        "label_yes_ratio": float((labels == 1).mean()) if labels.size else 0.0,
    }


def main() -> None:
    ImageFile.LOAD_TRUNCATED_IMAGES = env_bool("SFT_LOAD_TRUNCATED_IMAGES", False)
    model_path = os.environ["MODEL_PATH"]
    output_dir = os.environ["OUTPUT_DIR"]
    trust_remote_code = env_bool("TRUST_REMOTE_CODE", False)
    attn_implementation = os.getenv("ATTN_IMPLEMENTATION", "sdpa")

    tokenizer = get_tokenizer(model_path, trust_remote_code=trust_remote_code)
    yes_token_text = os.getenv("YES_TOKEN_TEXT", "Yes")
    no_token_text = os.getenv("NO_TOKEN_TEXT", "No")
    yes_token_id = get_single_token_id(tokenizer, yes_token_text, "YES")
    no_token_id = get_single_token_id(tokenizer, no_token_text, "NO")
    print(
        json.dumps(
            {
                "binary_answer_tokens": {
                    "class_0": {"label": "No", "text": no_token_text, "token_id": no_token_id},
                    "class_1": {"label": "Yes", "text": yes_token_text, "token_id": yes_token_id},
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    processor = get_processor(
        model_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
        max_pixels=env_int("MAX_PIXELS", 100352),
        min_pixels=env_int("MIN_PIXELS", 50176),
    )
    if processor is None:
        raise RuntimeError("A multimodal processor is required for IPR answer-binary SFT.")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if type(config) in AutoModelForVision2Seq._model_mapping.keys():
        model_cls = AutoModelForVision2Seq
    else:
        model_cls = AutoModelForCausalLM

    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )
    model.config.use_cache = False
    maybe_freeze_modules(
        model,
        freeze_vision=env_bool("FREEZE_VISION_TOWER", True),
        freeze_projector=env_bool("FREEZE_MM_PROJECTOR", False),
    )
    if env_bool("SFT_GRADIENT_CHECKPOINTING", True) and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    train_dataset = IPRCotSFTDataset(
        data_path=os.environ["TRAIN_FILE"],
        model_path=model_path,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=os.getenv("PROMPT_KEY", "problem"),
        target_key=os.getenv("TARGET_KEY", "solution"),
        max_prompt_length=env_int("MAX_PROMPT_LENGTH", 4096),
        max_pixels=env_int("MAX_PIXELS", 100352),
        min_pixels=env_int("MIN_PIXELS", 50176),
        trust_remote_code=trust_remote_code,
        max_samples=env_int("MAX_SAMPLES", 0),
        split_name="train",
        skip_bad_samples=env_bool("SFT_SKIP_BAD_SAMPLES", True),
    )

    eval_dataset = None
    val_file = os.getenv("VAL_FILE", "").strip()
    if val_file:
        if Path(val_file).exists():
            eval_dataset = IPRCotSFTDataset(
                data_path=val_file,
                model_path=model_path,
                tokenizer=tokenizer,
                processor=processor,
                prompt_key=os.getenv("PROMPT_KEY", "problem"),
                target_key=os.getenv("TARGET_KEY", "solution"),
                max_prompt_length=env_int("MAX_PROMPT_LENGTH", 4096),
                max_pixels=env_int("MAX_PIXELS", 100352),
                min_pixels=env_int("MIN_PIXELS", 50176),
                trust_remote_code=trust_remote_code,
                max_samples=env_int("MAX_EVAL_SAMPLES", 0),
                split_name="eval",
                skip_bad_samples=env_bool("SFT_EVAL_SKIP_BAD_SAMPLES", False),
            )
        else:
            print(f"[sft warning] VAL_FILE does not exist, disable eval: {val_file}")

    report_to = os.getenv("REPORT_TO", "wandb").strip()
    training_kwargs = {
        "output_dir": output_dir,
        "run_name": os.getenv("RUN_NAME", "valleylite-7b-ipr-answer-binary-sft"),
        "report_to": [] if report_to == "none" else [report_to],
        "num_train_epochs": env_float("NUM_TRAIN_EPOCHS", 1.0),
        "max_steps": env_int("MAX_STEPS", -1),
        "per_device_train_batch_size": env_int("PER_DEVICE_TRAIN_BATCH_SIZE", 1),
        "per_device_eval_batch_size": env_int("PER_DEVICE_EVAL_BATCH_SIZE", 1),
        "gradient_accumulation_steps": env_int("GRADIENT_ACCUMULATION_STEPS", 8),
        "learning_rate": env_float("LEARNING_RATE", 5e-6),
        "weight_decay": env_float("WEIGHT_DECAY", 0.01),
        "warmup_ratio": env_float("WARMUP_RATIO", 0.03),
        "max_grad_norm": env_float("MAX_GRAD_NORM", 1.0),
        "bf16": True,
        "logging_steps": env_int("LOGGING_STEPS", 1),
        "save_strategy": "steps",
        "save_steps": env_int("SAVE_STEPS", 200),
        "save_total_limit": env_int("SAVE_TOTAL_LIMIT", 3),
        "remove_unused_columns": False,
        "dataloader_num_workers": env_int("DATALOADER_NUM_WORKERS", 2),
        "ddp_find_unused_parameters": False,
        "gradient_checkpointing": env_bool("SFT_GRADIENT_CHECKPOINTING", True),
    }
    if "label_names" in inspect.signature(TrainingArguments).parameters:
        training_kwargs["label_names"] = ["class_labels"]
    deepspeed_config = os.getenv("DEEPSPEED_CONFIG", "").strip()
    fsdp = os.getenv("SFT_FSDP", "").strip()
    if deepspeed_config:
        training_kwargs["deepspeed"] = deepspeed_config
    elif fsdp:
        fsdp_config = {
            "activation_checkpointing": env_bool("SFT_FSDP_ACTIVATION_CHECKPOINTING", True),
            "use_orig_params": env_bool("SFT_FSDP_USE_ORIG_PARAMS", True),
        }
        transformer_layer_cls = os.getenv("SFT_FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP", "").strip()
        if transformer_layer_cls:
            fsdp_config["transformer_layer_cls_to_wrap"] = [
                item.strip() for item in transformer_layer_cls.split(",") if item.strip()
            ]
        training_kwargs["fsdp"] = fsdp
        training_kwargs["fsdp_config"] = fsdp_config
    if "gradient_checkpointing_kwargs" in inspect.signature(TrainingArguments).parameters:
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if eval_dataset is not None:
        audit_train_eval_relationship(train_dataset, eval_dataset)
        training_kwargs.update(
            {
                "eval_steps": env_int("EVAL_STEPS", 200),
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )
        if "eval_on_start" in inspect.signature(TrainingArguments).parameters:
            training_kwargs["eval_on_start"] = env_bool("EVAL_ON_START", True)
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters:
            training_kwargs["eval_strategy"] = "steps"
        else:
            training_kwargs["evaluation_strategy"] = "steps"
    else:
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters:
            training_kwargs["eval_strategy"] = "no"
        else:
            training_kwargs["evaluation_strategy"] = "no"
    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": IPRBinaryAnswerCollator(tokenizer),
        "yes_token_id": yes_token_id,
        "no_token_id": no_token_id,
        "compute_metrics": compute_binary_metrics,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    if "processing_class" in inspect.signature(Trainer).parameters:
        trainer_kwargs["processing_class"] = processor if processor is not None else tokenizer
    else:
        trainer_kwargs["tokenizer"] = processor if processor is not None else tokenizer
    trainer = IPRBinaryAnswerTrainer(**trainer_kwargs)
    trainer.train()

    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    if processor is not None:
        processor.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if trainer.is_world_process_zero():
        print(json.dumps({"sft_final_model_path": str(final_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
PY

torchrun \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${SFT_TRAIN_PY}" "$@"
