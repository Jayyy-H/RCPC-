#!/usr/bin/env bash
set -euo pipefail

set -x

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_PATH="${MODEL_PATH:-/mnt/bn/chenhaobo-va-data/lrj/checkpoints/valleylite-7b-ipr-cot-sft-0524/final}"
TRAIN_FILE="${TRAIN_FILE:-/mnt/bn/fengshuyang-bytenas-10t/jiangzishang/mllm/data_flywheel/result/grpo_data/train_clean.jsonl}"
PROMPT_KEY="${PROMPT_KEY:-problem}"
TARGET_KEY="${TARGET_KEY:-solution}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_PIXELS="${MAX_PIXELS:-100352}"
MIN_PIXELS="${MIN_PIXELS:-50176}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-}"

export MODEL_PATH TRAIN_FILE PROMPT_KEY TARGET_KEY MAX_NEW_TOKENS MAX_PIXELS MIN_PIXELS
export ATTN_IMPLEMENTATION TRUST_REMOTE_CODE PRECISION SEED

python3 - <<'PY'
import json
import math
import os
import random
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForVision2Seq

from verl.utils import get_processor, get_tokenizer


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


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


def as_list(value: Any):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def move_to_device(value: Any, device: torch.device, dtype: torch.dtype | None = None):
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


def load_random_row(train_file: str):
    with open(train_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise RuntimeError(f"empty train file: {train_file}")
    seed = os.getenv("SEED", "").strip()
    rng = random.Random(int(seed)) if seed else random.Random()
    index = rng.randrange(len(rows))
    return index, rows[index]


def choose_model_class(config, model_path: str, trust_remote_code: bool):
    if getattr(config, "model_type", None) == "valley":
        return AutoModel
    if type(config) in AutoModelForVision2Seq._model_mapping.keys():
        return AutoModelForVision2Seq
    return AutoModelForCausalLM


def build_inputs(row, tokenizer, processor, config, device, dtype, prompt_key, max_pixels, min_pixels):
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
            "images": move_to_device(processed.get("images"), device, dtype),
            "image_sizes": processed.get("image_sizes"),
            "pixel_values": move_to_device(processed.get("pixel_values"), device, dtype),
            "image_grid_thw": move_to_device(processed.get("image_grid_thw"), device),
        }
        model_inputs = {key: value for key, value in model_inputs.items() if value is not None}
        return model_inputs, input_ids.shape[-1], image_paths

    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    model_inputs = {}
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

    input_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
    model_inputs["input_ids"] = input_ids
    return model_inputs, input_ids.shape[-1], image_paths


def main():
    model_path = os.environ["MODEL_PATH"]
    train_file = os.environ["TRAIN_FILE"]
    prompt_key = os.getenv("PROMPT_KEY", "problem")
    target_key = os.getenv("TARGET_KEY", "solution")
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "1024"))
    max_pixels = int(os.getenv("MAX_PIXELS", "100352"))
    min_pixels = int(os.getenv("MIN_PIXELS", "50176"))
    trust_remote_code = env_bool("TRUST_REMOTE_CODE", True)
    attn_implementation = os.getenv("ATTN_IMPLEMENTATION", "sdpa")
    precision = os.getenv("PRECISION", "bf16")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

    if not Path(model_path).exists():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    if not Path(train_file).exists():
        raise FileNotFoundError(f"TRAIN_FILE does not exist: {train_file}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(model_path, trust_remote_code=trust_remote_code)
    processor = get_processor(
        model_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )
    if processor is None:
        raise RuntimeError(f"processor not found in {model_path}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model_cls = choose_model_class(config, model_path, trust_remote_code)
    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )
    model.eval().to(device)

    index, row = load_random_row(train_file)
    model_inputs, input_token_len, image_paths = build_inputs(
        row,
        tokenizer,
        processor,
        config,
        device,
        dtype,
        prompt_key,
        max_pixels,
        min_pixels,
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            do_sample=False,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    if hasattr(output_ids, "sequences"):
        output_ids = output_ids.sequences
    output_text = processor.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0]
    output_text = output_text.replace("<|im_end|>", "").strip()
    format_ok = bool(
        re.fullmatch(
            r"\s*<think>[\s\S]+?</think>\s*<answer>\s*(Yes|No)\s*</answer>\s*",
            output_text,
            flags=re.IGNORECASE,
        )
    )

    print("\n" + "=" * 32 + " RANDOM SAMPLE " + "=" * 32)
    print(f"sample_index: {index}")
    print(f"image_count: {len(image_paths)}")
    print(f"gold_answer: {row.get('answer', '')}")
    print("\n" + "=" * 32 + " GOLD SOLUTION " + "=" * 32)
    print(row.get(target_key, ""))
    print("\n" + "=" * 32 + " MODEL OUTPUT " + "=" * 32)
    print(output_text)
    print("\n" + "=" * 32 + " FORMAT CHECK " + "=" * 32)
    print(json.dumps({"format_ok": format_ok}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
PY
