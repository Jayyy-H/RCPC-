#!/usr/bin/env python3
"""Mode-controlled curriculum SFT for progressively internalizing IPR CoT."""

from __future__ import annotations

import inspect
import json
import math
import os
import random
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import SequentialSampler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    Trainer,
    TrainingArguments,
)

from verl.utils import get_processor, get_tokenizer


MODES = ("LONG_COT", "STRUCTURED_SHORT", "COMPRESSED_SHORT", "BRIDGE", "ANSWER_ONLY")
STAGE2_SCHEDULE = [
    {
        "progress": 0.00,
        "weights": {
            "LONG_COT": 0.50,
            "STRUCTURED_SHORT": 0.30,
            "COMPRESSED_SHORT": 0.025,
            "BRIDGE": 0.025,
            "ANSWER_ONLY": 0.15,
        },
    },
    {
        "progress": 0.33,
        "weights": {
            "LONG_COT": 0.20,
            "STRUCTURED_SHORT": 0.40,
            "COMPRESSED_SHORT": 0.10,
            "BRIDGE": 0.10,
            "ANSWER_ONLY": 0.20,
        },
    },
    {
        "progress": 0.66,
        "weights": {
            "LONG_COT": 0.08,
            "STRUCTURED_SHORT": 0.22,
            "COMPRESSED_SHORT": 0.125,
            "BRIDGE": 0.125,
            "ANSWER_ONLY": 0.45,
        },
    },
    {
        "progress": 1.00,
        "weights": {
            "LONG_COT": 0.03,
            "STRUCTURED_SHORT": 0.07,
            "COMPRESSED_SHORT": 0.075,
            "BRIDGE": 0.075,
            "ANSWER_ONLY": 0.75,
        },
    },
]
STAGE3_SCHEDULE = [
    {
        "progress": 0.0,
        "weights": {
            "LONG_COT": 0.01,
            "STRUCTURED_SHORT": 0.04,
            "COMPRESSED_SHORT": 0.03,
            "BRIDGE": 0.02,
            "ANSWER_ONLY": 0.90,
        },
    },
    {
        "progress": 1.0,
        "weights": {
            "LONG_COT": 0.01,
            "STRUCTURED_SHORT": 0.04,
            "COMPRESSED_SHORT": 0.03,
            "BRIDGE": 0.02,
            "ANSWER_ONLY": 0.90,
        },
    },
]


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


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


def process_image(image: Image.Image, max_pixels: int, min_pixels: int) -> Image.Image:
    if image.width * image.height > max_pixels:
        scale = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            resample=Image.Resampling.NEAREST,
        )
    if image.width * image.height < min_pixels:
        scale = math.sqrt(min_pixels / (image.width * image.height))
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            resample=Image.Resampling.NEAREST,
        )
    return image.convert("RGB") if image.mode != "RGB" else image


def parameter_name_matches(name: str, keywords: Sequence[str]) -> bool:
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
        ),
        flush=True,
    )


def normalize_weights(weights: Dict[str, float], available_modes: Sequence[str]) -> Dict[str, float]:
    filtered = {mode: max(0.0, float(weights.get(mode, 0.0))) for mode in available_modes}
    total = sum(filtered.values())
    if total <= 0:
        raise ValueError(f"Curriculum has zero probability for available modes: {available_modes}")
    return {mode: value / total for mode, value in filtered.items()}


def normalize_schedule(raw_schedule: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    schedule = []
    for anchor in raw_schedule:
        progress = float(anchor["progress"])
        weights = {mode: float((anchor.get("weights") or {}).get(mode, 0.0)) for mode in MODES}
        schedule.append({"progress": progress, "weights": weights})
    schedule.sort(key=lambda item: item["progress"])
    if not schedule or schedule[0]["progress"] > 0 or schedule[-1]["progress"] < 1:
        raise ValueError("Curriculum schedule must cover progress 0.0 through 1.0.")
    return schedule


def load_schedule() -> List[Dict[str, Any]]:
    custom = os.getenv("CURRICULUM_SCHEDULE_JSON", "").strip()
    if custom:
        if not custom.startswith(("[", "{")) and Path(custom).exists():
            custom = Path(custom).read_text(encoding="utf-8")
        return normalize_schedule(json.loads(custom))
    preset = os.getenv("CURRICULUM_PRESET", "stage2").strip().lower()
    if preset == "stage2":
        return normalize_schedule(STAGE2_SCHEDULE)
    if preset == "stage3":
        return normalize_schedule(STAGE3_SCHEDULE)
    raise ValueError(f"Unknown CURRICULUM_PRESET={preset!r}; expected stage2 or stage3.")


def interpolated_weights(schedule: List[Dict[str, Any]], progress: float) -> Dict[str, float]:
    progress = min(1.0, max(0.0, progress))
    left = schedule[0]
    right = schedule[-1]
    for index in range(1, len(schedule)):
        if progress <= schedule[index]["progress"]:
            left = schedule[index - 1]
            right = schedule[index]
            break
    span = right["progress"] - left["progress"]
    ratio = 0.0 if span <= 0 else (progress - left["progress"]) / span
    return {
        mode: left["weights"][mode] + ratio * (right["weights"][mode] - left["weights"][mode])
        for mode in MODES
    }


def infer_output_mode(row: Dict[str, Any]) -> str:
    explicit = str(row.get("output_mode") or "").strip().upper()
    if explicit in MODES:
        return explicit
    solution = str(row.get("solution") or "")
    if "<think>" not in solution.lower():
        return "ANSWER_ONLY"
    if re_search(r"Step\s*1\s*:", solution) and re_search(r"Step\s*3\s*:", solution):
        return "LONG_COT"
    if re_search(r"Evidence\s*[—:-]", solution) and re_search(r"Decision\s*[—:-]", solution):
        return "COMPRESSED_SHORT"
    return "BRIDGE"


def re_search(pattern: str, text: str) -> bool:
    import re

    return re.search(pattern, text, flags=re.IGNORECASE) is not None


class IPRItemBuilder:
    def __init__(
        self,
        *,
        model_path: str,
        tokenizer,
        processor,
        prompt_key: str,
        target_key: str,
        max_prompt_length: int,
        max_response_length: int,
        max_pixels: int,
        min_pixels: int,
        trust_remote_code: bool,
    ) -> None:
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.target_key = target_key
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.is_valley = getattr(config, "model_type", None) == "valley"

    def build(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(row)
        if "image" in row and "images" not in row:
            row["images"] = row["image"]
        images = as_list(row.get("images"))
        image_paths = [item for item in images if isinstance(item, str) and item.strip()]
        pil_images = [process_image(Image.open(path), self.max_pixels, self.min_pixels) for path in image_paths]
        prompt_text = str(row[self.prompt_key])
        target_text = str(row[self.target_key])

        if self.is_valley:
            processed = self.processor(
                {
                    "conversations": [{"role": "user", "content": prompt_text}],
                    "images": image_paths if image_paths else None,
                }
            )
            prompt_ids = squeeze_1d(processed["input_ids"])
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
                        f"image placeholder mismatch: used {image_index}, images={len(image_grid_thw)}"
                    )
                prompt = prompt.replace("<|placeholder|>", self.processor.image_token)
                item.update(image_inputs)
            prompt_ids = torch.tensor(self.tokenizer.encode(prompt, add_special_tokens=False), dtype=torch.long)
            item["input_ids"] = prompt_ids

        if prompt_ids.numel() > self.max_prompt_length:
            raise ValueError(f"prompt length {prompt_ids.numel()} exceeds {self.max_prompt_length}")
        target_ids = to_list_ids(self.tokenizer.encode(target_text, add_special_tokens=False))
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
            target_ids.append(int(eos_id))
        target_ids = target_ids[: self.max_response_length]
        if eos_id is not None and target_ids and target_ids[-1] != eos_id:
            target_ids[-1] = int(eos_id)
        item["target_ids"] = torch.tensor(target_ids, dtype=torch.long)
        return item


class IPRMapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        data_path: str,
        builder: IPRItemBuilder,
        max_samples: int,
        output_mode: str = "",
        skip_bad_samples: bool,
    ) -> None:
        dataset = load_dataset("json", data_files=data_path, split="train")
        self.rows = [dict(row) for row in dataset]
        if output_mode:
            self.rows = [row for row in self.rows if infer_output_mode(row) == output_mode]
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        self.builder = builder
        self.skip_bad_samples = skip_bad_samples
        self.max_retries = max(1, env_int("SFT_MAX_SAMPLE_RETRIES", 16))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        for retry in range(self.max_retries + 1):
            try:
                return self.builder.build(self.rows[index])
            except Exception as exc:
                if not self.skip_bad_samples or retry >= self.max_retries:
                    raise
                traceback.print_exc()
                print(f"[sft] skip bad map sample index={index}: {exc}", flush=True)
                index = random.randint(0, len(self.rows) - 1)
        raise RuntimeError("unreachable")


class CurriculumMapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        data_path: str,
        builder: IPRItemBuilder,
        schedule: List[Dict[str, Any]],
        total_global_samples: int,
        seed: int,
        max_samples: int,
        skip_bad_samples: bool,
    ) -> None:
        dataset = load_dataset("json", data_files=data_path, split="train")
        rows = [dict(row) for row in dataset]
        if max_samples > 0:
            rows = rows[:max_samples]
        self.pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            self.pools[infer_output_mode(row)].append(row)
        self.available_modes = [mode for mode in MODES if self.pools.get(mode)]
        if not self.available_modes:
            raise ValueError("No valid output-mode rows found in training data.")
        self.builder = builder
        self.schedule = schedule
        self.total_global_samples = total_global_samples
        self.seed = seed
        self.skip_bad_samples = skip_bad_samples
        self.max_retries = max(1, env_int("SFT_MAX_SAMPLE_RETRIES", 16))
        print(
            json.dumps(
                {
                    "curriculum_pool_sizes": {mode: len(self.pools.get(mode, [])) for mode in MODES},
                    "available_modes": self.available_modes,
                    "total_global_samples": total_global_samples,
                    "schedule": schedule,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    def __len__(self) -> int:
        return self.total_global_samples

    def __getitem__(self, position: int) -> Dict[str, Any]:
        progress = position / max(1, self.total_global_samples - 1)
        weights = normalize_weights(interpolated_weights(self.schedule, progress), self.available_modes)
        modes = list(weights)
        mode_probs = [weights[mode] for mode in modes]
        rng = random.Random(self.seed + position * 1000003)
        for retry in range(self.max_retries + 1):
            mode = rng.choices(modes, weights=mode_probs, k=1)[0]
            row = rng.choice(self.pools[mode])
            try:
                return self.builder.build(row)
            except Exception as exc:
                if not self.skip_bad_samples or retry >= self.max_retries:
                    raise
                print(f"[sft] skip bad curriculum sample mode={mode}: {exc}", flush=True)
        raise RuntimeError("unreachable")


class CurriculumTrainer(Trainer):
    """Keep virtual curriculum positions ordered; Accelerate shards them across ranks."""

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        return SequentialSampler(dataset)


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


class IPRSFTCollator:
    def __init__(self, tokenizer) -> None:
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        for feature in features:
            prompt_ids = squeeze_1d(feature["input_ids"])
            target_ids = squeeze_1d(feature["target_ids"])
            input_ids = torch.cat([prompt_ids, target_ids], dim=0)
            labels = torch.cat([torch.full_like(prompt_ids, -100), target_ids], dim=0)
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(torch.ones_like(input_ids))

        max_length = max(item.size(0) for item in input_ids_list)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for input_ids, labels, attention_mask in zip(input_ids_list, labels_list, attention_mask_list):
            pad_length = max_length - input_ids.size(0)
            batch["input_ids"].append(
                torch.cat([input_ids, torch.full((pad_length,), self.pad_token_id, dtype=torch.long)])
            )
            batch["labels"].append(torch.cat([labels, torch.full((pad_length,), -100, dtype=torch.long)]))
            batch["attention_mask"].append(
                torch.cat([attention_mask, torch.zeros((pad_length,), dtype=torch.long)])
            )
        tensor_batch: Dict[str, Any] = {key: torch.stack(value) for key, value in batch.items()}

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
            tensor_batch["pixel_values"] = collate_tensor_or_list(grouped["pixel_values"], cat=True)
        if len(grouped.get("image_grid_thw", [])) == len(features):
            tensor_batch["image_grid_thw"] = collate_tensor_or_list(grouped["image_grid_thw"], cat=True)
        if len(grouped.get("images", [])) == len(features):
            tensor_batch["images"] = grouped["images"]
        if len(grouped.get("image_sizes", [])) == len(features):
            tensor_batch["image_sizes"] = grouped["image_sizes"]
        return tensor_batch


def count_base_examples(path: str, max_samples: int) -> int:
    dataset = load_dataset("json", data_files=path, split="train")
    rows = dataset.select(range(min(len(dataset), max_samples))) if max_samples > 0 else dataset
    source_ids = {
        str(row.get("source_id") or f"physical-row-{index}")
        for index, row in enumerate(rows)
    }
    return len(source_ids)


def resolve_max_steps(train_rows: int) -> int:
    requested = env_int("MAX_STEPS", -1)
    if requested > 0:
        return requested
    world_size = max(1, int(os.getenv("WORLD_SIZE", str(env_int("NPROC_PER_NODE", 1)))))
    global_batch = (
        env_int("PER_DEVICE_TRAIN_BATCH_SIZE", 1)
        * world_size
        * env_int("GRADIENT_ACCUMULATION_STEPS", 8)
    )
    return max(1, math.ceil(env_float("NUM_TRAIN_EPOCHS", 1.0) * train_rows / global_batch))


def main() -> None:
    model_path = os.environ["MODEL_PATH"]
    train_file = os.environ["TRAIN_FILE"]
    output_dir = os.environ["OUTPUT_DIR"]
    trust_remote_code = env_bool("TRUST_REMOTE_CODE", False)
    tokenizer = get_tokenizer(model_path, trust_remote_code=trust_remote_code)
    processor = get_processor(
        model_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
        max_pixels=env_int("MAX_PIXELS", 100352),
        min_pixels=env_int("MIN_PIXELS", 50176),
    )
    if processor is None:
        raise RuntimeError("A multimodal processor is required for IPR curriculum SFT.")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model_cls = AutoModelForVision2Seq if type(config) in AutoModelForVision2Seq._model_mapping.keys() else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=os.getenv("ATTN_IMPLEMENTATION", "sdpa"),
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

    builder = IPRItemBuilder(
        model_path=model_path,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=os.getenv("PROMPT_KEY", "problem"),
        target_key=os.getenv("TARGET_KEY", "solution"),
        max_prompt_length=env_int("MAX_PROMPT_LENGTH", 4096),
        max_response_length=env_int("MAX_RESPONSE_LENGTH", 768),
        max_pixels=env_int("MAX_PIXELS", 100352),
        min_pixels=env_int("MIN_PIXELS", 50176),
        trust_remote_code=trust_remote_code,
    )
    max_samples = env_int("MAX_SAMPLES", 0)
    base_examples = count_base_examples(train_file, max_samples)
    max_steps = resolve_max_steps(base_examples)
    world_size = max(1, int(os.getenv("WORLD_SIZE", str(env_int("NPROC_PER_NODE", 1)))))
    total_global_samples = (
        max_steps
        * env_int("GRADIENT_ACCUMULATION_STEPS", 8)
        * env_int("PER_DEVICE_TRAIN_BATCH_SIZE", 1)
        * world_size
    )
    train_dataset = CurriculumMapDataset(
        data_path=train_file,
        builder=builder,
        schedule=load_schedule(),
        total_global_samples=total_global_samples,
        seed=env_int("SEED", 42),
        max_samples=max_samples,
        skip_bad_samples=env_bool("SFT_SKIP_BAD_SAMPLES", True),
    )

    eval_dataset = None
    val_file = os.getenv("VAL_FILE", "").strip()
    if val_file and Path(val_file).exists():
        eval_dataset = IPRMapDataset(
            data_path=val_file,
            builder=builder,
            max_samples=env_int("MAX_EVAL_SAMPLES", 0),
            output_mode=os.getenv("EVAL_OUTPUT_MODE", "").strip().upper(),
            skip_bad_samples=env_bool("SFT_EVAL_SKIP_BAD_SAMPLES", False),
        )

    report_to = os.getenv("REPORT_TO", "wandb").strip()
    training_kwargs = {
        "output_dir": output_dir,
        "run_name": os.getenv("RUN_NAME", "ipr-cot-curriculum-sft"),
        "report_to": [] if report_to == "none" else [report_to],
        "max_steps": max_steps,
        "per_device_train_batch_size": env_int("PER_DEVICE_TRAIN_BATCH_SIZE", 1),
        "per_device_eval_batch_size": env_int("PER_DEVICE_EVAL_BATCH_SIZE", 1),
        "gradient_accumulation_steps": env_int("GRADIENT_ACCUMULATION_STEPS", 8),
        "learning_rate": env_float("LEARNING_RATE", 2e-6),
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
    deepspeed_config = os.getenv("DEEPSPEED_CONFIG", "").strip()
    if deepspeed_config:
        training_kwargs["deepspeed"] = deepspeed_config
    if "gradient_checkpointing_kwargs" in inspect.signature(TrainingArguments).parameters:
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if eval_dataset is not None and len(eval_dataset) > 0:
        training_kwargs["eval_steps"] = env_int("EVAL_STEPS", 200)
        training_kwargs["metric_for_best_model"] = "eval_loss"
        training_kwargs["greater_is_better"] = False
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
        "data_collator": IPRSFTCollator(tokenizer),
    }
    if eval_dataset is not None and len(eval_dataset) > 0:
        trainer_kwargs["eval_dataset"] = eval_dataset
    if "processing_class" in inspect.signature(Trainer).parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor
    trainer = CurriculumTrainer(**trainer_kwargs)
    trainer.train()

    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if trainer.is_world_process_zero():
        print(
            json.dumps(
                {
                    "sft_final_model_path": str(final_dir),
                    "max_steps": max_steps,
                    "base_examples": base_examples,
                    "curriculum_preset": os.getenv("CURRICULUM_PRESET", "stage2"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
