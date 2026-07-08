#!/usr/bin/env python3
"""Text-only Qwen3 CoT SFT cold start for RCPC.

The script intentionally uses a text-only Qwen3 pipeline. It expects a JSON/JSONL
dataset with a prompt field and a CoT target field, while keeping a few fallbacks
for raw WebInstruct-style rows.
"""

from __future__ import annotations

import inspect
import json
import os
import random
import re
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments


_THINK_OPEN_RE = re.compile(r"<\s*think\s*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</\s*think\s*>", re.IGNORECASE)
_REASONING_RE = re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def split_path_and_split(data_path: str, default_split: str = "train") -> Tuple[str, str]:
    if "@" in data_path:
        path, split = data_path.rsplit("@", 1)
        return path, split
    return data_path, default_split


def load_any_dataset(data_path: str, default_split: str = "train") -> Dataset:
    path, split = split_path_and_split(data_path, default_split)
    if os.path.exists(path):
        ext = Path(path).suffix.lower()
        if ext == ".parquet":
            return load_dataset("parquet", data_files=path, split=split)
        if ext in {".json", ".jsonl"}:
            return load_dataset("json", data_files=path, split=split)
        raise ValueError(f"Unsupported local dataset extension: {path}")
    return load_dataset(path, split=split)


def first_nonempty(row: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        value = value.strip()
        if value:
            return value
    return ""


def format_problem_from_webinstruct(row: Dict[str, Any], prompt_key: str) -> str:
    prompt = first_nonempty(row, [prompt_key, "problem", "prompt", "instruction", "input"])
    if prompt:
        return prompt

    parts: List[str] = []
    discipline = first_nonempty(row, ["discipline"])
    difficulty = first_nonempty(row, ["difficulty"])
    task_type = first_nonempty(row, ["type", "question_type"])
    original_document = first_nonempty(row, ["original_document", "document", "context"])
    design_logic = first_nonempty(row, ["design_logic"])
    question = first_nonempty(row, ["question", "query"])

    metadata = []
    if discipline:
        metadata.append(f"Discipline: {discipline}")
    if difficulty:
        metadata.append(f"Difficulty: {difficulty}")
    if task_type:
        metadata.append(f"Type: {task_type}")
    if metadata:
        parts.append("\n".join(metadata))
    if original_document:
        parts.append(f"[Context]\n{original_document}")
    if design_logic:
        parts.append(f"[Design Logic]\n{design_logic}")
    if question:
        parts.append(f"[Question]\n{question}")

    if not parts:
        raise ValueError("Could not build prompt: no prompt/problem/question/context fields found.")
    return "\n\n".join(parts).strip()


def maybe_wrap_question(text: str) -> str:
    text = text.strip()
    if not env_bool("SFT_WRAP_QUESTION_TAGS", True):
        return text
    lowered = text.lower()
    if "<question>" in lowered and "</question>" in lowered:
        return text
    return f"<question>\n{text}\n</question>"


def sft_response_instruction() -> str:
    return os.getenv(
        "SFT_RESPONSE_INSTRUCTION",
        "Please solve the problem and output exactly one response in this format:\n"
        "<reasoning>\n"
        "Your concise reasoning.\n"
        "</reasoning>\n"
        "<answer>\n"
        "Your final answer.\n"
        "</answer>\n"
        "You must close </reasoning> before starting <answer>. Do not output <think>, </think>, "
        "<tool_call>, tool-use markup, function-call markup, markdown code fences, or any text "
        "outside the <reasoning>...</reasoning><answer>...</answer> tags.",
    ).strip()


def append_response_instruction(problem: str) -> str:
    instruction = sft_response_instruction()
    return f"{problem.strip()}\n\n{instruction}" if instruction else problem.strip()


def apply_chat_template_no_thinking(tokenizer, messages: Sequence[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def normalize_reasoning_tags(text: str) -> str:
    text = _THINK_OPEN_RE.sub("<reasoning>", text)
    text = _THINK_CLOSE_RE.sub("</reasoning>", text)
    return text


def normalize_target(row: Dict[str, Any], target_key: str, answer_key: str, wrap_untagged: bool) -> str:
    target = first_nonempty(
        row,
        [
            target_key,
            "solution",
            "cot_solution",
            "cot",
            "reasoning",
            "response",
            "assistant_response",
            "output",
        ],
    )
    answer = first_nonempty(row, [answer_key, "answer", "final_answer", "reference_answer"])

    if not target:
        raise ValueError("Could not build target: no solution/response/cot fields found.")

    target = normalize_reasoning_tags(target)
    has_reasoning = bool(_REASONING_RE.search(target))
    has_answer = bool(_ANSWER_RE.search(target))

    if wrap_untagged and not has_reasoning:
        if answer and not has_answer:
            target = f"<reasoning>\n{target.strip()}\n</reasoning>\n<answer>\n{answer}\n</answer>"
        else:
            target = f"<reasoning>\n{target.strip()}\n</reasoning>"
        has_answer = bool(_ANSWER_RE.search(target))

    if answer and not has_answer:
        target = target.rstrip() + f"\n<answer>\n{answer}\n</answer>"

    return target.strip()


class Qwen3CotSFTDataset(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer,
        *,
        prompt_key: str,
        target_key: str,
        answer_key: str,
        max_prompt_length: int,
        max_response_length: int,
        skip_bad_samples: bool,
        max_sample_retries: int,
        truncate_target: bool,
        wrap_untagged_target: bool,
        print_bad_traceback: bool,
        audit_prefix: str,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.target_key = target_key
        self.answer_key = answer_key
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.skip_bad_samples = skip_bad_samples
        self.max_sample_retries = max(1, max_sample_retries)
        self.truncate_target = truncate_target
        self.wrap_untagged_target = wrap_untagged_target
        self.print_bad_traceback = print_bad_traceback
        self.audit_prefix = audit_prefix

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not self.skip_bad_samples:
            return self._build_item(index)

        last_error: Optional[BaseException] = None
        original_index = index
        for _ in range(self.max_sample_retries):
            try:
                return self._build_item(index)
            except Exception as exc:  # noqa: BLE001 - dataset resilience during long distributed runs.
                last_error = exc
                if self.print_bad_traceback:
                    traceback.print_exc()
                print(
                    f"[{self.audit_prefix}] skipping bad sample index={index}; reason={exc}",
                    flush=True,
                )
                index = random.randint(0, len(self.dataset) - 1)
        raise RuntimeError(
            f"failed to load a valid sample after {self.max_sample_retries} retries; original_index={original_index}"
        ) from last_error

    def _build_item(self, index: int) -> Dict[str, Any]:
        row = dict(self.dataset[index])
        problem = maybe_wrap_question(format_problem_from_webinstruct(row, self.prompt_key))
        user_content = append_response_instruction(problem)
        target = normalize_target(row, self.target_key, self.answer_key, self.wrap_untagged_target)

        messages = [{"role": "user", "content": user_content}]
        prompt = apply_chat_template_no_thinking(self.tokenizer, messages)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > self.max_prompt_length:
            raise ValueError(f"prompt length {len(prompt_ids)} exceeds max_prompt_length={self.max_prompt_length}")

        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("tokenizer.eos_token_id is required")
        if not target_ids or target_ids[-1] != eos_id:
            target_ids.append(eos_id)

        if len(target_ids) > self.max_response_length:
            if not self.truncate_target:
                raise ValueError(
                    f"target length {len(target_ids)} exceeds max_response_length={self.max_response_length}"
                )
            target_ids = target_ids[: self.max_response_length]
            target_ids[-1] = eos_id

        return {
            "input_ids": prompt_ids + target_ids,
            "labels": [-100] * len(prompt_ids) + target_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
            "prompt_length": len(prompt_ids),
            "target_length": len(target_ids),
        }


class DataCollatorForCausalCot:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        input_ids = []
        attention_mask = []
        labels = []
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def maybe_select(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return dataset.select(indices[:max_samples])


def build_train_val_datasets(seed: int) -> Tuple[Dataset, Optional[Dataset]]:
    train_file = os.environ["TRAIN_FILE"]
    val_file = os.getenv("VAL_FILE", "").strip()
    val_split_size = env_int("VAL_SPLIT_SIZE", 200)

    train_dataset = load_any_dataset(train_file, "train")
    if val_file:
        val_dataset = load_any_dataset(val_file, "train")
    elif val_split_size > 0 and len(train_dataset) > val_split_size:
        split = train_dataset.train_test_split(test_size=val_split_size, seed=seed, shuffle=True)
        train_dataset = split["train"]
        val_dataset = split["test"]
    else:
        val_dataset = None

    train_dataset = maybe_select(train_dataset, env_int("MAX_SAMPLES", 0), seed)
    if val_dataset is not None:
        val_dataset = maybe_select(val_dataset, env_int("MAX_EVAL_SAMPLES", 0), seed + 1)
    return train_dataset, val_dataset


def audit_dataset(name: str, dataset: Dataset) -> None:
    print(
        json.dumps(
            {"dataset": name, "rows": len(dataset), "columns": list(dataset.column_names)},
            ensure_ascii=False,
        ),
        flush=True,
    )


def resolve_training_arg_names(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    signature = inspect.signature(TrainingArguments.__init__)
    params = signature.parameters

    if "eval_strategy" in params and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    elif "evaluation_strategy" in params and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    filtered = {key: value for key, value in kwargs.items() if key in params}
    return filtered


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    path = Path(output_dir)
    if not path.exists():
        return None
    candidates = []
    for child in path.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        try:
            step = int(child.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        candidates.append((step, child))
    if not candidates:
        return None
    return str(max(candidates, key=lambda item: item[0])[1])


def load_model(model_path: str, trust_remote_code: bool, attn_implementation: str):
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    allow_fallback = env_bool("ALLOW_ATTN_FALLBACK", True)
    try:
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except Exception:
        if not allow_fallback or attn_implementation in {"", "sdpa"}:
            raise
        print(
            f"[sft] failed to load with attn_implementation={attn_implementation!r}; retrying with sdpa",
            flush=True,
        )
        kwargs["attn_implementation"] = "sdpa"
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def build_generation_samples(
    dataset: Dataset,
    tokenizer,
    *,
    prompt_key: str,
    count: int,
) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    for index in range(len(dataset)):
        if len(samples) >= count:
            break
        try:
            row = dict(dataset[index])
            problem = maybe_wrap_question(format_problem_from_webinstruct(row, prompt_key))
            user_content = append_response_instruction(problem)
            prompt = apply_chat_template_no_thinking(tokenizer, [{"role": "user", "content": user_content}])
            samples.append({"index": str(index), "problem": problem, "prompt": prompt})
        except Exception as exc:  # noqa: BLE001 - sample printing must not break training.
            print(f"[sft sample generation] skip preview sample index={index}; reason={exc}", flush=True)
    return samples


def is_rank_zero_process(args) -> bool:
    process_index = getattr(args, "process_index", None)
    if process_index is not None:
        return int(process_index) == 0
    return int(os.getenv("RANK", "0")) == 0


def unwrap_generation_model(model):
    if hasattr(model, "generate"):
        return model
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "generate"):
        return module
    return model


class PeriodicCotGenerationCallback(TrainerCallback):
    def __init__(
        self,
        *,
        tokenizer,
        samples: Sequence[Dict[str, str]],
        every_steps: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        prompt_preview_chars: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.samples = list(samples)
        self.every_steps = max(0, int(every_steps))
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.do_sample = bool(do_sample)
        self.prompt_preview_chars = max(0, int(prompt_preview_chars))

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_steps <= 0 or not self.samples:
            return control
        if state.global_step <= 0 or state.global_step % self.every_steps != 0:
            return control
        if not is_rank_zero_process(args):
            return control

        model = kwargs.get("model")
        if model is None:
            return control
        generation_model = unwrap_generation_model(model)
        was_training = bool(getattr(generation_model, "training", False))
        generation_model.eval()

        try:
            device = next(generation_model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[sft sample generation] step={state.global_step} begin", flush=True)
        with torch.no_grad():
            for sample_number, sample in enumerate(self.samples, start=1):
                prompt = sample["prompt"]
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                generation_kwargs = dict(
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                if self.do_sample:
                    generation_kwargs["temperature"] = self.temperature
                    generation_kwargs["top_p"] = self.top_p
                generated = generation_model.generate(
                    **inputs,
                    **generation_kwargs,
                )
                new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
                output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                prompt_preview = sample["problem"]
                if self.prompt_preview_chars and len(prompt_preview) > self.prompt_preview_chars:
                    prompt_preview = prompt_preview[: self.prompt_preview_chars] + "...[truncated]"
                print(
                    f"[sft sample generation] step={state.global_step} sample={sample_number} "
                    f"dataset_index={sample['index']} prompt_preview:\n{prompt_preview}",
                    flush=True,
                )
                print(
                    f"[sft sample generation] step={state.global_step} sample={sample_number} model_cot:\n{output}",
                    flush=True,
                )
        print(f"[sft sample generation] step={state.global_step} end", flush=True)

        if was_training:
            generation_model.train()
        return control


def main() -> None:
    seed = env_int("SEED", 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_path = os.environ["MODEL_PATH"]
    output_dir = os.environ["OUTPUT_DIR"]
    trust_remote_code = env_bool("TRUST_REMOTE_CODE", True)
    attn_implementation = os.getenv("ATTN_IMPLEMENTATION", "flash_attention_2").strip()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        use_fast=env_bool("TOKENIZER_USE_FAST", True),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_raw, val_raw = build_train_val_datasets(seed)
    if env_bool("SFT_AUDIT_DATASETS", True):
        audit_dataset("train", train_raw)
        if val_raw is not None:
            audit_dataset("val", val_raw)

    dataset_kwargs = {
        "tokenizer": tokenizer,
        "prompt_key": os.getenv("PROMPT_KEY", "question"),
        "target_key": os.getenv("TARGET_KEY", "cot"),
        "answer_key": os.getenv("ANSWER_KEY", "answer"),
        "max_prompt_length": env_int("MAX_PROMPT_LENGTH", 8192),
        "max_response_length": env_int("MAX_RESPONSE_LENGTH", 4096),
        "skip_bad_samples": env_bool("SFT_SKIP_BAD_SAMPLES", True),
        "max_sample_retries": env_int("SFT_MAX_SAMPLE_RETRIES", 32),
        "truncate_target": env_bool("SFT_TRUNCATE_TARGET", False),
        "wrap_untagged_target": env_bool("SFT_WRAP_UNTAGGED_TARGET", True),
        "print_bad_traceback": env_bool("SFT_PRINT_BAD_SAMPLE_TRACEBACK", False),
    }
    train_dataset = Qwen3CotSFTDataset(train_raw, audit_prefix="sft-train", **dataset_kwargs)
    eval_dataset = (
        Qwen3CotSFTDataset(
            val_raw,
            audit_prefix="sft-val",
            **{**dataset_kwargs, "skip_bad_samples": env_bool("SFT_EVAL_SKIP_BAD_SAMPLES", True)},
        )
        if val_raw is not None
        else None
    )

    model = load_model(model_path, trust_remote_code, attn_implementation)
    model.config.use_cache = False
    if env_bool("SFT_GRADIENT_CHECKPOINTING", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    report_to_raw = os.getenv("REPORT_TO", "wandb").strip()
    report_to: List[str]
    if report_to_raw.lower() in {"", "none", "disabled", "false", "0"}:
        report_to = []
    else:
        report_to = [item.strip() for item in report_to_raw.split(",") if item.strip()]

    deepspeed_config = os.getenv("DEEPSPEED_CONFIG", "").strip()
    if deepspeed_config and not Path(deepspeed_config).exists():
        raise FileNotFoundError(f"DEEPSPEED_CONFIG does not exist: {deepspeed_config}")

    eval_strategy = "steps" if eval_dataset is not None else "no"
    max_steps = env_int("MAX_STEPS", -1)
    args = {
        "output_dir": output_dir,
        "overwrite_output_dir": env_bool("OVERWRITE_OUTPUT_DIR", False),
        "run_name": os.getenv("RUN_NAME", "qwen3-4b-webinstruct-cot-sft"),
        "report_to": report_to,
        "num_train_epochs": env_float("NUM_TRAIN_EPOCHS", 1.0),
        "max_steps": max_steps,
        "per_device_train_batch_size": env_int("PER_DEVICE_TRAIN_BATCH_SIZE", 1),
        "per_device_eval_batch_size": env_int("PER_DEVICE_EVAL_BATCH_SIZE", 1),
        "gradient_accumulation_steps": env_int("GRADIENT_ACCUMULATION_STEPS", 8),
        "learning_rate": env_float("LEARNING_RATE", 5e-6),
        "weight_decay": env_float("WEIGHT_DECAY", 0.01),
        "warmup_ratio": env_float("WARMUP_RATIO", 0.03),
        "max_grad_norm": env_float("MAX_GRAD_NORM", 1.0),
        "logging_steps": env_int("LOGGING_STEPS", 1),
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": env_int("SAVE_STEPS", 20),
        "save_total_limit": env_int("SAVE_TOTAL_LIMIT", 4),
        "evaluation_strategy": eval_strategy,
        "eval_steps": env_int("EVAL_STEPS", 20),
        "bf16": True,
        "fp16": False,
        "remove_unused_columns": False,
        "dataloader_num_workers": env_int("DATALOADER_NUM_WORKERS", 2),
        "dataloader_pin_memory": True,
        "ddp_find_unused_parameters": False,
        "gradient_checkpointing": env_bool("SFT_GRADIENT_CHECKPOINTING", True),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "deepspeed": deepspeed_config or None,
        "optim": os.getenv("OPTIM", "adamw_torch"),
        "seed": seed,
    }
    training_args = TrainingArguments(**resolve_training_arg_names(args))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForCausalCot(tokenizer),
        tokenizer=tokenizer,
    )

    generation_every = env_int("SFT_PRINT_GENERATION_EVERY", 5)
    generation_count = env_int("SFT_PRINT_GENERATION_COUNT", 3)
    if generation_every > 0 and generation_count > 0:
        generation_source = val_raw if env_bool("SFT_PRINT_GENERATION_FROM_VAL", True) and val_raw is not None else train_raw
        samples = build_generation_samples(
            generation_source,
            tokenizer,
            prompt_key=dataset_kwargs["prompt_key"],
            count=generation_count,
        )
        trainer.add_callback(
            PeriodicCotGenerationCallback(
                tokenizer=tokenizer,
                samples=samples,
                every_steps=generation_every,
                max_new_tokens=env_int("SFT_PRINT_GENERATION_MAX_NEW_TOKENS", 1024),
                temperature=env_float("SFT_PRINT_GENERATION_TEMPERATURE", 0.0),
                top_p=env_float("SFT_PRINT_GENERATION_TOP_P", 1.0),
                do_sample=env_bool("SFT_PRINT_GENERATION_DO_SAMPLE", False),
                prompt_preview_chars=env_int("SFT_PRINT_GENERATION_PROMPT_PREVIEW_CHARS", 1200),
            )
        )

    resume = os.getenv("RESUME_FROM_CHECKPOINT", "").strip()
    if resume.lower() == "auto":
        resume = find_latest_checkpoint(output_dir) or ""
    trainer.train(resume_from_checkpoint=resume or None)

    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[sft] saved final checkpoint to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
