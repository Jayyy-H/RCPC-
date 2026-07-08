# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import random
import re
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.models.transformers.qwen2_5_vl import get_rope_index


_THINK_OPEN_RE = re.compile(r"<\s*think\s*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</\s*think\s*>", re.IGNORECASE)


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        if key not in ["pixel_values", "image_grid_thw"]:
            tensors[key] = torch.stack(value, dim=0)

    return {**tensors, **non_tensors}


def process_image(image: ImageObject, max_pixels: int, min_pixels: int) -> ImageObject:
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _first_nonempty(row_dict: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = row_dict.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if value:
            return value
    return ""


def _format_webinstruct_prompt(row_dict: Dict[str, Any], prompt_key: str) -> str:
    prompt = _first_nonempty(row_dict, [prompt_key, "problem", "prompt", "instruction", "input"])
    if prompt:
        return prompt

    parts = []
    metadata = []
    discipline = _first_nonempty(row_dict, ["discipline"])
    difficulty = _first_nonempty(row_dict, ["difficulty"])
    task_type = _first_nonempty(row_dict, ["type", "question_type"])
    if discipline:
        metadata.append(f"Discipline: {discipline}")
    if difficulty:
        metadata.append(f"Difficulty: {difficulty}")
    if task_type:
        metadata.append(f"Type: {task_type}")
    if metadata:
        parts.append("\n".join(metadata))

    original_document = _first_nonempty(row_dict, ["original_document", "document", "context"])
    design_logic = _first_nonempty(row_dict, ["design_logic"])
    question = _first_nonempty(row_dict, ["question", "query"])
    if original_document:
        parts.append(f"[Context]\n{original_document}")
    if design_logic:
        parts.append(f"[Design Logic]\n{design_logic}")
    if question:
        parts.append(f"[Question]\n{question}")
    if not parts:
        raise KeyError(f"missing prompt field {prompt_key!r} and WebInstruct fallback fields")
    return "\n\n".join(parts).strip()


def _format_sft_target(row_dict: Dict[str, Any], target_key: str) -> str:
    target = _first_nonempty(
        row_dict,
        [target_key, "solution", "cot_solution", "cot", "reasoning", "response", "assistant_response", "output"],
    )
    if not target:
        raise KeyError(f"missing target field {target_key!r} and target fallback fields")
    target = _THINK_OPEN_RE.sub("<reasoning>", target)
    target = _THINK_CLOSE_RE.sub("</reasoning>", target)
    lower = target.lower()
    if _env_flag("ROPD_WRAP_UNTAGGED_TARGET", True) and "<reasoning>" not in lower:
        answer = _first_nonempty(row_dict, ["answer", "final_answer", "reference_answer"])
        if answer and "<answer>" not in lower:
            return f"<reasoning>{target.strip()}</reasoning><answer>{answer}</answer>"
        return f"<reasoning>{target.strip()}</reasoning>"
    answer = _first_nonempty(row_dict, ["answer", "final_answer", "reference_answer"])
    if answer and "<answer>" not in lower:
        return target.rstrip() + f"<answer>{answer}</answer>"
    return target


def _apply_chat_template_no_thinking(tokenizer: PreTrainedTokenizer, messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def _check_qwen_image_prompt(
    tokenizer: PreTrainedTokenizer,
    processor: ProcessorMixin,
    prompt: str,
    image_grid_thw: torch.Tensor,
    max_prompt_length: int,
) -> None:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) > max_prompt_length:
        raise ValueError(
            f"expanded multimodal prompt length {len(prompt_ids)} exceeds max_prompt_length={max_prompt_length}"
        )

    image_token = getattr(processor, "image_token", None)
    if image_token is None:
        return

    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    merge_length = processor.image_processor.merge_size**2
    expected = sum(int(grid.prod().item() // merge_length) for grid in image_grid_thw)
    actual = sum(1 for token_id in prompt_ids if token_id == image_token_id)
    if actual != expected:
        raise ValueError(f"image token mismatch: actual={actual}, expected={expected}")


class SFTRLDataset(Dataset):
    """Qwen2.5-VL style dataset with an extra expert target trajectory."""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key="prompt",
        target_key="solution",
        max_prompt_length=1024,
        truncation="error",
        max_pixels=None,
        min_pixels=None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.target_key = target_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.skip_bad_samples = _env_flag("ROPD_SKIP_BAD_SAMPLES", True)
        self.max_sample_retries = _env_int("ROPD_MAX_SAMPLE_RETRIES", 16)
        self.print_bad_sample_traceback = _env_flag("ROPD_PRINT_BAD_SAMPLE_TRACEBACK", False)

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.exists(data_path):
            ext = os.path.splitext(data_path)[-1]
            if ext == ".parquet":
                self.dataset = load_dataset("parquet", data_files=data_path, split=data_split)
            elif ext in [".json", ".jsonl"]:
                self.dataset = load_dataset("json", data_files=data_path, split=data_split)
            else:
                raise NotImplementedError()
        else:
            self.dataset = load_dataset(data_path, split=data_split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if not self.skip_bad_samples:
            return self._build_item(index)

        last_error = None
        for _ in range(self.max_sample_retries):
            try:
                return self._build_item(index)
            except Exception as exc:
                last_error = exc
                if self.print_bad_sample_traceback:
                    traceback.print_exc()
                print(f" >>> skipping bad sample index={index}")
                print(f" >>> reason {exc}")
                index = random.randint(0, self.__len__() - 1)

        raise RuntimeError(f"failed to load a valid sample after {self.max_sample_retries} retries") from last_error

    def _build_item(self, index):
        row_dict = dict(self.dataset[index])
        prompt_text = _format_webinstruct_prompt(row_dict, self.prompt_key)
        target_text = _format_sft_target(row_dict, self.target_key)

        messages = [{"role": "user", "content": prompt_text}]
        prompt = _apply_chat_template_no_thinking(self.tokenizer, messages)

        if "image" in row_dict:
            row_dict["images"] = row_dict["image"]
            del row_dict["image"]

        if "images" not in row_dict or row_dict["images"] is None:
            row_dict["images"] = None
        elif type(row_dict["images"]) != list:
            row_dict["images"] = [row_dict["images"]]
        elif len(row_dict["images"]) == 0:
            row_dict["images"] = None

        image_paths = []
        if row_dict["images"] is not None:
            image_paths = [image for image in row_dict["images"] if isinstance(image, str)]

        if row_dict["images"] is not None:
            placeholder_count = prompt.count("<image>")
            if placeholder_count != len(row_dict["images"]):
                raise ValueError(
                    f"image placeholder count {placeholder_count} does not match image count {len(row_dict['images'])}"
                )

            raw_prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")

            if type(row_dict["images"][0]) == str:
                row_dict["images"] = [
                    process_image(Image.open(image), self.max_pixels, self.min_pixels) for image in row_dict["images"]
                ]
            else:
                row_dict["images"] = [
                    process_image(image, self.max_pixels, self.min_pixels) for image in row_dict["images"]
                ]

            image_inputs = self.processor.image_processor(row_dict["images"], return_tensors="pt")
            image_grid_thw = image_inputs["image_grid_thw"]
            row_dict.update(image_inputs)

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                image_index = 0
                while "<image>" in prompt:
                    prompt = prompt.replace(
                        "<image>",
                        "<|vision_start|>"
                        + "<|placeholder|>" * (image_grid_thw[image_index].prod() // merge_length)
                        + "<|vision_end|>",
                        1,
                    )
                    image_index += 1

                prompt = prompt.replace("<|placeholder|>", self.processor.image_token)
                if image_index != len(image_grid_thw):
                    raise ValueError(f"expanded {image_index} image placeholders but got {len(image_grid_thw)} images")
                _check_qwen_image_prompt(self.tokenizer, self.processor, prompt, image_grid_thw, self.max_prompt_length)
        else:
            raw_prompt = prompt

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if row_dict["images"] is not None:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
        if isinstance(target_ids, torch.Tensor):
            target_ids = target_ids.tolist()
        elif isinstance(target_ids, np.ndarray):
            target_ids = target_ids.tolist()
        elif isinstance(target_ids, int):
            target_ids = [target_ids]

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        assert eos_id is not None
        if len(target_ids) == 0 or target_ids[-1] != eos_id:
            target_ids.append(eos_id)

        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        row_dict["raw_prompt"] = prompt_text
        row_dict["image_paths"] = image_paths
        row_dict["target_ids"] = target_ids
        row_dict["answer"] = _first_nonempty(row_dict, ["answer", "final_answer", "reference_answer"])
        return row_dict
