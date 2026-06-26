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
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.models.transformers.qwen2_5_vl import get_rope_index


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


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key="prompt",
        max_prompt_length=1024,
        truncation="error",
        max_pixels=None,
        min_pixels=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.skip_bad_samples = _env_flag("ROPD_SKIP_BAD_SAMPLES", True)
        self.max_sample_retries = _env_int("ROPD_MAX_SAMPLE_RETRIES", 16)

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
                traceback.print_exc()
                print(f" >>> failed to load sample index={index}")
                print(f" >>> reason {exc}")
                index = random.randint(0, self.__len__() - 1)

        raise RuntimeError(f"failed to load a valid sample after {self.max_sample_retries} retries") from last_error

    def _build_item(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = dict(self.dataset[index])
        messages = [
            {"role": "system", "content": r"Please reason step by step, and put your final answer within \boxed{}."},
            {"role": "user", "content": row_dict[self.prompt_key] + r"Please reason step by step, first output your thinking process, and then put your final answer within \boxed{}."},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        if "image" in row_dict: # Robust
            row_dict["images"] = row_dict["image"]
            del row_dict["image"]

        if "images" not in row_dict or row_dict["images"] is None:
            row_dict["images"] = None
        elif type(row_dict["images"]) != list:
            row_dict["images"] = [row_dict["images"]]
        elif len(row_dict["images"]) == 0:
            row_dict["images"] = None

        if row_dict["images"] is not None:  # expand image token
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
            )  # (3, seq_len)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seqlen,)

        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        return row_dict
