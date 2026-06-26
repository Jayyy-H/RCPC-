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

import os
import math
import numpy as np
import torch
import copy
import random
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional
from PIL import Image
from PIL.Image import Image as ImageObject
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin, AutoProcessor

def inference_left_pad(input_ids, batch_first, padding_value):
    max_len = max([id.shape[0] for id in input_ids])
    pad_tensor = []
    for id in input_ids:
        pad_number = max_len - id.shape[0]
        id = torch.cat([torch.tensor([padding_value] * pad_number).long(), id])
        pad_tensor.append(id)
    return torch.stack(pad_tensor, dim=0)

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

def _sequence_length(input_ids) -> int:
    if isinstance(input_ids, torch.Tensor):
        return input_ids.shape[-1]
    if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)

def collate_fn(instances: List[Dict[str, Any]]) -> Dict[str, Any]:
    pad_token_id = instances[0]["pad_token_id"]
    input_ids = [instance["input_ids"][0] for instance in instances]
    input_ids = inference_left_pad(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = input_ids.ne(pad_token_id)
    position_ids = torch.zeros_like(input_ids)
    for i in range(position_ids.shape[0]):
        cur_len = torch.sum(attention_mask[i]).item()
        position_ids[i, -cur_len:] = torch.arange(0, cur_len)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids
    }

    if "images" in instances[0]:
        batch["images"] = [instance["images"][0] for instance in instances]
    
    if "image_sizes" in instances[0]:
        batch["image_sizes"] = [instance["image_sizes"][0] for instance in instances]

    if "pixel_values" in instances[0]:
        batch["pixel_values"] = [instance["pixel_values"] for instance in instances]
        batch["image_grid_thw"] = [instance["image_grid_thw"] for instance in instances]

    if "answer" in instances[0]:
        batch["answer"] = [instance["answer"] for instance in instances]

    if "raw_prompt" in instances[0]:
        batch["raw_prompt"] = [instance["raw_prompt"] for instance in instances]

    if "image_paths" in instances[0]:
        batch["image_paths"] = [instance["image_paths"] for instance in instances]
    
    if "raw_prompt_ids" in instances[0]:
        batch["raw_prompt_ids"] = [instance["raw_prompt_ids"] for instance in instances]
    
    if "navit_processed_images" in instances[0]:
        batch["navit_processed_images"] = [instance["navit_processed_images"] for instance in instances]

    if "target_ids" in instances[0]:
        batch["target_ids"] = [instance["target_ids"] for instance in instances]

    return batch


class SFTRLDatasetValley(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

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
        **kwargs
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
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = dict(self.dataset[index])

        if "image" in row_dict:
            row_dict["images"] = row_dict["image"]
            del row_dict["image"]
        
        if "images" not in row_dict:
            row_dict["images"] = None
        elif type(row_dict["images"]) != list:
            row_dict["images"] = [row_dict["images"]]

        image_paths = []
        if row_dict["images"] is not None:
            image_paths = [image for image in row_dict["images"] if isinstance(image, str)]

        # processed_data = self.processor({
        #     "conversations": [
        #         {"role": "system", "content": r"Please reason step by step, and put your final answer within \boxed{}."},
        #         {"role": "user", "content": row_dict[self.prompt_key] + r"Please reason step by step, first output your thinking process, and then put your final answer within \boxed{}."},
        #     ],
        #     "images": row_dict["images"] 
        # })
        processed_data = self.processor({
            "conversations": [
                {"role": "user", "content": row_dict[self.prompt_key]},
            ],
            "images": row_dict["images"] 
        })
        prompt_length = _sequence_length(processed_data["input_ids"])
        if prompt_length > self.max_prompt_length:
            raise ValueError(f"processed prompt length {prompt_length} exceeds max_prompt_length={self.max_prompt_length}")

        processed_data["pad_token_id"] = self.tokenizer.pad_token_id
        processed_data["answer"] = row_dict["answer"]
        processed_data["raw_prompt"] = row_dict[self.prompt_key]
        processed_data["image_paths"] = image_paths
        
        if type(row_dict["images"][0]) == str:
            processed_data["navit_processed_images"] = [
                process_image(Image.open(image), self.max_pixels, self.min_pixels) for image in row_dict["images"]
            ]
        else:
            processed_data["navit_processed_images"] = [
                process_image(image, self.max_pixels, self.min_pixels) for image in row_dict["images"]
            ]

        vision_token_id = self.tokenizer.encode("<|vision_pad|>")[0]
        processed_data["raw_prompt_ids"] = [
            i if i != -200 else vision_token_id 
            for i in processed_data["input_ids"].tolist()[0]
        ]

        # if processed_data["images"][0] is None, we need to add a dummy image
        if processed_data["images"][0] is None:
            processed_data["images"][0] = torch.zeros((len(row_dict["images"]), 3, 10, 10))

        # 添加target_ids
        target_ids = self.tokenizer.encode(
            row_dict[self.target_key],
            add_special_tokens=False
        )

        if isinstance(target_ids, torch.Tensor):
            target_ids = target_ids.tolist()
        elif isinstance(target_ids, np.ndarray):
            target_ids = target_ids.tolist()
        elif isinstance(target_ids, int):  # 单个 token 的情况
            target_ids = [target_ids]
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        assert eos_id is not None
        if (len(target_ids) == 0 or target_ids[-1] != eos_id):
            target_ids.append(eos_id)
        
        processed_data["target_ids"] = target_ids
        return processed_data
