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
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single GPU model.
Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model to perform generation.
"""
import contextlib
from contextlib import contextmanager
import torch
import torch.distributed
from tensordict import TensorDict
from typing import Any, List, Union
from torch import nn
from copy import deepcopy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from .base import BaseRollout

from transformers import GenerationConfig, AutoModel, PreTrainedTokenizer
from verl.workers.rollout.config import RolloutConfig
from contextlib import contextmanager
from typing import Any, List, Union

import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.config import RolloutConfig

__all__ = ['HFRollout']


def _repeat_interleave(features: Union[torch.Tensor, List[Any]], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(features, torch.Tensor):
        return features.repeat_interleave(repeats, dim=0)
    else:
        return [feature for feature in features for _ in range(repeats)]

class HFRolloutValley(BaseRollout):

    def __init__(self, module: nn.Module, config: RolloutConfig, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.config = config
        self.module = module
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def _generation_config(self, do_sample: bool) -> tuple[GenerationConfig, int]:
        if not do_sample:
            return (
                GenerationConfig(
                    num_return_sequences=1,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=-1,
                    max_new_tokens=self.config.response_length,
                ),
                1,
            )

        return (
            GenerationConfig(
                do_sample=True,
                num_return_sequences=self.config.n,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                # top_k=self.config.top_k,
                max_new_tokens=self.config.response_length,
            ),
            self.config.n,
        )

    def _repeat_for_rollout(self, value: Union[torch.Tensor, List[Any]], generated_per_prompt: int, include_target: bool):
        repeats = generated_per_prompt + int(include_target)
        return _repeat_interleave(value, repeats)

    @torch.no_grad()
    def _generate_sequences(self, prompts: DataProto, include_targets: bool, **kwargs) -> DataProto:
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        images: list = prompts.non_tensor_batch["images"]
        image_sizes: list = prompts.non_tensor_batch["image_sizes"]
        pixel_values: list = prompts.non_tensor_batch["pixel_values"]
        image_grid_thw: list = prompts.non_tensor_batch["image_grid_thw"]
        non_tensor_batch = prompts.non_tensor_batch
        target_ids = non_tensor_batch.pop("target_ids", None) if include_targets else None
        non_tensor_batch.pop("raw_prompt_ids", None)
        non_tensor_batch.pop("navit_processed_images", None)
        batch_size = input_ids.size(0)

        do_sample = prompts.meta_info.get("do_sample", True)
        generation_config, generated_per_prompt = self._generation_config(do_sample)
        gen_image_grid_thw = image_grid_thw
        gen_pixel_values = pixel_values
        gen_images = images
        gen_image_sizes = image_sizes
        if generated_per_prompt > 1:
            gen_image_grid_thw = _repeat_interleave(image_grid_thw, generated_per_prompt)
            gen_pixel_values = _repeat_interleave(pixel_values, generated_per_prompt)
            gen_images = _repeat_interleave(images, generated_per_prompt)
            gen_image_sizes = _repeat_interleave(image_sizes, generated_per_prompt)

        self.module.eval()
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                self.module.to(device="cuda", dtype=torch.bfloat16)
                self.module.eval()
                
                # Transfer the `image` to the GPU.
                if type(gen_images[0]) == list:
                    images_copy = [[x.to(device="cuda", dtype=torch.bfloat16) for x in img] for img in deepcopy(gen_images)]
                else:
                    images_copy = [img.to(device="cuda", dtype=torch.bfloat16) for img in deepcopy(gen_images)]
                
                # Generate responses
                prompt_completion_ids = self.module.generate(
                    input_ids=input_ids.to(device="cuda"),
                    attention_mask=attention_mask.to(device="cuda"),
                    images=images_copy,
                    image_sizes=gen_image_sizes,
                    pixel_values=torch.cat(gen_pixel_values, dim=0).to(device="cuda", dtype=torch.bfloat16),
                    image_grid_thw=torch.cat(gen_image_grid_thw, dim=0).to(device="cuda"),
                    do_sample=do_sample,
                    generation_config=generation_config,
                    use_cache=True)

        # Construct prompt_ids, completion_ids, prompt_completion_ids
        prompt_completion_ids = prompt_completion_ids.to(input_ids.device)
        prompt_length = input_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completion_ids = pad_2d_list_to_length(
            completion_ids.tolist(), self.pad_token_id, max_length=self.config.response_length
        ).to(input_ids.device)

        # # ========== DEBUG ======== #
        # input_token_len = input_ids.shape[1]
        # generation_text = self.tokenizer.batch_decode(prompt_completion_ids[:, input_token_len:])[0]
        # generation_text = generation_text.replace("<|im_end|>", "")
        # generation_text = generation_text.replace("<|endoftext|>", "")
        # print(generation_text)
        # # ========== DEBUG ======== #

        include_target = target_ids is not None
        if include_target:
            target_tensor = pad_2d_list_to_length(
                target_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            prompt_ids = prompt_ids.view(batch_size, generated_per_prompt, prompt_length)
            completion_ids = completion_ids.view(batch_size, generated_per_prompt, -1)
            interleaved_prompt_ids = []
            interleaved_completion_ids = []
            is_onpolicy = []
            for i in range(batch_size):
                for j in range(generated_per_prompt):
                    interleaved_prompt_ids.append(prompt_ids[i, j])
                    interleaved_completion_ids.append(completion_ids[i, j])
                    is_onpolicy.append(True)
                interleaved_prompt_ids.append(input_ids[i])
                interleaved_completion_ids.append(target_tensor[i])
                is_onpolicy.append(False)
            prompt_ids = torch.stack(interleaved_prompt_ids, dim=0)
            completion_ids = torch.stack(interleaved_completion_ids, dim=0)
        else:
            is_onpolicy = [True] * completion_ids.size(0)

        attention_mask = self._repeat_for_rollout(attention_mask, generated_per_prompt, include_target)
        position_ids = self._repeat_for_rollout(position_ids, generated_per_prompt, include_target)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        batch_size = prompt_completion_ids.size(0)
        
        completion_mask = get_eos_mask(
            response_ids=completion_ids, 
            eos_token=self.tokenizer.eos_token_id,
            dtype=attention_mask.dtype
        )
        attention_mask = torch.cat([attention_mask, completion_mask], dim=1)

        # Construct batch and non_tensor_batch, ready to compute logps and update policy model
        batch = TensorDict(
            {
                'prompts': prompt_ids,
                'responses': completion_ids,
                'input_ids': prompt_completion_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size
        )
        non_tensor_batch["image_grid_thw"] = self._repeat_for_rollout(image_grid_thw, generated_per_prompt, include_target)
        non_tensor_batch["pixel_values"] = self._repeat_for_rollout(pixel_values, generated_per_prompt, include_target)
        non_tensor_batch["images"] = self._repeat_for_rollout(images, generated_per_prompt, include_target)
        non_tensor_batch["image_sizes"] = self._repeat_for_rollout(image_sizes, generated_per_prompt, include_target)
        non_tensor_batch["is_onpolicy"] = is_onpolicy

        # empty cache before compute old_log_prob
        torch.cuda.empty_cache()

        self.module.train()
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        return self._generate_sequences(prompts=prompts, include_targets=True, **kwargs)

    @torch.no_grad()
    def generate_sequences_val(self, prompts: DataProto, **kwargs) -> DataProto:
        return self._generate_sequences(prompts=prompts, include_targets=False, **kwargs)
