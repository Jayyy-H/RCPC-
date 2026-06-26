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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
"""

import inspect
from contextlib import contextmanager
from typing import Any, List, Union

import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer
from vllm import LLM, RequestOutput, SamplingParams

try:
    from PIL import Image
except ImportError:
    Image = None

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.config import RolloutConfig


def _repeat_interleave(features: Union[torch.Tensor, List[Any]], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(features, torch.Tensor):
        return features.repeat_interleave(repeats, dim=0)
    else:
        return [feature for feature in features for _ in range(repeats)]


def _sanitize_vllm_image(image: Any) -> Any:
    if Image is None or not isinstance(image, Image.Image):
        return image
    try:
        image.getexif()
        return image
    except Exception:
        clean = image.copy()
        clean.info = {
            key: value
            for key, value in clean.info.items()
            if key.lower() not in {"exif", "raw profile type exif", "xml:com.adobe.xmp"}
        }
        if hasattr(clean, "_exif"):
            clean._exif = None
        return clean


def _sanitize_vllm_images(images: Any) -> Any:
    if isinstance(images, list):
        return [_sanitize_vllm_images(image) for image in images]
    return _sanitize_vllm_image(images)


def _make_vllm_image_uuids(prefix: str, call_id: int, request_index: int, images: Any) -> Any:
    if isinstance(images, list):
        return [f"{prefix}-{call_id}-{request_index}-{image_index}" for image_index in range(len(images))]
    return f"{prefix}-{call_id}-{request_index}-0"


def _force_qwen25_vl_vision_sdpa() -> None:
    """Keep LLM paged attention on FlashAttention while avoiding ViT FA head-dim limits."""
    try:
        from vllm.model_executor.models import qwen2_5_vl
    except Exception:
        return

    backend_enum = getattr(qwen2_5_vl, "_Backend", None)
    if backend_enum is None or not hasattr(backend_enum, "TORCH_SDPA"):
        return

    qwen2_5_vl.get_vit_attn_backend = lambda head_size, dtype: backend_enum.TORCH_SDPA
    qwen2_5_vl.check_upstream_fa_availability = lambda dtype: False


class vLLMRolloutValley(BaseRollout):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer: PreTrainedTokenizer):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.forced_response_prefix = str(config.forced_response_prefix or "")
        self.forced_response_prefix_ids = (
            tokenizer.encode(self.forced_response_prefix, add_special_tokens=False)
            if self.forced_response_prefix
            else []
        )
        self._mm_uuid_prefix = "verl-vllm-valley-rollout"
        self._mm_uuid_call_id = 0
        if len(self.forced_response_prefix_ids) >= config.response_length:
            raise ValueError("forced_response_prefix must be shorter than rollout response_length.")
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if not config.enforce_eager and config.free_cache_engine:
            raise ValueError("CUDA graph should be disabled when `free_cache_engine` is True.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        _force_qwen25_vl_vision_sdpa()

        vllm_init_kwargs = {}
        image_limit = config.limit_images if config.limit_images > 0 else config.max_image_num
        if image_limit > 0:
            vllm_init_kwargs["limit_mm_per_prompt"] = {"image": image_limit}
        try:
            llm_signature = inspect.signature(LLM)
        except (TypeError, ValueError):
            llm_signature = None
        if (
            config.mm_processor_cache_gb >= 0
            and llm_signature is not None
            and "mm_processor_cache_gb" in llm_signature.parameters
        ):
            vllm_init_kwargs["mm_processor_cache_gb"] = config.mm_processor_cache_gb
        if (
            config.disable_mm_preprocessor_cache
            and llm_signature is not None
            and "disable_mm_preprocessor_cache" in llm_signature.parameters
        ):
            vllm_init_kwargs["disable_mm_preprocessor_cache"] = True
        mm_processor_kwargs = {}
        if config.min_pixels > 0:
            mm_processor_kwargs["min_pixels"] = config.min_pixels
        if config.max_pixels > 0:
            mm_processor_kwargs["max_pixels"] = config.max_pixels
        supports_mm_processor_kwargs = (
            llm_signature is not None and "mm_processor_kwargs" in llm_signature.parameters
        )
        if mm_processor_kwargs and supports_mm_processor_kwargs:
            vllm_init_kwargs["mm_processor_kwargs"] = mm_processor_kwargs

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=config.dtype,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.enforce_eager,
            max_model_len=config.prompt_length + config.response_length,
            max_num_batched_tokens=config.max_num_batched_tokens,
            enable_sleep_mode=True,
            distributed_executor_backend="external_launcher",
            disable_custom_all_reduce=True,
            disable_log_stats=config.disable_log_stats,
            enable_chunked_prefill=config.enable_chunked_prefill,
            seed=config.seed,
            trust_remote_code=True,
            **vllm_init_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        max_tokens = config.response_length - len(self.forced_response_prefix_ids)
        sampling_kwargs = {"max_tokens": max_tokens, "detokenize": False}
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)
        sampling_kwargs["max_tokens"] = max_tokens

        print(f"Sampling params: {sampling_kwargs}.")
        if self.forced_response_prefix:
            print(f"Forced response prefix: {self.forced_response_prefix!r}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        target_ids = prompts.non_tensor_batch.pop("target_ids")  # (bs, prompt_length)
        batch_size = input_ids.size(0)

        do_sample = prompts.meta_info.get("do_sample", True)
        if not do_sample:
            kwargs = {
                "n": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
            }

        non_tensor_batch = prompts.non_tensor_batch
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "navit_processed_images" in non_tensor_batch:
            vllm_inputs = []
            mm_uuid_call_id = self._mm_uuid_call_id
            self._mm_uuid_call_id += 1
            for request_index, (raw_prompt_ids, images) in enumerate(
                zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("navit_processed_images"))
            ):
                raw_prompt_ids = list(raw_prompt_ids) + self.forced_response_prefix_ids
                images = _sanitize_vllm_images(images)
                image_uuids = _make_vllm_image_uuids(
                    self._mm_uuid_prefix, mm_uuid_call_id, request_index, images
                )
                vllm_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": {"image": images},
                        "multi_modal_uuids": {"image": image_uuids},
                    }
                )
        else:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids) + self.forced_response_prefix_ids}
                for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            completions: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params
            )

        response_ids = []
        is_onpolicy = []
        for i, comp in enumerate(completions):
            # on-policy
            for out in comp.outputs:
                response_ids.append(self.forced_response_prefix_ids + list(out.token_ids))
                is_onpolicy.append(True)
            # off-policy
            response_ids.append(target_ids[i])
            is_onpolicy.append(False)


        response_ids = pad_2d_list_to_length(
            response_ids, self.pad_token_id, max_length=self.config.response_length
        ).to(input_ids.device)
        
        if self.config.n > 1 and do_sample:
            batch_size = batch_size * (self.config.n + 1)
            input_ids = _repeat_interleave(input_ids, self.config.n + 1)
            attention_mask = _repeat_interleave(attention_mask, self.config.n + 1)
            position_ids = _repeat_interleave(position_ids, self.config.n + 1)
            if "pixel_values" in non_tensor_batch.keys():
                non_tensor_batch["pixel_values"] = _repeat_interleave(non_tensor_batch["pixel_values"], self.config.n + 1)
                non_tensor_batch["image_grid_thw"] = _repeat_interleave(
                    non_tensor_batch["image_grid_thw"], self.config.n + 1
                )
            if "images" in non_tensor_batch.keys():
                non_tensor_batch["images"] = _repeat_interleave(non_tensor_batch["images"], self.config.n + 1)
                non_tensor_batch["image_sizes"] = _repeat_interleave(non_tensor_batch["image_sizes"], self.config.n + 1)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(
            response_ids=response_ids, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        non_tensor_batch["is_onpolicy"] = is_onpolicy
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    @torch.no_grad()
    def generate_sequences_val(self, prompts: DataProto, **kwargs) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        do_sample = prompts.meta_info.get("do_sample", True)
        if not do_sample:
            kwargs = {
                "n": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
            }

        non_tensor_batch = prompts.non_tensor_batch
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "navit_processed_images" in non_tensor_batch:
            vllm_inputs = []
            mm_uuid_call_id = self._mm_uuid_call_id
            self._mm_uuid_call_id += 1
            for request_index, (raw_prompt_ids, images) in enumerate(
                zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("navit_processed_images"))
            ):
                raw_prompt_ids = list(raw_prompt_ids) + self.forced_response_prefix_ids
                images = _sanitize_vllm_images(images)
                image_uuids = _make_vllm_image_uuids(
                    self._mm_uuid_prefix, mm_uuid_call_id, request_index, images
                )
                vllm_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": {"image": images},
                        "multi_modal_uuids": {"image": image_uuids},
                    }
                )
        else:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids) + self.forced_response_prefix_ids}
                for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            completions: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params
            )

        response_ids = []
        for completion in completions:
            for output in completion.outputs:
                response_ids.append(self.forced_response_prefix_ids + list(output.token_ids))

        response_ids = pad_2d_list_to_length(
            response_ids, self.pad_token_id, max_length=self.config.response_length
        ).to(input_ids.device)

        if self.config.n > 1 and do_sample:
            batch_size = batch_size * self.config.n
            input_ids = _repeat_interleave(input_ids, self.config.n)
            attention_mask = _repeat_interleave(attention_mask, self.config.n)
            position_ids = _repeat_interleave(position_ids, self.config.n)
            if "pixel_values" in non_tensor_batch.keys():
                non_tensor_batch["pixel_values"] = _repeat_interleave(non_tensor_batch["pixel_values"], self.config.n)
                non_tensor_batch["image_grid_thw"] = _repeat_interleave(
                    non_tensor_batch["image_grid_thw"], self.config.n
                )
            if "images" in non_tensor_batch.keys():
                non_tensor_batch["images"] = _repeat_interleave(non_tensor_batch["images"], self.config.n)
                non_tensor_batch["image_sizes"] = _repeat_interleave(non_tensor_batch["image_sizes"], self.config.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(
            response_ids=response_ids, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
