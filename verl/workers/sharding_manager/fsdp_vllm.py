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
import re
from typing import Dict, Iterable, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel
from vllm import LLM
from vllm.distributed import parallel_state as vllm_ps

from verl import DataProto
from verl.utils.performance import log_gpu_memory_usage
from verl.workers.rollout.vllm_rollout import load_dtensor_weights

from .base import BaseShardingManager


class FSDPVLLMShardingManager(BaseShardingManager):
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        device_mesh: DeviceMesh = None,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.device_mesh = device_mesh
        FSDP.set_state_dict_type(
            self.module,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(),
        )

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    @staticmethod
    def _env_flag(name: str, default: str = "0") -> bool:
        value = os.getenv(name, default).strip().lower()
        return value in {"1", "true", "yes", "on"}

    @classmethod
    def _maybe_sync_cuda(cls):
        if cls._env_flag("ROPD_SYNC_VLLM_PHASES") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _get_vllm_model(self):
        model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
        if hasattr(model_runner, "get_model"):
            return model_runner.get_model()
        model = model_runner.model
        if hasattr(model, "unwrap"):
            return model.unwrap()
        return model

    def _rename_weight_keys(self, actor_weights: Dict[str, Union[torch.Tensor, DTensor]], model: PreTrainedModel):
        # Keep compatibility with newer transformers/vLLM Qwen2.5-VL checkpoint key conversions.
        if not hasattr(model, "_checkpoint_conversion_mapping"):
            return actor_weights

        reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
        original_weights = {}
        for key, value in actor_weights.items():
            for pattern, replacement in reverse_key_mapping.items():
                replacement = replacement.lstrip("^")
                replacement = re.sub(r"\(.*\)", "", replacement)
                key, n_replace = re.subn(pattern, replacement, key)
                if n_replace > 0:
                    break
            original_weights[key] = value
        return original_weights

    def _make_weight_iterator(
        self, actor_weights: Dict[str, Union[torch.Tensor, DTensor]]
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        for name, tensor in actor_weights.items():
            if hasattr(tensor, "full_tensor"):
                tensor = tensor.full_tensor()
            yield name, tensor

    def _sync_weights_to_vllm(self, actor_weights: Dict[str, Union[torch.Tensor, DTensor]]):
        vllm_model = self._get_vllm_model()
        if hasattr(vllm_model, "load_weights"):
            actor_model = self.module._fsdp_wrapped_module
            actor_weights = self._rename_weight_keys(actor_weights, actor_model)
            vllm_model.load_weights(self._make_weight_iterator(actor_weights))
        else:
            load_dtensor_weights(actor_weights, vllm_model)

    def __enter__(self):
        self._maybe_sync_cuda()
        log_gpu_memory_usage("Before state_dict() in sharding manager")
        actor_weights = self.module.state_dict()
        self._maybe_sync_cuda()
        log_gpu_memory_usage("After state_dict() in sharding manager")

        self.inference_engine.wake_up()
        self._maybe_sync_cuda()
        self._sync_weights_to_vllm(actor_weights)
        self._maybe_sync_cuda()
        log_gpu_memory_usage("After sync model weights in sharding manager")

        del actor_weights
        self._maybe_sync_cuda()
        torch.cuda.empty_cache()
        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager")
        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage("Before vllm offload in sharding manager")
        self._maybe_sync_cuda()
        self.inference_engine.sleep(level=1)
        self._maybe_sync_cuda()
        log_gpu_memory_usage("After vllm offload in sharding manager")

        self.module.train()
        self._maybe_sync_cuda()
        torch.cuda.empty_cache()  # add empty cache after each compute

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        tp_group = vllm_ps.get_tensor_model_parallel_group().device_group
        data = data.to("cuda")
        data.all_gather(tp_group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        dp_rank = dist.get_rank()
        tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            data = data.chunk(chunks=tp_size)[dp_rank % tp_size]

        return data
