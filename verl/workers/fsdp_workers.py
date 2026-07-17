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
The main entry point to run the PPO algorithm
"""

from typing import List, Literal, Sequence

import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from codetiming import Timer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import get_tokenizer, get_processor
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_fn,
    load_fsdp_model,
    load_fsdp_optimizer,
    offload_fsdp_model,
    offload_fsdp_optimizer,
)
from verl.utils.model_utils import print_model_size
from verl.utils.performance import log_gpu_memory_usage
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_constant_schedule_with_warmup
from verl.workers.actor import DataParallelPPOActor
from verl.workers.config import FSDPConfig, ModelConfig, OptimConfig, WorkerConfig
from verl.workers.critic import DataParallelPPOCritic
from verl.workers.rollout.vllm_rollout import vLLMRollout
from verl.workers.sharding_manager import FSDPVLLMShardingManager
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager


def _dispatch_dp_compute_proto_chunks(worker_group, prompt_chunks, **kwargs):
    """Shard every global DataProto chunk across the same DP worker group."""
    if not isinstance(prompt_chunks, Sequence):
        raise TypeError("prompt_chunks must be a sequence of DataProto objects")

    chunks_per_worker: List[List[DataProto]] = [
        [] for _ in range(worker_group.world_size)
    ]
    for chunk_index, prompt_chunk in enumerate(prompt_chunks):
        if not isinstance(prompt_chunk, DataProto):
            raise TypeError(
                "prompt_chunks[{}] must be DataProto, got {}".format(
                    chunk_index, type(prompt_chunk).__name__
                )
            )
        worker_shards = prompt_chunk.chunk(chunks=worker_group.world_size)
        for worker_index, shard in enumerate(worker_shards):
            chunks_per_worker[worker_index].append(shard)

    dispatched_kwargs = {
        key: [value] * worker_group.world_size for key, value in kwargs.items()
    }
    return [chunks_per_worker], dispatched_kwargs


def _collect_dp_compute_proto_chunks(worker_group, worker_outputs):
    """Restore the global batch for each chunk without merging chunk padding."""
    if len(worker_outputs) != worker_group.world_size:
        raise RuntimeError(
            "expected {} worker outputs, got {}".format(
                worker_group.world_size, len(worker_outputs)
            )
        )
    if not worker_outputs:
        return []

    chunk_count = len(worker_outputs[0])
    for worker_index, outputs in enumerate(worker_outputs):
        if len(outputs) != chunk_count:
            raise RuntimeError(
                "worker {} returned {} chunks, expected {}".format(
                    worker_index, len(outputs), chunk_count
                )
            )
        if any(not isinstance(output, DataProto) for output in outputs):
            raise TypeError("counterfactual chunk workers must return DataProto objects")

    return [
        DataProto.concat(
            [worker_outputs[worker_index][chunk_index] for worker_index in range(worker_group.world_size)]
        )
        for chunk_index in range(chunk_count)
    ]


_DP_COMPUTE_PROTO_CHUNKS = {
    "dispatch_fn": _dispatch_dp_compute_proto_chunks,
    "collect_fn": _collect_dp_compute_proto_chunks,
}


def print_trainable_params(model):
    import re
    from prettytable import PrettyTable
    trainable_params = [k for k,v in model.named_parameters() if v.requires_grad]
    trainable_params_group = {}
    for para in trainable_params:
        layer_num = re.findall(r'layers.(\d+)\.',para)
        block_num = re.findall(r'blocks.(\d+)\.',para)
        if layer_num:
            cur_layer = int(layer_num[0])
            if para.replace('layers.'+layer_num[0],'layers.*') not in trainable_params_group:
                trainable_params_group[para.replace('layers.'+layer_num[0],'layers.*')] = layer_num[0]
            elif cur_layer > int(trainable_params_group[para.replace('layers.'+layer_num[0],'layers.*')]):
                trainable_params_group[para.replace('layers.'+layer_num[0],'layers.*')] = layer_num[0]
        elif block_num:
            cur_layer = int(block_num[0])
            if para.replace('blocks.'+block_num[0],'blocks.*') not in trainable_params_group:
                trainable_params_group[para.replace('blocks.'+block_num[0],'blocks.*')] = block_num[0]
            elif cur_layer > int(trainable_params_group[para.replace('blocks.'+block_num[0],'blocks.*')]):
                trainable_params_group[para.replace('blocks.'+block_num[0],'blocks.*')] = block_num[0]
        else:
            trainable_params_group[para] = '0'
    table = PrettyTable(['Parameter Name','Max Layer Number'])
    for key in trainable_params_group.keys():
        table.add_row([key, str(int(trainable_params_group[key])+1)])
    
    print(table)
    total_num = sum([v.numel() for k,v in model.named_parameters()])
    trainable_num = sum([v.numel() for k,v in model.named_parameters() if v.requires_grad])
    print('Total: {:.2f}M'.format(total_num/1e6))
    print(' Trainable: {:.2f}M'.format(trainable_num/1e6))

class FSDPWorker(Worker):
    def __init__(
        self,
        config: WorkerConfig,
        role: Literal["actor", "critic", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        super().__init__()
        self.config = config
        self._counterfactual_rollout_session_active = False

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # build device mesh for FSDP
        # TODO: support FSDP hybrid shard for larger model
        world_size = dist.get_world_size()
        self.device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_sequence_parallel_size = self.config.actor.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(world_size // self.ulysses_sequence_parallel_size, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )
        else:
            self.ulysses_device_mesh = None

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.role = role
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_critic = self.role == "critic"
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        self._use_param_offload = False
        self._use_optimizer_offload = False
        self.sft_coef = 0
        if self._is_actor:
            self._use_param_offload = self.config.actor.offload.param_offload
            self._use_optimizer_offload = self.config.actor.offload.optimizer_offload
        elif self._is_critic:
            self._use_param_offload = self.config.critic.offload.param_offload
            self._use_optimizer_offload = self.config.critic.offload.optimizer_offload
        elif self._is_ref:
            # NOTE: it seems that manual offload is slowly than FSDP offload
            self._use_param_offload = self.config.ref.offload.param_offload

        # normalize config
        if self._is_actor:
            self.config.actor.global_batch_size *= self.config.rollout.n
            self.config.actor.global_batch_size_per_device = (
                self.config.actor.global_batch_size // self.device_mesh.shape[0] * self.ulysses_sequence_parallel_size
            )
            assert (
                self.config.actor.global_batch_size_per_device
                % self.config.actor.micro_batch_size_per_device_for_update
                == 0
            )
        elif self._is_critic:
            self.config.critic.global_batch_size *= self.config.rollout.n
            self.config.critic.global_batch_size_per_device = (
                self.config.critic.global_batch_size // self.device_mesh.shape[0] * self.ulysses_sequence_parallel_size
            )
            assert (
                self.config.critic.global_batch_size_per_device
                % self.config.critic.micro_batch_size_per_device_for_update
                == 0
            )
    def _set_sft_coef(self, sft_coef: float):
        self.sft_coef = sft_coef

    def _build_model_optimizer(
        self,
        model_config: ModelConfig,
        fsdp_config: FSDPConfig,
        optim_config: OptimConfig,
        padding_free: bool = False,
    ) -> None:
        self.tokenizer = get_tokenizer(model_config.tokenizer_path, trust_remote_code=model_config.trust_remote_code)
        self.processor = get_processor(
            model_config.tokenizer_path,
            trust_remote_code=model_config.trust_remote_code,
            use_fast=True,
        )
        self.model_config = AutoConfig.from_pretrained(
            model_config.model_path,
            trust_remote_code=model_config.trust_remote_code,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_config.override_config,
        )

        try:
            self.generation_config = GenerationConfig.from_pretrained(model_config.model_path)
        except Exception:
            self.generation_config = GenerationConfig.from_model_config(self.model_config)

        self.print_rank0(f"Model config: {self.model_config}")

        if padding_free:
            raise NotImplementedError("Padding free is not implemented yet.")

        if fsdp_config.torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor or self._is_critic else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)

        if self._is_critic:
            auto_class = AutoModelForTokenClassification
        elif type(self.model_config) in AutoModelForVision2Seq._model_mapping.keys():
            auto_class = AutoModelForVision2Seq
        else:
            auto_class = AutoModelForCausalLM

        if self.rank == 0:
            model = auto_class.from_pretrained(
                model_config.model_path,
                config=self.model_config,
                torch_dtype=torch_dtype,
                attn_implementation=model_config.attn_implementation,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            with no_init_weights(), init_empty_weights():
                model = auto_class.from_config(
                    self.model_config,
                    torch_dtype=torch_dtype,
                    attn_implementation=model_config.attn_implementation,
                    trust_remote_code=model_config.trust_remote_code,
                )

        # Tune all parameters
        for param in model.parameters():
            param.requires_grad = True 

        assert isinstance(model, PreTrainedModel)  # lint
        model.tie_weights()  # avoid hanging
        model = model.to(torch_dtype)
        if model_config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        dist.barrier()
        if self.rank == 0:
            print_model_size(model)

        log_gpu_memory_usage("After init from huggingface model")
        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype),
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype),
        )
        auto_wrap_policy = get_fsdp_wrap_policy(model)
        if fsdp_config.enable_full_shard:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        if fsdp_config.param_offload or fsdp_config.optimizer_offload:
            cpu_offload = CPUOffload(offload_params=fsdp_config.param_offload)
        else:
            cpu_offload = None

        if self.rank == 0:
            print(f"FSDP wrap policy: {auto_wrap_policy}.")
            print_trainable_params(model)

        self.fsdp_module = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            param_init_fn=get_init_fn(model, device="cuda") if self.rank != 0 else None,
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
            forward_prefetch=False,
            use_orig_params=True,
            device_mesh=self.device_mesh,
        )
        log_gpu_memory_usage("After Actor FSDP init")

        if self._is_actor or self._is_critic:
            self.optimizer = torch.optim.AdamW(
                self.fsdp_module.parameters(),
                lr=optim_config.lr,
                betas=optim_config.betas,
                weight_decay=optim_config.weight_decay,
            )
            num_warmup_steps = int(optim_config.lr_warmup_steps_ratio * optim_config.training_steps)
            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps
            )
        else:
            self.optimizer, self.lr_scheduler = None, None

        log_gpu_memory_usage("After actor optimizer init")

    def _build_rollout(self) -> None:
        if self.config.rollout.name == "vllm":
            # TODO(sgm): support FSDP hybrid shard for larger model
            tp_size = self.config.rollout.tensor_parallel_size
            dp_size = self.world_size // tp_size
            assert self.world_size % tp_size == 0, (
                f"rollout world_size: {self.world_size} is not divisible by tp_size: {tp_size}"
            )
            rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=["dp", "tp"])
            log_gpu_memory_usage("Before building vllm rollout")
            self.rollout = vLLMRollout(
                model_path=self.config.actor.model.model_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
            )
            log_gpu_memory_usage("After building vllm rollout")

            self.rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.fsdp_module,
                inference_engine=self.rollout.inference_engine,
                device_mesh=rollout_device_mesh,
                enable_prefix_caching=self.config.rollout.enable_prefix_caching,
            )
            log_gpu_memory_usage("After building sharding manager")
        else:
            raise NotImplementedError(f"Unsupported rollout backend: {self.config.rollout.name}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self._is_critic:
            model_config = self.config.critic.model
            fsdp_config = self.config.critic.fsdp
            optim_config = self.config.critic.optim
            padding_free = self.config.critic.padding_free
        else:
            model_config = self.config.actor.model
            fsdp_config = self.config.actor.fsdp
            optim_config = self.config.actor.optim
            padding_free = self.config.actor.padding_free

        if self._is_actor or self._is_critic or self._is_ref:
            self._build_model_optimizer(
                model_config=model_config,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                padding_free=padding_free,
            )
            # get the original unwrapped module
            self.unwrapped_model = self.fsdp_module._fsdp_wrapped_module
            if self._use_optimizer_offload and not self._is_critic:
                offload_fsdp_optimizer(optimizer=self.optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init")

        if self._is_actor:
            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.fsdp_module,
                actor_optimizer=self.optimizer,
            )

        if self._is_critic:
            self.critic = DataParallelPPOCritic(
                config=self.config,
                critic_module=self.fsdp_module,
                critic_optimizer=self.optimizer,
            )

        if self._is_rollout:
            self._build_rollout()

        if self._is_ref:
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.fsdp_module)

        if self._is_actor or self._is_critic:
            self.flops_counter = FlopsCounter(self.model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.fsdp_module,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
            )

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str, global_step: int = 0, remove_previous_ckpt: bool = False):
        assert self._is_actor or self._is_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.save_checkpoint(
            local_path=path,
            global_step=global_step,
            remove_previous_ckpt=remove_previous_ckpt,
        )
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str, del_local_after_load: bool = True):
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.load_checkpoint(path=path, del_local_after_load=del_local_after_load)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

    """ActorRolloutRefWorker"""

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._is_actor

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        log_gpu_memory_usage("Before update policy")
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_policy", logger=None) as timer:
                self.actor._set_sft_coef(self.sft_coef)
                metrics = self.actor.update_policy(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            log_gpu_memory_usage("After update policy")

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto, **kwargs):
        assert self._is_rollout

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            # after parameters sync with rollout, offload actor model to CPU
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)

            log_gpu_memory_usage("After entering rollout sharding manager")

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts, **kwargs)
            log_gpu_memory_usage("After rollout generation")

            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")
        torch.cuda.empty_cache()  # clear kv cache
        log_gpu_memory_usage("After recompute log prob")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences_val(self, prompts: DataProto, **kwargs):
        assert self._is_rollout

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            # after parameters sync with rollout, offload actor model to CPU
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)

            log_gpu_memory_usage("After entering rollout sharding manager")

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences_val(prompts=prompts, **kwargs)
            log_gpu_memory_usage("After rollout generation")

            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")
        torch.cuda.empty_cache()  # clear kv cache
        log_gpu_memory_usage("After recompute log prob")
        return output

    @register(dispatch_mode=_DP_COMPUTE_PROTO_CHUNKS)
    def generate_sequences_val_chunks(self, prompt_chunks: Sequence[DataProto], **kwargs):
        """Generate many independently padded chunks with one FSDP-to-vLLM sync."""
        assert self._is_rollout
        if not prompt_chunks:
            return []

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        for prompts in prompt_chunks:
            prompts.meta_info.update(meta_info)

        outputs = []
        with self.rollout_sharding_manager:
            # Synchronize actor weights once, then keep vLLM awake while every
            # counterfactual chunk is generated.
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)
            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)

            log_gpu_memory_usage("After entering batched rollout sharding manager")
            for prompts in prompt_chunks:
                local_prompts = self.rollout_sharding_manager.preprocess_data(prompts)
                output = self.rollout.generate_sequences_val(
                    prompts=local_prompts, **kwargs
                )
                output = self.rollout_sharding_manager.postprocess_data(output)
                outputs.append(output.to("cpu"))

        torch.cuda.empty_cache()
        log_gpu_memory_usage("After batched counterfactual rollout")
        return outputs

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def begin_counterfactual_rollout_session(self):
        """Wake and synchronize vLLM once for a driver-side producer loop."""
        assert self._is_rollout
        if self._counterfactual_rollout_session_active:
            raise RuntimeError("counterfactual rollout session is already active")
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        entered = False
        try:
            self.rollout_sharding_manager.__enter__()
            entered = True
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)
            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)
            self._counterfactual_rollout_session_active = True
            log_gpu_memory_usage("After beginning counterfactual rollout session")
            return True
        except Exception:
            if entered:
                self.rollout_sharding_manager.__exit__(None, None, None)
            raise

    @register(dispatch_mode=_DP_COMPUTE_PROTO_CHUNKS)
    def generate_sequences_val_chunks_in_session(
        self, prompt_chunks: Sequence[DataProto], **kwargs
    ):
        """Generate chunks while an explicit counterfactual session is active."""
        assert self._is_rollout
        if not self._counterfactual_rollout_session_active:
            raise RuntimeError("counterfactual rollout session is not active")

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        outputs = []
        for prompts in prompt_chunks:
            prompts.meta_info.update(meta_info)
            local_prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences_val(
                prompts=local_prompts, **kwargs
            )
            output = self.rollout_sharding_manager.postprocess_data(output)
            outputs.append(output.to("cpu"))
        return outputs

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def end_counterfactual_rollout_session(self):
        """Close a producer session and restore actor/vLLM memory state."""
        if not self._counterfactual_rollout_session_active:
            return False
        try:
            self.rollout_sharding_manager.__exit__(None, None, None)
        finally:
            self._counterfactual_rollout_session_active = False
            torch.cuda.empty_cache()
        log_gpu_memory_usage("After ending counterfactual rollout session")
        return True

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            log_probs, token_entropies = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(
                tensors={
                    "old_log_probs": log_probs,
                    "token_entropies": token_entropies,
                },
                meta_info={"temperature": self.config.rollout.temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        torch.cuda.empty_cache()
        log_gpu_memory_usage("After compute_log_prob")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            log_probs, _token_entropies = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={"ref_log_prob": log_probs})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        torch.cuda.empty_cache()
        log_gpu_memory_usage("After compute_ref_log_prob")
        return output

    """CriticWorker"""

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        assert self._is_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to("cpu")
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["mfu/critic"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to("cpu")
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        torch.cuda.empty_cache()
        return output
