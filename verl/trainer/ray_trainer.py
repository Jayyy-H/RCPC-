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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""
import re
import os
import uuid
import glob
import math
import shutil
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
from codetiming import Timer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin, AutoConfig

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer import core_algos
from verl.trainer.config import PPOConfig
from verl.utils.rl_dataset import RLHFDataset, collate_fn as collate_fn_default
from verl.utils.rl_dataset_valley import RLHFDatasetValley, collate_fn as collate_fn_valley
from verl.utils.sftrl_dataset import SFTRLDataset, collate_fn as collate_fn_sftrl
from verl.utils.sftrl_dataset_valley import SFTRLDatasetValley, collate_fn as collate_fn_sftrl_valley
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import Tracking
from verl.workers.fsdp_workers import FSDPWorker
from verl.utils.checkpoint.fsdp_checkpoint_manager import merge_shards


WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if "ref_log_prob" in data.batch.keys():
        kld = core_algos.kl_penalty(
            data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
        )  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"critic/kl": current_kl, "critic/kl_coeff": beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == "gae":
        values = data.batch["values"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch["token_level_rewards"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards, values=values, eos_mask=response_mask, gamma=gamma, lam=lam
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == "grpo":
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        is_onpolicy = data.non_tensor_batch['is_onpolicy']
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index,
            is_onpolicy=is_onpolicy,
            criterion_advantages=(
                data.batch["criterion_advantages"] if "criterion_advantages" in data.batch else None
            ),
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if "criterion_advantages" in data.batch:
            data.batch.pop("criterion_advantages")
    elif adv_estimator == "reinforce_plus_plus":
        token_level_rewards = data.batch["token_level_rewards"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == "remax":
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch["reward_baselines"]

        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=token_level_rewards, reward_baselines=reward_baselines, eos_mask=response_mask
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: Dict[str, Any]):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)

    return metrics


def _compute_response_info(batch: DataProto):
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True):
    # TODO: add response length
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # Keep one canonical reward family. Raw/final and policy split metrics were
        # redundant for the current on-policy, zero-KL ROPD setup.
        "reward/final/mean": torch.mean(sequence_reward).detach().item(),
        "reward/final/max": torch.max(sequence_reward).detach().item(),
        "reward/final/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield

    timing_raw[name] = timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.worker.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.best_val_reward_score = -1.0
        self.val_reward_score = 0.0

        # define KL control
        if self.use_reference_policy:
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.0)

        if self.config.algorithm.adv_estimator == "gae":
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == "grpo":
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == "reinforce_plus_plus":
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == "remax":
            self.use_critic = False
        else:
            raise NotImplementedError
        
        # get dataset_cls and collate_fn accroding to model_type
        model_config = AutoConfig.from_pretrained(self.config.worker.actor.model.model_path)
        if getattr(model_config, "model_type", None) in ["valley"]:
            if self.config.algorithm.rl_paradigm == 'sft+rl':
                self.dataset_cls = SFTRLDatasetValley
                self.collate_fn = collate_fn_sftrl_valley
            else:
                self.dataset_cls = RLHFDatasetValley
                self.collate_fn = collate_fn_valley
        else:
            if self.config.algorithm.rl_paradigm == 'sft+rl':
                self.dataset_cls = SFTRLDataset
                self.collate_fn = collate_fn_sftrl
            else:
                self.dataset_cls = RLHFDataset
                self.collate_fn = collate_fn_default
        self._create_dataloader()

    def _create_dataloader(self):
        self.train_dataset = self.dataset_cls(
            data_path=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            target_key=self.config.data.target_key,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            min_pixels=self.config.data.min_pixels,
            max_pixels=self.config.data.max_pixels,
        )
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.rollout_batch_size,
            num_workers=0,
            drop_last=True,
            collate_fn=self.collate_fn,
            sampler=sampler,
        )

        self.val_dataset = self.dataset_cls(
            data_path=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            target_key=self.config.data.target_key,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            min_pixels=self.config.data.min_pixels,
            max_pixels=self.config.data.max_pixels,
        )

        if self.config.data.val_batch_size is None:
            val_batch_size = len(self.val_dataset) 
        else:
            val_batch_size = self.config.data.val_batch_size

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn,
        )



        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        if self.config.trainer.max_steps is not None:
            training_steps = self.config.trainer.max_steps
        else:
            training_steps = len(self.train_dataloader) * self.config.trainer.total_episodes

        self.training_steps = training_steps
        self.config.worker.actor.optim.training_steps = training_steps
        self.config.worker.critic.optim.training_steps = training_steps
        print(f"Total training steps: {self.training_steps}")

    def _maybe_log_val_generations_to_wandb(self, inputs, outputs, scores):
        """Log a table of validation samples to wandb"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and "wandb" not in self.config.trainer.logger:
            print("WARNING: `val_generations_to_log_to_wandb` is set, but no wandb logger is found.")
            return

        import wandb

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Create column names for all samples
        columns = ["step"] + sum(
            [[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], []
        )

        # Log only the current validation step. Copying every previous row into
        # every new table makes local W&B media usage grow quadratically.
        new_table = wandb.Table(columns=columns)
        row_data = [self.global_steps]
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Observability failures must not terminate a long-running training job.
        try:
            wandb.log({"val/generations": new_table}, step=self.global_steps)
        except Exception as exc:
            print(
                f"WARNING: failed to log val/generations to wandb at step={self.global_steps}; "
                f"training will continue. {type(exc).__name__}: {exc}"
            )

    @staticmethod
    def _env_flag(name: str, default: str = "1") -> bool:
        value = os.getenv(name, default).strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _tokenizer_vocab_size(self) -> Optional[int]:
        sizes = []
        value = getattr(self.tokenizer, "vocab_size", None)
        if isinstance(value, int) and value > 0:
            sizes.append(value)
        try:
            value = len(self.tokenizer)
            if isinstance(value, int) and value > 0:
                sizes.append(value)
        except TypeError:
            pass
        return max(sizes) if sizes else None

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError("[actor batch validation] " + message)

    @staticmethod
    def _require_finite(name: str, tensor: torch.Tensor) -> None:
        if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            return
        if tensor.numel() > 0 and not torch.isfinite(tensor).all().item():
            bad_count = int((~torch.isfinite(tensor)).sum().item())
            raise ValueError(
                f"[actor batch validation] {name} contains {bad_count} non-finite values; "
                f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
            )

    @staticmethod
    def _require_token_range(name: str, tensor: torch.Tensor, vocab_size: Optional[int]) -> None:
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return
        min_id = int(tensor.min().item())
        max_id = int(tensor.max().item())
        if min_id < 0:
            raise ValueError(
                f"[actor batch validation] {name} has negative token id {min_id}; "
                f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
            )
        if vocab_size is not None and max_id >= vocab_size:
            raise ValueError(
                f"[actor batch validation] {name} has token id {max_id} >= vocab_size {vocab_size}; "
                f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
            )

    def _validate_actor_update_batch(self, batch: DataProto, stage: str) -> None:
        if not self._env_flag("ROPD_VALIDATE_ACTOR_BATCH", "1"):
            return
        batch.check_consistency()
        required = [
            "input_ids",
            "responses",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "token_level_scores",
            "token_level_rewards",
            "sft_coef_tensor",
        ]
        missing = [key for key in required if key not in batch.batch]
        self._require(not missing, f"{stage}: missing tensor keys {missing}")

        input_ids = batch.batch["input_ids"]
        responses = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]
        position_ids = batch.batch["position_ids"]
        batch_size, sequence_length = input_ids.shape
        response_length = responses.shape[-1]
        vocab_size = self._tokenizer_vocab_size()

        self._require(input_ids.dim() == 2, f"{stage}: input_ids must be 2D, got {tuple(input_ids.shape)}")
        self._require(responses.dim() == 2, f"{stage}: responses must be 2D, got {tuple(responses.shape)}")
        self._require(responses.shape[0] == batch_size, f"{stage}: responses batch mismatch")
        self._require(attention_mask.shape == input_ids.shape, f"{stage}: attention_mask shape mismatch")
        self._require(sequence_length > response_length, f"{stage}: sequence_length <= response_length")
        if position_ids.dim() == 2:
            self._require(position_ids.shape == input_ids.shape, f"{stage}: position_ids shape mismatch")
        elif position_ids.dim() == 3:
            self._require(
                position_ids.shape[0] == batch_size and position_ids.shape[-1] == sequence_length,
                f"{stage}: mrope position_ids shape mismatch, got {tuple(position_ids.shape)}",
            )
        else:
            raise ValueError(f"[actor batch validation] {stage}: unsupported position_ids dim {position_ids.dim()}")

        if attention_mask.dtype != torch.bool:
            is_binary = ((attention_mask == 0) | (attention_mask == 1)).all().item()
            self._require(bool(is_binary), f"{stage}: attention_mask must be binary")
        response_mask = attention_mask[:, -response_length:]
        self._require((response_mask.sum(dim=-1) > 0).all().item(), f"{stage}: empty response mask row")

        self._require_token_range(f"{stage}:input_ids", input_ids, vocab_size)
        self._require_token_range(f"{stage}:responses", responses, vocab_size)
        for key in [
            "old_log_probs",
            "advantages",
            "token_level_scores",
            "token_level_rewards",
            "sft_coef_tensor",
            "ref_log_prob",
        ]:
            if key in batch.batch:
                tensor = batch.batch[key]
                self._require(tensor.shape == responses.shape, f"{stage}: {key} shape {tuple(tensor.shape)} != responses {tuple(responses.shape)}")
                self._require_finite(f"{stage}:{key}", tensor)

        if "is_onpolicy" in batch.non_tensor_batch:
            is_onpolicy = batch.non_tensor_batch["is_onpolicy"]
            self._require(len(is_onpolicy) == batch_size, f"{stage}: is_onpolicy length mismatch")

        if "global_token_num" in batch.meta_info:
            self._require(
                len(batch.meta_info["global_token_num"]) == batch_size,
                f"{stage}: global_token_num length mismatch",
            )

    def _validate_actor_metrics(self, metrics: Dict[str, Any], stage: str) -> None:
        if not self._env_flag("ROPD_VALIDATE_ACTOR_BATCH", "1"):
            return
        for key, value in metrics.items():
            values = value if isinstance(value, list) else [value]
            for item in values:
                if isinstance(item, (int, float)) and not math.isfinite(item):
                    raise ValueError(f"[actor metric validation] {stage}: metric {key} is non-finite: {item}")

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        max_val_batches = self.config.trainer.max_val_batches
        for i, test_data in enumerate(self.val_dataloader):
            if max_val_batches is not None and i >= max_val_batches:
                break
            print(f">>> Validation Step: {i} / {len(self.val_dataloader)}")
            test_batch = DataProto.from_single_dict(test_data)
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_ids_copy = deepcopy(input_ids)
            input_ids_copy[input_ids_copy < 0] = 0 # Convert the vocab_id of the image placeholder (-200) to 0.
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids_copy]
            sample_inputs.extend(input_texts)

            # pop batch_keys and non_tensor_batch_keys
            batch_keys=["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys=[]
            if "raw_prompt_ids" in test_batch.non_tensor_batch.keys():
                non_tensor_batch_keys.append("raw_prompt_ids")
            if "pixel_values" in test_batch.non_tensor_batch.keys():
                non_tensor_batch_keys += ["pixel_values", "image_grid_thw", "images"]
            if "image_sizes" in test_batch.non_tensor_batch.keys():
                non_tensor_batch_keys.append("image_sizes")
            if "navit_processed_images" in test_batch.non_tensor_batch.keys():
                non_tensor_batch_keys.append("navit_processed_images")

            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys,
                non_tensor_batch_keys=non_tensor_batch_keys
            )

            test_gen_batch.meta_info = {"do_sample": False}

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences_val(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            )

        self._maybe_log_val_generations_to_wandb(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards)
        metric_dict["val/test_score/mean"] = reward_tensor.mean().item()
        self.val_reward_score = metric_dict["val/test_score/mean"]
        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg: FSDPWorker = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg: FSDPWorker = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg: FSDPWorker = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg: FSDPWorker = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        
        if self.val_reward_score >= self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_steps
        # path: {save_checkpoint_path}/global_step_{global_steps}/actor
        local_global_step_folder = os.path.join(
            self.config.trainer.save_checkpoint_path, f"global_step_{self.global_steps}"
        )
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            self.global_steps,
            remove_previous_ckpt=self.config.trainer.remove_previous_ckpt,
        )
        if self.config.trainer.auto_merge_shards:
            merge_shards(local_dir=os.path.join(local_global_step_folder, "actor"))
        self._limit_checkpoints_and_shards("actor")


        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            self.critic_wg.save_checkpoint(
                critic_local_path,
                self.global_steps,
                remove_previous_ckpt=self.config.trainer.remove_previous_ckpt,
            )
            if self.config.trainer.auto_merge_shards:
                merge_shards(local_dir=os.path.join(local_global_step_folder, "critic"))
            self._limit_checkpoints_and_shards("critic")

        dataloader_path = os.path.join(local_global_step_folder, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)
        print(f"Save dataloader to {dataloader_path}")

        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.save_checkpoint_path, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        pattern = re.compile(r'global_step_(\d+)')
        # check existed ckpts in save_checkpoint_path
        if self.config.trainer.load_checkpoint_path is None:
            # extract and sort ckpts
            if os.path.exists(self.config.trainer.save_checkpoint_path):
                subdirs = os.listdir(self.config.trainer.save_checkpoint_path) 
            else:
                subdirs = []
            subdirs = [d for d in subdirs if pattern.search(d)]
            extract_step = lambda name: int(pattern.search(name).group(1))
            subdirs.sort(key=extract_step)
            
            # if there is no avalible ckpts, return
            if len(subdirs) == 0:
                return
            
            # on the other hand, use the latest ckpt
            self.config.trainer.load_checkpoint_path = os.path.join(
                self.config.trainer.save_checkpoint_path, 
                subdirs[-1]
            )

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}")
        match = pattern.search(self.config.trainer.load_checkpoint_path)
        self.resume_steps = int(match.group(1)) if match else 0
        
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
            self.load_dataloader_state = True
            print(f"Loaded dataloader state from {dataloader_path}")
        else:
            self.load_dataloader_state = False
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _limit_checkpoints_and_shards(self, role="actor"):
        """
        Limit the number of checkpoints and shards.
        """
        # existed ckpts in save_checkpoint_path
        pattern = re.compile(r'global_step_(\d+)')
        if os.path.exists(self.config.trainer.save_checkpoint_path):
                subdirs = os.listdir(self.config.trainer.save_checkpoint_path) 
        else:
            subdirs = []
        subdirs = [d for d in subdirs if pattern.search(d)]
        extract_step = lambda name: int(pattern.search(name).group(1))
        subdirs.sort(key=extract_step)
        
        # limit the number of checkpoints
        if self.config.trainer.save_ckpt_limit and len(subdirs) > self.config.trainer.save_ckpt_limit:
            # identify the best checkpoint directory
            best_ckp_dir = None
            if hasattr(self, 'best_global_step'):
                best_ckp_dir = f"global_step_{self.best_global_step}"
            
            # delete the oldest checkpoints, but skip the best checkpoint
            to_delete = len(subdirs) - self.config.trainer.save_ckpt_limit
            deleted = 0
            i = 0
            while deleted < to_delete and i < len(subdirs):
                if best_ckp_dir and subdirs[i] == best_ckp_dir:
                    # Skip the best checkpoint
                    i += 1
                    continue
                
                shutil.rmtree(os.path.join(self.config.trainer.save_checkpoint_path, subdirs[i]), ignore_errors=True)
                subdirs.pop(i)  # No need to increment i since we're removing the element
                deleted += 1
        
        # limit the number of shards
        if self.config.trainer.save_shard_limit and len(subdirs) > self.config.trainer.save_shard_limit:
            # identify the best checkpoint directory
            best_ckp_dir = None
            if hasattr(self, 'best_global_step'):
                best_ckp_dir = f"global_step_{self.best_global_step}"
            
            # delete the oldest shards, but skip the best checkpoint
            to_delete = len(subdirs) - self.config.trainer.save_shard_limit
            deleted = 0
            i = 0
            while deleted < to_delete and i < len(subdirs):
                if best_ckp_dir and subdirs[i] == best_ckp_dir:
                    # Skip the best checkpoint
                    i += 1
                    continue
                    
                actor_path = os.path.join(self.config.trainer.save_checkpoint_path, subdirs[i], role)
                shards = glob.glob(actor_path + "/*.pt")
                for shard in shards:
                    os.remove(shard)
                deleted += 1
                i += 1

    def cosine_decay_sft_coef(self,
    ) -> float:
        if self.config.worker.actor.sft_coef_decay_step is None or self.config.worker.actor.sft_coef_decay_step <= 0:
            return self.config.worker.actor.sft_coef

        step = min(self.global_steps, self.config.worker.actor.sft_coef_decay_step)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * step / self.config.worker.actor.sft_coef_decay_step))
        ratio = self.config.worker.actor.sft_coef_decay_rate + (1 - self.config.worker.actor.sft_coef_decay_rate) * cosine_factor
        return self.config.worker.actor.sft_coef * ratio

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=self.config.to_dict(),
        )
        self.global_steps = 0
        self.sft_coef = self.config.worker.actor.sft_coef

        # load checkpoint before doing anything
        self.resume_steps = None
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps if self.resume_steps is None else self.resume_steps)
            if self.config.trainer.val_only:
                return

        if self.resume_steps is not None:
            if self.load_dataloader_state:
                self.global_steps = self.resume_steps
            else:
                pass # self.global_steps == 0, need to dequeue data from the beginning

        for _ in range(self.config.trainer.total_episodes):
            for batch_dict in self.train_dataloader:
                self.global_steps += 1
                if self.global_steps >= self.training_steps:
                    break

                if self.resume_steps is not None and self.load_dataloader_state == False:
                    # Conditions explained:
                    # 1. self.resume_steps is not None: There are steps to resume from
                    # 2. self.load_dataloader_state == False: Not loading dataloader state (need to iterate from beginning)
                    # 3. self.global_steps <= self.resume_steps: Current step hasn't reached the resume point
                    # Scenario: Resume from step 1000, but dataloader starts from beginning, need to skip first 1000 batches
                    if self.global_steps <= self.resume_steps:
                        continue

                print(f">>> {self.global_steps} / {self.training_steps}")

                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys=["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys=[]
                if "raw_prompt_ids" in batch.non_tensor_batch.keys():
                    non_tensor_batch_keys.append("raw_prompt_ids")
                if "pixel_values" in batch.non_tensor_batch.keys():
                    non_tensor_batch_keys += ["pixel_values", "image_grid_thw", "images"]
                if "image_sizes" in batch.non_tensor_batch.keys():
                    non_tensor_batch_keys.append("image_sizes")
                if "navit_processed_images" in batch.non_tensor_batch.keys():
                    non_tensor_batch_keys.append("navit_processed_images")
                if "target_ids" in batch.non_tensor_batch.keys():
                    non_tensor_batch_keys.append("target_ids")

                gen_batch = batch.pop(
                    batch_keys=batch_keys,
                    non_tensor_batch_keys=non_tensor_batch_keys
                )

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):  # wg: worker group
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == "remax":
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.worker.rollout.n + 1, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    # self._balance_batch(batch, metrics=metrics) # TODO: re-enable balance batch

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_reward_model:
                            raise NotImplementedError

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch["token_level_scores"] = reward_tensor
                        ropd_metrics = batch.meta_info.pop("ropd_metrics", None)
                        if isinstance(ropd_metrics, dict):
                            metrics.update(ropd_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.worker.actor.use_kl_loss:  # not grpo
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.worker.rollout.n,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)

                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)


                    self.sft_coef = self.cosine_decay_sft_coef()
                    sft_coef_tensor = torch.full_like(reward_tensor, fill_value=self.sft_coef)

                    batch.batch["sft_coef_tensor"] = sft_coef_tensor
                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            self._validate_actor_update_batch(
                                batch, stage=f"step={self.global_steps}/before_update_actor"
                            )
                            actor_output = self.actor_rollout_wg.update_actor(batch)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        self._validate_actor_metrics(
                            actor_output_metrics, stage=f"step={self.global_steps}/after_update_actor"
                        )
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and self.global_steps % self.config.trainer.test_freq == 0
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

        # perform validation after training
        if self.val_reward_fn is not None:
            val_metrics = self._validate()
            pprint(f"Final validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)

        self._save_checkpoint()
