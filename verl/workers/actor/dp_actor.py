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
Implement Actor
"""

import os
import math
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm
import numpy as np

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer import core_algos
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.workers.actor.base import BasePPOActor
from verl.workers.actor.config import ActorConfig
from contextlib import contextmanager

__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)
        self.in_update=False
        self.sft_coef = 0

    @staticmethod
    def _image_to_cuda(image):
        if torch.is_tensor(image):
            return image.to(device="cuda", dtype=torch.bfloat16, non_blocking=True)
        if hasattr(image, "to"):
            return image.to(device="cuda", dtype=torch.bfloat16)
        return image

    @staticmethod
    def _env_flag(name: str, default: str = "0") -> bool:
        value = os.getenv(name, default).strip().lower()
        return value in {"1", "true", "yes", "on"}

    @classmethod
    def _maybe_sync_cuda(cls, env_name: str):
        if cls._env_flag(env_name) and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _actor_vocab_size(self) -> Optional[int]:
        module = self.actor_module
        inner = getattr(module, "module", None)
        config = getattr(module, "config", None) or getattr(inner, "config", None)
        value = getattr(config, "vocab_size", None)
        return int(value) if isinstance(value, int) and value > 0 else None

    @staticmethod
    def _raise_validation_error(stage: str, message: str) -> None:
        raise ValueError(f"[actor micro-batch validation] {stage}: {message}")

    def _validate_token_tensor(self, name: str, tensor: torch.Tensor, vocab_size: Optional[int], stage: str) -> None:
        if not torch.is_tensor(tensor):
            self._raise_validation_error(stage, f"{name} is not a tensor")
        if tensor.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.long,
        }:
            self._raise_validation_error(stage, f"{name} must be integer dtype, got {tensor.dtype}")
        if tensor.numel() == 0:
            self._raise_validation_error(stage, f"{name} is empty")
        min_id = int(tensor.min().item())
        max_id = int(tensor.max().item())
        if min_id < 0:
            self._raise_validation_error(stage, f"{name} has negative token id {min_id}; shape={tuple(tensor.shape)}")
        if vocab_size is not None and max_id >= vocab_size:
            self._raise_validation_error(
                stage, f"{name} has token id {max_id} >= vocab_size {vocab_size}; shape={tuple(tensor.shape)}"
            )

    def _validate_finite_tensor(self, name: str, tensor: torch.Tensor, stage: str) -> None:
        if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            return
        if tensor.numel() > 0 and not torch.isfinite(tensor).all().item():
            bad_count = int((~torch.isfinite(tensor)).sum().item())
            self._raise_validation_error(
                stage,
                f"{name} contains {bad_count} non-finite values; "
                f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}",
            )

    def _validate_actor_micro_batch(self, micro_batch: Dict[str, Any], stage: str) -> None:
        if not self._env_flag("ROPD_VALIDATE_ACTOR_BATCH", "1"):
            return
        required = ["input_ids", "responses", "attention_mask", "position_ids"]
        missing = [key for key in required if key not in micro_batch]
        if missing:
            self._raise_validation_error(stage, f"missing keys {missing}")

        input_ids = micro_batch["input_ids"]
        responses = micro_batch["responses"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        vocab_size = self._actor_vocab_size()

        if input_ids.dim() != 2 or responses.dim() != 2:
            self._raise_validation_error(stage, f"input_ids/responses must be 2D, got {tuple(input_ids.shape)} / {tuple(responses.shape)}")
        batch_size, sequence_length = input_ids.shape
        response_length = responses.shape[-1]
        if responses.shape[0] != batch_size:
            self._raise_validation_error(stage, "responses batch size mismatch")
        if attention_mask.shape != input_ids.shape:
            self._raise_validation_error(stage, f"attention_mask shape {tuple(attention_mask.shape)} != input_ids {tuple(input_ids.shape)}")
        if sequence_length <= response_length:
            self._raise_validation_error(stage, f"sequence_length {sequence_length} <= response_length {response_length}")
        if position_ids.dim() == 2:
            if position_ids.shape != input_ids.shape:
                self._raise_validation_error(stage, f"position_ids shape {tuple(position_ids.shape)} != input_ids {tuple(input_ids.shape)}")
        elif position_ids.dim() == 3:
            if position_ids.shape[0] != batch_size or position_ids.shape[-1] != sequence_length:
                self._raise_validation_error(stage, f"mrope position_ids shape mismatch: {tuple(position_ids.shape)}")
        else:
            self._raise_validation_error(stage, f"unsupported position_ids dim {position_ids.dim()}")

        if attention_mask.dtype != torch.bool and not ((attention_mask == 0) | (attention_mask == 1)).all().item():
            self._raise_validation_error(stage, "attention_mask must be binary")
        response_mask = attention_mask[:, -response_length:]
        if not (response_mask.sum(dim=-1) > 0).all().item():
            self._raise_validation_error(stage, "response_mask contains an empty row")

        self._validate_token_tensor("input_ids", input_ids, vocab_size, stage)
        self._validate_token_tensor("responses", responses, vocab_size, stage)
        for key in ["old_log_probs", "advantages", "sft_coef_tensor", "ref_log_prob"]:
            if key in micro_batch:
                tensor = micro_batch[key]
                if torch.is_tensor(tensor) and tensor.shape != responses.shape:
                    self._raise_validation_error(stage, f"{key} shape {tuple(tensor.shape)} != responses {tuple(responses.shape)}")
                self._validate_finite_tensor(key, tensor, stage)

        is_onpolicy = micro_batch.get("is_onpolicy")
        if is_onpolicy is not None:
            if torch.is_tensor(is_onpolicy):
                count = is_onpolicy.numel()
            else:
                count = len(is_onpolicy)
            if count != batch_size:
                self._raise_validation_error(stage, f"is_onpolicy length {count} != batch_size {batch_size}")

    def _validate_forward_outputs(
        self,
        entropy: torch.Tensor,
        log_prob: torch.Tensor,
        responses: torch.Tensor,
        stage: str,
    ) -> None:
        if not self._env_flag("ROPD_VALIDATE_ACTOR_BATCH", "1"):
            return
        if entropy.shape != responses.shape:
            self._raise_validation_error(stage, f"entropy shape {tuple(entropy.shape)} != responses {tuple(responses.shape)}")
        if log_prob.shape != responses.shape:
            self._raise_validation_error(stage, f"log_prob shape {tuple(log_prob.shape)} != responses {tuple(responses.shape)}")
        self._validate_finite_tensor("entropy", entropy, stage)
        self._validate_finite_tensor("log_prob", log_prob, stage)

    def _validate_loss(self, loss: torch.Tensor, stage: str) -> None:
        if not self._env_flag("ROPD_VALIDATE_ACTOR_BATCH", "1"):
            return
        if not torch.is_tensor(loss) or loss.numel() != 1:
            self._raise_validation_error(stage, f"loss must be a scalar tensor, got {type(loss)}")
        if not torch.isfinite(loss.detach()).all().item():
            self._raise_validation_error(stage, f"loss is non-finite: {loss.detach().item()}")

    @contextmanager
    def updating(self):
        """Temporarily set self.in_update = True during an operation."""
        old_state = self.in_update
        self.in_update = True
        try:
            yield
        finally:
            self.in_update = old_state

    def _set_sft_coef(self, sft_coef: float):
        self.sft_coef = sft_coef

    def _forward_micro_batch(
        self, micro_batch: Dict[str, torch.Tensor], temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        # is_onpolicy = micro_batch["is_onpolicy"] # (bs, 1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        vision_inputs = {}
        if "pixel_values" in micro_batch:
            vision_inputs["pixel_values"] = torch.cat(micro_batch["pixel_values"], dim=0).to(device="cuda")
            vision_inputs["image_grid_thw"] = torch.cat(micro_batch["image_grid_thw"], dim=0).to(device="cuda")
        
        if "image_sizes" in micro_batch:
            images = micro_batch["images"]
            image_sizes = micro_batch["image_sizes"]
            if type(images[0]) == list:
                images_copy = [[self._image_to_cuda(x) for x in img] for img in images]
            else:
                images_copy = [self._image_to_cuda(img) for img in images]
            vision_inputs["images"] = images_copy
            vision_inputs["image_sizes"] = image_sizes

        if self.config.padding_free:
            # TODO (yaowei): preprocess data for padding_free and ulysses
            raise NotImplementedError
        else:
            if self.actor_module.config.model_type == "valley":                
                attention_mask_copy = attention_mask.clone()
                attention_mask_copy[:, -response_length:] = True
                self.actor_module.right_padding = False
                output = self.actor_module(
                    input_ids=input_ids.to(device="cuda"),
                    attention_mask=attention_mask_copy.to(device="cuda"),
                    # position_ids=position_ids, 
                    **vision_inputs,
                    use_cache=False,
                )
                self.actor_module.right_padding = None
                logits: torch.Tensor = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, responses)  # (bsz, response_length)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                
            else:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **vision_inputs,
                    use_cache=False,
                )
                logits: torch.Tensor = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)

                def safe_logprobs(_logits, _targets):
                    if _logits.numel() == 0:
                        return _logits.new_zeros((0, response_length))
                    return logprobs_from_logits(_logits, _targets)

                def safe_entropy(_logits):
                    if _logits.numel() == 0:
                        return _logits.new_zeros((0, response_length))
                    return verl_F.entropy_from_logits(_logits)       

                # if self.in_update:
                #     is_onpolicy = micro_batch["is_onpolicy"] # (bs, 1)
                #     # 统一的 on/off 划分逻辑
                #     on_mask = is_onpolicy.view(-1).to(dtype=torch.bool, device=logits.device)
                #     off_mask = ~on_mask
                #     bs = logits.size(0)
                #     logits_on = logits[on_mask, :, :] if on_mask.any() else logits.new_zeros((0, response_length, logits.size(-1)))
                #     logits_off = logits[off_mask, :, :] if off_mask.any() else logits.new_zeros((0, response_length, logits.size(-1)))
                #     responses_on = responses[on_mask, :] if on_mask.any() else responses.new_zeros((0, response_length))
                #     responses_off = responses[off_mask, :] if off_mask.any() else responses.new_zeros((0, response_length))

                #     # 只用 on 计算 entropy
                #     entropy = safe_entropy(logits_on)

                #     # 分别计算 log_probs
                #     log_probs_on = safe_logprobs(logits_on, responses_on)
                #     log_probs_off = safe_logprobs(logits_off, responses_off)

                #     return entropy, log_probs_on, log_probs_off

                log_probs = logprobs_from_logits(logits, responses)  # (bsz, response_length)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

        return entropy, log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip actor optimizer step.")
        else:
            self.actor_optimizer.step()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = []
        if "pixel_values" in data.non_tensor_batch.keys():
            non_tensor_select_keys += ["pixel_values", "image_grid_thw"]
        if "image_sizes" in data.non_tensor_batch.keys():
            non_tensor_select_keys += ["image_sizes", "images"]
        if len(non_tensor_select_keys) == 0:
            non_tensor_select_keys = None

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        for micro_batch in tqdm(micro_batches, desc="Compute log probs", disable=(self.rank != 0)):
            micro_batch.to("cuda")
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            self._validate_actor_micro_batch(model_inputs, stage="compute_log_prob")
            _, log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            self._validate_finite_tensor("compute_log_prob.log_probs", log_probs, stage="compute_log_prob")
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages", "sft_coef_tensor"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        # build non_tensor_select_keys
        non_tensor_select_keys = []
        if "pixel_values" in data.non_tensor_batch.keys():
            non_tensor_select_keys += ["pixel_values", "image_grid_thw"]
        if "image_sizes" in data.non_tensor_batch.keys():
            non_tensor_select_keys += ["image_sizes", "images"]
        if "is_onpolicy" in data.non_tensor_batch.keys():
            non_tensor_select_keys += ["is_onpolicy"]
        if len(non_tensor_select_keys) == 0:
            non_tensor_select_keys = None
        
        # TODO (yaowei): support ppo epochs
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347


        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        n = len(mini_batches)
        for i, mini_batch in enumerate(mini_batches):
            micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

            self.actor_optimizer.zero_grad()
            for micro_batch in tqdm(micro_batches, desc=f"Update policy [{i + 1}/{n}]", disable=(self.rank != 0)):
                micro_batch.to("cuda")
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                stage = f"update_policy[{i + 1}/{n}]"
                self._validate_actor_micro_batch(model_inputs, stage=stage)
                responses = model_inputs["responses"]
                response_length = responses.size(1)
                attention_mask = model_inputs["attention_mask"]
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = model_inputs["old_log_probs"]
                advantages = model_inputs["advantages"]
                is_onpolicy = model_inputs["is_onpolicy"]
                sft_coef_tensor = model_inputs["sft_coef_tensor"]
                sft_coef = sft_coef_tensor.mean().item()

                
                if isinstance(is_onpolicy, list):
                    is_onpolicy = torch.tensor(is_onpolicy)

                on_mask = is_onpolicy.squeeze(-1).to(device=old_log_prob.device, dtype=torch.bool)
                # print("on_mask shape", on_mask.shape)


                                
                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(model_inputs, temperature=temperature)
                self._validate_forward_outputs(entropy, log_prob, responses, stage=stage)

                on_any  = bool(on_mask.any().item())
                off_any = bool((~on_mask).any().item())


                log_prob_on  = log_prob[on_mask, :]  if on_any else None
                log_prob_off = log_prob[~on_mask, :] if off_any else None
                entropy_on   = entropy[on_mask, :]   if on_any else None


                pg_loss = pg_clipfrac = ppo_kl = entropy_loss = kl_loss = None
                sft_loss = None
                clip_ratio = self.config.clip_ratio 
                entropy_coeff = self.config.entropy_coeff

                # ---- On-policy: RL ----
                if on_any:
                    old_log_prob_on   = old_log_prob[on_mask, :]
                    advantages_on     = advantages[on_mask, :]
                    response_mask_on  = response_mask[on_mask, :]

                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                        old_log_prob=old_log_prob_on,
                        log_prob=log_prob_on,
                        advantages=advantages_on,
                        eos_mask=response_mask_on,
                        cliprange=clip_ratio,
                    )
                    entropy_loss = verl_F.masked_mean(entropy_on, response_mask_on)
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob_on = model_inputs["ref_log_prob"][on_mask, :]
                        kld = core_algos.kl_penalty(
                            logprob=log_prob_on,
                            ref_logprob=ref_log_prob_on,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = masked_mean(kld, response_mask_on)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef

                # ---- Off-policy: SFT ----
                # sft_coef = self.sft_coef
                if off_any:

                    response_mask_off = response_mask[~on_mask, :]
                    # sft_coef_tensor = sft_coef_tensor[~on_mask, :] #(bs, 1)
                    # sft_coef = sft_coef_tensor.mean().item()
                    
                    # # p(1-p) 塑形，防止off-policy梯度过大压倒on-policy RL
                    # p = torch.exp(log_prob_off)
                    # weights = p * (1 - p)
                    # weighted_loss = -weights * log_prob_off
                    sft_loss = masked_mean(-log_prob_off, response_mask_off)

                # ---- 汇总 loss（只把存在的分支相加）----
                loss_terms = []
                if on_any:
                    loss_terms.append((1-sft_coef) * policy_loss)
                if off_any:
                    loss_terms.append(sft_coef * sft_loss)

                if loss_terms:
                    total_loss = loss_terms[0]
                    for t in loss_terms[1:]:
                        total_loss = total_loss + t
                    self._validate_loss(total_loss, stage=stage)
                    total_loss.backward()
                    self._maybe_sync_cuda("ROPD_SYNC_AFTER_BACKWARD")
                else:
                    metrics["actor/empty_micro_batch"] = 1.0  # 几乎走不到该分支

                def add_metric(name, value):
                    if value is None:
                        return
                    if isinstance(value, torch.Tensor):
                        value = value.detach().item()
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        if name not in metrics:
                            metrics[name] = []
                     
                        metrics[name].append(value)
                
                for k, v in {
                    "actor/entropy_loss": entropy_loss,
                    "actor/pg_loss":      pg_loss,
                    "actor/pg_clipfrac":  pg_clipfrac,
                    "actor/ppo_kl":       ppo_kl,
                    "actor/kl_loss":      kl_loss if on_any and self.config.use_kl_loss else None,
                    "actor/kl_coef":      getattr(self.config, "kl_loss_coef", None) if on_any else None,
                    "actor/sft_loss":     sft_loss if off_any else None,
                    "actor/sft_coef":     sft_coef if off_any else None,
                }.items():
                    add_metric(k, v)

            grad_norm = self._optimizer_step()
            self._maybe_sync_cuda("ROPD_SYNC_AFTER_OPTIM")
            append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
