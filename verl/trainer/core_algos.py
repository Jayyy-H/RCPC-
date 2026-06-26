# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


if TYPE_CHECKING:
    from verl.trainer.config import AlgorithmConfig


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef: float):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(algorithm_config: "AlgorithmConfig"):
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError("Unknown kl_ctrl type")

    return kl_ctrl


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# ROPD may provide a precomputed criterion-level advantage. Other callers keep
# the original outcome-reward GRPO behavior.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    is_onpolicy,
    criterion_advantages: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
):
    """
    Compute advantage for GRPO. When criterion_advantages is provided, use it
    directly; otherwise operate on one scalar outcome reward per response.
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    if criterion_advantages is not None:
        if criterion_advantages.shape != token_level_rewards.shape:
            raise ValueError(
                "criterion_advantages shape {} != token_level_rewards shape {}".format(
                    tuple(criterion_advantages.shape), tuple(token_level_rewards.shape)
                )
            )
        advantages = criterion_advantages.to(
            device=token_level_rewards.device,
            dtype=token_level_rewards.dtype,
        ) * eos_mask
        return advantages, advantages

    device = token_level_rewards.device
    dtype  = token_level_rewards.dtype

    bs, response_length = token_level_rewards.shape
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    # if not torch.is_tensor(index):
    #     index = torch.tensor(index, device=device)
    # index = index.view(-1).to(device)

    if is_onpolicy is None:
        onpol = torch.ones(bs, dtype=torch.bool, device=device)
    else:
        if not torch.is_tensor(is_onpolicy):
            is_onpolicy = torch.as_tensor(is_onpolicy, device=device)
        onpol = is_onpolicy.view(-1).to(device=device, dtype=torch.bool)

    # 仅用 on-policy 样本计算组统计量
    id2scores = defaultdict(list)
    with torch.no_grad():
        for i in range(bs):
            if onpol[i]:
                id2scores[index[i]].append(scores[i])

        id2mean, id2std = {}, {}
        for idx, lst in id2scores.items():
            if len(lst) <= 1:
                # 与原逻辑一致：样本不足时退化为 (0, 1)
                id2mean[idx] = torch.tensor(0.0, device=device, dtype=dtype)
                id2std[idx]  = torch.tensor(1.0, device=device, dtype=dtype)
            else:
                stack = torch.stack(lst)          # (k,)
                id2mean[idx] = stack.mean()       # 默认 unbiased=True 的 std 语义
                id2std[idx]  = stack.std()        # 不再将小 std 替换为 1.0

    # 组归一化仅作用于 on-policy；off-policy 直接为 1
    norm_scores = torch.ones_like(scores, device=device, dtype=dtype)
    with torch.no_grad():
        for i in range(bs):
            if onpol[i]:
                idx = index[i]
                mean = id2mean.get(idx, torch.tensor(0.0, device=device, dtype=dtype))
                std  = id2std.get(idx,  torch.tensor(1.0, device=device, dtype=dtype))
                norm_scores[i] = (scores[i] - mean) / (std + epsilon)
            # else: 保持 1.0

    norm_scores = norm_scores.unsqueeze(-1).expand(-1, response_length) * eos_mask
    return norm_scores, norm_scores


def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, gamma: torch.Tensor
):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * eos_mask[:, t]

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask

    return advantages, returns


def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, eos_mask: torch.Tensor
):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    # scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(
    old_log_prob, log_prob, advantages, eos_mask, cliprange
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.Tensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
