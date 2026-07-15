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
Reward config
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardConfig:
    reward_type: str = "function"
    compute_score: str = "math"
    num_examine: int = 1
    ropd_model: str = "judge-model"
    ropd_teacher_model: Optional[str] = None
    ropd_rubricator_model: Optional[str] = None
    ropd_verifier_model: Optional[str] = None
    ropd_api_style: str = "responses"
    ropd_api_key_env: str = "JUDGE_API_KEY"
    ropd_base_url: Optional[str] = None
    ropd_base_url_env: str = "JUDGE_BASE_URL"
    ropd_teacher_answer_count: int = 1
    ropd_max_concurrency: int = 4
    ropd_request_timeout: float = 120.0
    ropd_teacher_temperature: Optional[float] = None
    ropd_rubricator_temperature: Optional[float] = None
    ropd_verifier_temperature: Optional[float] = None
    ropd_teacher_max_output_tokens: int = 2048
    ropd_rubricator_max_output_tokens: int = 4096
    ropd_verifier_max_output_tokens: int = 4096
    ropd_prompt_dir: str = "prompts/rcpc_rubric_judge"
    ropd_include_ground_truth: bool = True
    ropd_include_images: bool = True
    ropd_filter_teacher_by_answer: bool = True
    ropd_print_teacher_outputs: bool = True
    ropd_print_student_outputs: bool = False
    ropd_print_max_student_outputs: int = 0
    ropd_print_rubric_outputs: bool = False
    ropd_print_verifier_outputs: bool = False
    ropd_require_strict_cot_format: bool = True
    # Keep strict format as a measurable signal without erasing semantic reward.
    # Legacy all-or-nothing behavior remains available as an explicit override.
    ropd_zero_score_on_format_error: bool = False
    ropd_zero_criteria_on_format_error: bool = False
    ropd_final_label_points_cap: int = 1
    ropd_format_points_cap: int = 1
    ropd_max_image_bytes: int = 8388608
    ropd_fallback_to_ipr: bool = True
    ropd_score_offpolicy: bool = False
    ropd_use_criterion_advantage: bool = False
    ropd_shadow_attribution_enabled: bool = False
    ropd_shadow_attribution_max_groups_per_batch: int = 1
    ropd_shadow_attribution_max_output_tokens: int = 4096
    ropd_print_shadow_attributions: bool = False
    ropd_rcpc_enabled: bool = False
    ropd_rcpc_use_token_advantage: bool = True
    # Canonical RCPC intervention budget B_group. When
    # ropd_rcpc_derive_candidates_from_budget is true, this single knob
    # determines the candidate/action pool and the number of causal
    # interventions per prompt group.
    ropd_rcpc_budget: int = 32
    ropd_rcpc_derive_candidates_from_budget: bool = True
    # Advanced override knobs for ablations only. They are ignored by the
    # default budget-derived path above.
    ropd_rcpc_top_actions: int = 12
    ropd_rcpc_top_blocks: int = 6
    ropd_rcpc_min_action_chars: int = 12
    ropd_rcpc_max_action_chars: int = 260
    ropd_rcpc_max_action_tokens: int = 24
    ropd_rcpc_min_robust_denom: float = 0.05
    ropd_rcpc_min_anchor_z: float = 0.5
    ropd_rcpc_intervention_enabled: bool = False
    # -1 means all prompt groups in the batch. 0 disables intervention.
    ropd_rcpc_intervention_max_groups_per_batch: int = -1
    # Advanced override knobs for ablations only. In the default path,
    # ropd_rcpc_budget sets the actual prompt-group intervention budget and
    # per-answer caps are disabled.
    ropd_rcpc_intervention_max_blocks_per_answer: int = 0
    ropd_rcpc_intervention_max_blocks_per_group: int = 32
    ropd_rcpc_intervention_mode: str = "mask"
    ropd_rcpc_batch_counterfactual: bool = True
    # Number of paired prefix-regeneration samples per arm and selected block.
    # m=2 estimates both the paired effect and its sampling uncertainty.
    ropd_rcpc_counterfactual_samples: int = 2
    # Prefix-regeneration counterfactuals are generated batch-wide. <=0 means
    # one request for all counterfactuals, which avoids repeated vLLM
    # wake/sync/sleep cycles; set a positive value only if the single request is
    # too large for the runtime.
    ropd_rcpc_counterfactual_batch_size: int = 0
    # Bound each LLM-as-judge request independently from counterfactual
    # generation batching. Large verifier payloads are more likely to return
    # incomplete structured output.
    ropd_rcpc_verifier_batch_size: int = 12
    # The verifier packs factual/control pairs by both answer count and the
    # rendered prompt token count. This leaves headroom for structured output
    # under the 40k-context judge endpoints used by the training jobs.
    ropd_rcpc_verifier_max_input_tokens: int = 28000
    ropd_rcpc_verifier_min_output_tokens: int = 256
    ropd_rcpc_verifier_output_tokens_per_answer: int = 160
    ropd_rcpc_verifier_max_retries: int = 2
    # Weak variance floor on one paired outcome. The estimator divides it by
    # the valid pair count, so uncertainty still decreases with more samples.
    ropd_rcpc_effect_variance_prior: float = 0.1
    # Production RCPC runs should fail visibly instead of silently reverting to
    # baseline token advantages when an intervention plan cannot be scored.
    ropd_rcpc_fail_on_intervention_error: bool = False
    ropd_rcpc_transport_lambda: float = 1.0
    ropd_rcpc_fallback_to_criterion_advantage: bool = True
    ropd_print_rcpc_outputs: bool = False
    ropd_rcpc_print_intervention_summary: bool = True
    ropd_rcpc_print_interval: int = 10
    ropd_rcpc_print_max_groups: int = 1
    ropd_rcpc_print_max_blocks: int = 16
    ropd_debug_path: Optional[str] = None
