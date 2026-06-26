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
    ropd_model: str = "gpt-5.4"
    ropd_teacher_model: Optional[str] = None
    ropd_rubricator_model: Optional[str] = None
    ropd_verifier_model: Optional[str] = None
    ropd_api_style: str = "responses"
    ropd_api_key_env: str = "OPENAI_API_KEY"
    ropd_base_url: Optional[str] = None
    ropd_base_url_env: str = "OPENAI_BASE_URL"
    ropd_teacher_answer_count: int = 1
    ropd_max_concurrency: int = 4
    ropd_request_timeout: float = 120.0
    ropd_teacher_temperature: Optional[float] = None
    ropd_rubricator_temperature: Optional[float] = None
    ropd_verifier_temperature: Optional[float] = None
    ropd_teacher_max_output_tokens: int = 2048
    ropd_rubricator_max_output_tokens: int = 4096
    ropd_verifier_max_output_tokens: int = 4096
    ropd_prompt_dir: str = "prompts/ropd"
    ropd_include_ground_truth: bool = True
    ropd_include_images: bool = True
    ropd_filter_teacher_by_answer: bool = True
    ropd_print_teacher_outputs: bool = True
    ropd_print_student_outputs: bool = False
    ropd_print_rubric_outputs: bool = False
    ropd_print_verifier_outputs: bool = False
    ropd_require_strict_cot_format: bool = True
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
    ropd_debug_path: Optional[str] = None
