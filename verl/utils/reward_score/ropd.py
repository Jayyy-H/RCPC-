import base64
import hashlib
import json
import mimetypes
import os
import re
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from verl import DataProto
from verl.utils.reward_score.ipr import ipr_compute_score
from verl.utils.reward_score.rcpc import (
    apply_intervention,
    build_candidates,
    build_group_candidates,
    build_token_advantages,
    build_token_offsets,
    measure_deleted_content_reappearance,
    paired_effect_statistics,
)


RUBRIC_SCHEMA_VERSION = "ropd.rubric.v1"
BATCH_VERIFIER_SCHEMA_VERSION = "ropd.batch_verifier.v2"
SHADOW_ATTRIBUTION_SCHEMA_VERSION = "ropd.shadow_attribution.v1"


_PLACEHOLDER_RE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_SUBSTANTIVE_RE = re.compile(r"[\w\u4e00-\u9fff]", re.UNICODE)

# Keep the training dashboard focused on whether RCPC is producing a reliable
# and optimization-relevant perturbation. Detailed constants and raw counters
# remain available in the printed intervention summaries/debug JSON.
_RCPC_LOG_METRICS = frozenset(
    {
        "rcpc/effective_group_ratio",
        "rcpc/intervention_plan_success_ratio",
        "rcpc/intervention_plan_error_count",
        "rcpc/intervention_item_count",
        "rcpc/pair_semantic_valid_ratio",
        "rcpc/successful_response_ratio",
        "rcpc/successful_advantage_delta_l1_ratio",
        "rcpc/successful_advantage_cosine_to_baseline",
        "rcpc/successful_transport_kl",
        "rcpc/token_coverage",
        "rcpc/nonzero_blocks",
        "rcpc/calibrated_effect_abs_mean",
        "rcpc/effect_nonzero_ratio",
        "rcpc/effect_standard_error_mean",
        "rcpc/effect_reliability_mean",
        "rcpc/deleted_content_reappearance_rate",
        "rcpc/verifier_retry_ratio",
        "rcpc/verifier_leaf_failure_ratio",
    }
)
_RCPC_LOG_TIMINGS = frozenset(
    {
        "timing_s/rcpc/counterfactual_generation_wall",
        "timing_s/rcpc/intervention_score_wall",
        "timing_s/rcpc/pipeline_wall",
    }
)


def _has_strict_cot_format(text: str) -> bool:
    reasoning_matches = list(_REASONING_RE.finditer(str(text)))
    answer_matches = list(_ANSWER_RE.finditer(str(text)))
    if len(reasoning_matches) != 1 or len(answer_matches) != 1:
        return False

    reasoning_match = reasoning_matches[0]
    answer_match = answer_matches[0]
    if reasoning_match.end() > answer_match.start():
        return False
    if not _SUBSTANTIVE_RE.search(reasoning_match.group(1).strip()):
        return False

    return bool(_SUBSTANTIVE_RE.search(answer_match.group(1).strip()))


def _cfg(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _env_or_value(value: Optional[str], env_name: Optional[str]) -> Optional[str]:
    if value is not None and str(value).strip():
        return str(value).strip()
    if env_name is None:
        return None
    env_value = os.getenv(env_name)
    if env_value is not None and env_value.strip():
        return env_value.strip()
    return None


def _normalize_openai_base_url(base_url: Optional[str], api_style: str) -> Optional[str]:
    if base_url is None:
        return None
    normalized = str(base_url).strip().rstrip("/")
    # OpenAI SDK appends `/responses` for `client.responses.create(...)`.
    # Some gateways store the full responses endpoint in their base URL.
    if api_style == "responses" and normalized.endswith("/responses"):
        normalized = normalized[: -len("/responses")]
    if api_style == "chat_completions" and normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return normalized


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def _is_structural_verifier_error(exc: Exception) -> bool:
    """Return true when retrying the same verifier payload cannot fix it."""
    if isinstance(exc, json.JSONDecodeError):
        return True
    message = str(exc).lower()
    markers = (
        "schema_version mismatch",
        "answer count mismatch",
        "judgement length mismatch",
        "answer_index must preserve input order",
        "judgement values must be booleans",
        "answers for chunk size",
    )
    return any(marker in message for marker in markers)


def _extract_json_payload(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("Judge output must be a JSON object.")
    return payload


def _render_template(template: str, replacements: Mapping[str, str]) -> str:
    unknown_placeholders = {
        match.group(1) for match in _PLACEHOLDER_RE.finditer(template) if match.group(1) not in replacements
    }
    if unknown_placeholders:
        raise ValueError("Unsupported prompt placeholder(s): " + ", ".join(sorted(unknown_placeholders)))
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _render_answer_block(label: str, answers: Sequence[str], start_index: int = 1) -> str:
    return "\n\n".join(
        "{} {}:\n{}".format(label, index, answer)
        for index, answer in enumerate(answers, start=start_index)
    )


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray, str)):
        value = value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _population_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _sample_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _compute_group_criterion_advantages(
    rubric: Mapping[str, Any],
    verifier_answers: Sequence[Mapping[str, Any]],
    format_valid: Sequence[bool],
    *,
    require_strict_cot_format: bool,
    epsilon: float = 1e-6,
) -> Tuple[List[float], Dict[str, Dict[str, List[float]]]]:
    """Compute one group-relative advantage per rubric, then combine by active rubric points."""
    rubrics = list(rubric["rubrics"])
    response_count = len(verifier_answers)
    combined = [0.0] * response_count
    active_weight = 0.0
    criterion_stats: Dict[str, Dict[str, List[float]]] = {}

    for criterion_index, criterion in enumerate(rubrics):
        judgements = []
        for response_index, answer in enumerate(verifier_answers):
            judgement = bool(answer["judgement"][criterion_index])
            if require_strict_cot_format and not format_valid[response_index]:
                judgement = False
            judgements.append(1.0 if judgement else 0.0)

        mean = sum(judgements) / len(judgements) if judgements else 0.0
        std = _sample_std(judgements)
        if std > epsilon:
            criterion_advantages = [(value - mean) / (std + epsilon) for value in judgements]
            weight = float(criterion["points"])
            active_weight += weight
            for response_index, advantage in enumerate(criterion_advantages):
                combined[response_index] += weight * advantage
        else:
            criterion_advantages = [0.0] * response_count

        criterion_stats[str(criterion["criterion_id"])] = {
            "judgements": judgements,
            "advantages": criterion_advantages,
        }

    if active_weight > 0:
        combined = [value / active_weight for value in combined]
        # Keep the final policy-gradient scale comparable to ordinary GRPO,
        # independent of how many active criteria the rubric contains.
        combined_std = _sample_std(combined)
        if combined_std > epsilon:
            combined = [value / (combined_std + epsilon) for value in combined]
    return combined, criterion_stats


def _compute_outcome_group_advantages(scores: Sequence[float], epsilon: float = 1e-6) -> List[float]:
    """Fallback used only when rubric/verifier generation fails for a group."""
    if not scores:
        return []
    std = _sample_std(scores)
    if std <= epsilon:
        return [0.0] * len(scores)
    mean = sum(scores) / len(scores)
    return [(score - mean) / (std + epsilon) for score in scores]


class RopdOpenAIClient:
    def __init__(
        self,
        *,
        api_key_env: str,
        base_url: Optional[str],
        base_url_env: str,
        api_style: str,
        timeout: float,
        max_retries: int,
        include_images: bool,
        max_image_bytes: int,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "ROPD reward requires the `openai` package. Install requirements.txt before using compute_score=ropd_ipr."
            ) from exc

        api_key = _env_or_value(None, api_key_env)
        if api_key is None:
            raise RuntimeError(
                "ROPD reward requires an API key. Set {} or change worker.reward.ropd_api_key_env.".format(
                    api_key_env
                )
            )

        self.api_style = api_style
        if self.api_style not in {"responses", "chat_completions"}:
            raise ValueError("worker.reward.ropd_api_style must be `responses` or `chat_completions`.")
        resolved_base_url = _normalize_openai_base_url(_env_or_value(base_url, base_url_env), self.api_style)
        self.include_images = include_images
        self.max_image_bytes = max_image_bytes
        client_kwargs = {
            "api_key": api_key,
            "timeout": timeout,
            # RCPC owns validation-aware retry and request splitting. Keep SDK
            # retries bounded so one malformed response cannot multiply tail
            # latency invisibly below that layer.
            "max_retries": max(0, int(max_retries)),
        }
        if resolved_base_url is not None:
            client_kwargs["base_url"] = resolved_base_url
        self.client = OpenAI(**client_kwargs)

    def create_text(
        self,
        *,
        model: str,
        text: str,
        image_paths: Sequence[str],
        temperature: Optional[float],
        max_output_tokens: int,
        json_mode: bool,
    ) -> str:
        image_urls = self._image_paths_to_urls(image_paths) if self.include_images else []
        if self.api_style == "chat_completions":
            messages = self._build_chat_messages(text, image_urls)
            request_kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_output_tokens,
            }
            if temperature is not None:
                request_kwargs["temperature"] = temperature
            if json_mode:
                request_kwargs["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(**request_kwargs)
            return (response.choices[0].message.content or "").strip()

        input_payload = self._build_responses_input(text, image_urls)
        request_kwargs = {
            "model": model,
            "input": input_payload,
            "max_output_tokens": max_output_tokens,
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if json_mode:
            request_kwargs["text"] = {"format": {"type": "json_object"}}
        response = self.client.responses.create(**request_kwargs)
        return self._extract_responses_text(response).strip()

    def _image_paths_to_urls(self, image_paths: Sequence[str]) -> List[str]:
        urls = []
        for image_path in image_paths:
            if image_path is None:
                continue
            image_text = str(image_path).strip()
            if not image_text:
                continue
            if image_text.startswith(("http://", "https://", "data:")):
                urls.append(image_text)
                continue

            path = Path(image_text).expanduser()
            if not path.exists() or not path.is_file():
                continue
            if self.max_image_bytes > 0 and path.stat().st_size > self.max_image_bytes:
                continue
            mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            urls.append("data:{};base64,{}".format(mime_type, encoded))
        return urls

    def count_usable_image_paths(self, image_paths: Sequence[str]) -> int:
        count = 0
        for image_path in image_paths:
            if image_path is None:
                continue
            image_text = str(image_path).strip()
            if not image_text:
                continue
            if image_text.startswith(("http://", "https://", "data:")):
                count += 1
                continue

            path = Path(image_text).expanduser()
            if not path.exists() or not path.is_file():
                continue
            if self.max_image_bytes > 0 and path.stat().st_size > self.max_image_bytes:
                continue
            count += 1
        return count

    def skipped_image_paths(self, image_paths: Sequence[str]) -> List[str]:
        skipped = []
        for image_path in image_paths:
            if image_path is None:
                continue
            image_text = str(image_path).strip()
            if not image_text or image_text.startswith(("http://", "https://", "data:")):
                continue
            path = Path(image_text).expanduser()
            if not path.exists() or not path.is_file():
                skipped.append("{} (missing)".format(image_text))
            elif self.max_image_bytes > 0 and path.stat().st_size > self.max_image_bytes:
                skipped.append("{} ({} bytes > max {})".format(image_text, path.stat().st_size, self.max_image_bytes))
        return skipped[:5]

    def _build_responses_input(self, text: str, image_urls: Sequence[str]) -> List[Dict[str, Any]]:
        content = [{"type": "input_text", "text": text}]
        for image_url in image_urls:
            content.append({"type": "input_image", "image_url": image_url})
        return [{"role": "user", "content": content}]

    def _build_chat_messages(self, text: str, image_urls: Sequence[str]) -> List[Dict[str, Any]]:
        if not image_urls:
            # Match the known-working GPT-5.5 gateway request shape used by
            # the data-generation pipeline. Some OpenAI-compatible gateways
            # accept typed multimodal content only when media is present.
            return [{"role": "user", "content": text}]
        content = [{"type": "text", "text": text}]
        for image_url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        return [{"role": "user", "content": content}]

    def _extract_responses_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)

        parts = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)


class RopdIPRRewardScorer:
    def __init__(self, *, tokenizer: Any, reward_config: Any, num_examine: int = 0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.already_print = 0
        self._print_lock = threading.Lock()
        self._counterfactual_lock = threading.Lock()
        self._current_global_step = -1
        self._rcpc_summary_printed_by_step: Dict[int, int] = {}
        self._counterfactual_generator: Optional[Callable[[Sequence[Mapping[str, Any]]], List[Any]]] = None
        self._counterfactual_session_factory: Optional[Callable[[], Any]] = None

        base_model = os.getenv("JUDGE_MODEL") or os.getenv("ROPD_MODEL") or _cfg(
            reward_config, "ropd_model", "judge-model"
        )
        self.teacher_model = os.getenv("ROPD_TEACHER_MODEL") or _cfg(
            reward_config, "ropd_teacher_model", None
        ) or base_model
        self.rubricator_model = os.getenv("ROPD_RUBRICATOR_MODEL") or _cfg(
            reward_config, "ropd_rubricator_model", None
        ) or base_model
        self.verifier_model = os.getenv("ROPD_VERIFIER_MODEL") or _cfg(
            reward_config, "ropd_verifier_model", None
        ) or base_model

        self.teacher_answer_count = max(1, int(_cfg(reward_config, "ropd_teacher_answer_count", 1)))
        self.max_concurrency = max(1, int(_cfg(reward_config, "ropd_max_concurrency", 4)))
        self._rcpc_verifier_executor = ThreadPoolExecutor(
            max_workers=self.max_concurrency,
            thread_name_prefix="rcpc-verifier",
        )
        self.include_ground_truth = bool(_cfg(reward_config, "ropd_include_ground_truth", True))
        self.filter_teacher_by_answer = bool(_cfg(reward_config, "ropd_filter_teacher_by_answer", True))
        self.print_teacher_outputs = bool(_cfg(reward_config, "ropd_print_teacher_outputs", True))
        self.print_student_outputs = bool(_cfg(reward_config, "ropd_print_student_outputs", False))
        self.print_max_student_outputs = int(_cfg(reward_config, "ropd_print_max_student_outputs", 0))
        self.print_rubric_outputs = bool(_cfg(reward_config, "ropd_print_rubric_outputs", False))
        self.print_verifier_outputs = bool(_cfg(reward_config, "ropd_print_verifier_outputs", False))
        self.print_shadow_attributions = bool(_cfg(reward_config, "ropd_print_shadow_attributions", False))
        self.fallback_to_ipr = bool(_cfg(reward_config, "ropd_fallback_to_ipr", True))
        self.score_offpolicy = bool(_cfg(reward_config, "ropd_score_offpolicy", False))
        self.use_criterion_advantage = bool(_cfg(reward_config, "ropd_use_criterion_advantage", False))
        self.shadow_attribution_enabled = bool(_cfg(reward_config, "ropd_shadow_attribution_enabled", False))
        self.shadow_attribution_max_groups_per_batch = max(
            0, int(_cfg(reward_config, "ropd_shadow_attribution_max_groups_per_batch", 1))
        )
        self.rcpc_enabled = bool(_cfg(reward_config, "ropd_rcpc_enabled", False))
        self.rcpc_use_token_advantage = bool(_cfg(reward_config, "ropd_rcpc_use_token_advantage", True))
        self.rcpc_budget = max(1, int(_cfg(reward_config, "ropd_rcpc_budget", 32)))
        self.rcpc_derive_candidates_from_budget = bool(
            _cfg(reward_config, "ropd_rcpc_derive_candidates_from_budget", True)
        )
        if self.rcpc_derive_candidates_from_budget:
            # The method-level budget is B_group: the number of local causal
            # interventions allowed for one prompt group. Candidate discovery
            # is derived from the same knob so experiments do not silently use
            # inconsistent action/block/intervention budgets.
            self.rcpc_top_blocks = self.rcpc_budget
            self.rcpc_top_actions = max(2, self.rcpc_budget * 2)
        else:
            self.rcpc_top_actions = max(1, int(_cfg(reward_config, "ropd_rcpc_top_actions", 12)))
            self.rcpc_top_blocks = max(1, int(_cfg(reward_config, "ropd_rcpc_top_blocks", 6)))
        self.rcpc_min_action_chars = max(1, int(_cfg(reward_config, "ropd_rcpc_min_action_chars", 12)))
        self.rcpc_max_action_chars = max(1, int(_cfg(reward_config, "ropd_rcpc_max_action_chars", 260)))
        self.rcpc_max_action_tokens = max(1, int(_cfg(reward_config, "ropd_rcpc_max_action_tokens", 24)))
        self.rcpc_min_robust_denom = float(_cfg(reward_config, "ropd_rcpc_min_robust_denom", 0.05))
        self.rcpc_min_anchor_z = float(_cfg(reward_config, "ropd_rcpc_min_anchor_z", 0.5))
        self.rcpc_intervention_enabled = bool(_cfg(reward_config, "ropd_rcpc_intervention_enabled", False))
        self.rcpc_intervention_max_groups_per_batch = int(
            _cfg(reward_config, "ropd_rcpc_intervention_max_groups_per_batch", -1)
        )
        if self.rcpc_derive_candidates_from_budget:
            self.rcpc_intervention_max_blocks_per_answer = 0
            self.rcpc_intervention_max_blocks_per_group = self.rcpc_budget
        else:
            self.rcpc_intervention_max_blocks_per_answer = int(
                _cfg(reward_config, "ropd_rcpc_intervention_max_blocks_per_answer", 0)
            )
            self.rcpc_intervention_max_blocks_per_group = max(
                0, int(_cfg(reward_config, "ropd_rcpc_intervention_max_blocks_per_group", 16))
            )
        self.rcpc_intervention_mode = str(_cfg(reward_config, "ropd_rcpc_intervention_mode", "mask"))
        self.rcpc_batch_counterfactual = bool(_cfg(reward_config, "ropd_rcpc_batch_counterfactual", True))
        self.rcpc_overlap_generation_and_judge = bool(
            _cfg(reward_config, "ropd_rcpc_overlap_generation_and_judge", False)
        )
        self.rcpc_counterfactual_samples = max(
            1, int(_cfg(reward_config, "ropd_rcpc_counterfactual_samples", 2))
        )
        self.rcpc_counterfactual_batch_size = int(
            _cfg(reward_config, "ropd_rcpc_counterfactual_batch_size", 0)
        )
        self.rcpc_verifier_batch_size = max(
            1, int(_cfg(reward_config, "ropd_rcpc_verifier_batch_size", 12))
        )
        self.rcpc_verifier_max_input_tokens = max(
            512, int(_cfg(reward_config, "ropd_rcpc_verifier_max_input_tokens", 28000))
        )
        self.rcpc_verifier_min_output_tokens = max(
            64, int(_cfg(reward_config, "ropd_rcpc_verifier_min_output_tokens", 256))
        )
        self.rcpc_verifier_output_tokens_per_answer = max(
            32,
            int(_cfg(reward_config, "ropd_rcpc_verifier_output_tokens_per_answer", 160)),
        )
        self.rcpc_verifier_max_retries = max(
            0, int(_cfg(reward_config, "ropd_rcpc_verifier_max_retries", 2))
        )
        self.rcpc_effect_variance_prior = max(
            0.0, float(_cfg(reward_config, "ropd_rcpc_effect_variance_prior", 0.1))
        )
        self.rcpc_fail_on_intervention_error = bool(
            _cfg(reward_config, "ropd_rcpc_fail_on_intervention_error", False)
        )
        self.rcpc_transport_lambda = float(_cfg(reward_config, "ropd_rcpc_transport_lambda", 1.0))
        self.rcpc_fallback_to_criterion_advantage = bool(
            _cfg(reward_config, "ropd_rcpc_fallback_to_criterion_advantage", True)
        )
        self.print_rcpc_outputs = bool(_cfg(reward_config, "ropd_print_rcpc_outputs", False))
        self.rcpc_print_intervention_summary = bool(
            _cfg(reward_config, "ropd_rcpc_print_intervention_summary", True)
        )
        self.rcpc_print_interval = int(_cfg(reward_config, "ropd_rcpc_print_interval", 10))
        self.rcpc_print_max_groups = int(_cfg(reward_config, "ropd_rcpc_print_max_groups", 1))
        self.rcpc_print_max_blocks = int(_cfg(reward_config, "ropd_rcpc_print_max_blocks", 16))
        self.require_strict_cot_format = bool(_cfg(reward_config, "ropd_require_strict_cot_format", True))
        self.zero_score_on_format_error = bool(
            _cfg(reward_config, "ropd_zero_score_on_format_error", self.require_strict_cot_format)
        )
        self.zero_criteria_on_format_error = bool(
            _cfg(reward_config, "ropd_zero_criteria_on_format_error", self.require_strict_cot_format)
        )
        self.final_label_points_cap = int(_cfg(reward_config, "ropd_final_label_points_cap", 1))
        self.format_points_cap = int(_cfg(reward_config, "ropd_format_points_cap", 1))
        self.teacher_temperature = _optional_float(_cfg(reward_config, "ropd_teacher_temperature", None))
        self.rubricator_temperature = _optional_float(_cfg(reward_config, "ropd_rubricator_temperature", None))
        self.verifier_temperature = _optional_float(_cfg(reward_config, "ropd_verifier_temperature", None))
        self.teacher_max_output_tokens = int(_cfg(reward_config, "ropd_teacher_max_output_tokens", 2048))
        self.rubricator_max_output_tokens = int(_cfg(reward_config, "ropd_rubricator_max_output_tokens", 4096))
        self.verifier_max_output_tokens = int(_cfg(reward_config, "ropd_verifier_max_output_tokens", 4096))
        self.shadow_attribution_max_output_tokens = int(
            _cfg(reward_config, "ropd_shadow_attribution_max_output_tokens", 4096)
        )

        repo_root = Path(__file__).resolve().parents[3]
        prompt_dir_value = str(_cfg(reward_config, "ropd_prompt_dir", "prompts/rcpc_rubric_judge"))
        prompt_dir = Path(prompt_dir_value)
        if not prompt_dir.is_absolute():
            prompt_dir = repo_root / prompt_dir

        def _read_optional_prompt(filename: str) -> Optional[str]:
            path = prompt_dir / filename
            if path.exists():
                return path.read_text(encoding="utf-8")
            return None

        self.teacher_template = _read_optional_prompt("teacher.txt")
        self.rubricator_template = _read_optional_prompt("rubricator.txt")
        self.verifier_template = (prompt_dir / "verifier.txt").read_text(encoding="utf-8")
        self.attributor_template = None
        if self.shadow_attribution_enabled:
            self.attributor_template = _read_optional_prompt("attributor.txt")
            if self.attributor_template is None:
                raise FileNotFoundError(f"Missing attributor prompt: {prompt_dir / 'attributor.txt'}")

        debug_path_value = _cfg(reward_config, "ropd_debug_path", None)
        self.debug_path = Path(debug_path_value) if debug_path_value else None
        if self.debug_path is not None and not self.debug_path.is_absolute():
            self.debug_path = repo_root / self.debug_path

        self.client = RopdOpenAIClient(
            api_key_env=str(_cfg(reward_config, "ropd_api_key_env", "JUDGE_API_KEY")),
            base_url=_cfg(reward_config, "ropd_base_url", None),
            base_url_env=str(_cfg(reward_config, "ropd_base_url_env", "JUDGE_BASE_URL")),
            api_style=str(_cfg(reward_config, "ropd_api_style", "responses")),
            timeout=float(_cfg(reward_config, "ropd_request_timeout", 120.0)),
            max_retries=int(_cfg(reward_config, "ropd_openai_max_retries", 1)),
            include_images=bool(_cfg(reward_config, "ropd_include_images", True)),
            max_image_bytes=int(_cfg(reward_config, "ropd_max_image_bytes", 8388608)),
        )

    def set_counterfactual_generator(
        self,
        generator: Optional[Callable[[Sequence[Mapping[str, Any]]], List[Any]]],
    ) -> None:
        self._counterfactual_generator = generator

    def set_counterfactual_session_factory(
        self,
        session_factory: Optional[Callable[[], Any]],
    ) -> None:
        self._counterfactual_session_factory = session_factory

    def __call__(self, data: DataProto) -> torch.Tensor:
        call_start = time.perf_counter()
        call_metrics: Dict[str, float] = {}
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        self._current_global_step = int(data.meta_info.get("global_step", -1))
        section_start = time.perf_counter()
        response_infos = self._collect_response_infos(data)
        call_metrics["timing_s/reward/collect_response_infos"] = time.perf_counter() - section_start
        skip_rcpc_credit = bool(data.meta_info.get("skip_rcpc_credit", False))

        if not self._has_training_group_keys(data):
            self._fill_rule_rewards(reward_tensor, response_infos)
            return reward_tensor

        grouped = self._group_onpolicy_infos(response_infos)
        groups = list(grouped.values())
        for group in groups:
            group[0]["skip_rcpc_credit"] = skip_rcpc_credit
        defer_rcpc_interventions = self._should_defer_rcpc_interventions(skip_rcpc_credit)
        if self.shadow_attribution_enabled:
            for group_index, group in enumerate(groups):
                group[0]["run_shadow_attribution"] = (
                    group_index < self.shadow_attribution_max_groups_per_batch
                )
        if self.rcpc_enabled and self.rcpc_intervention_enabled and not skip_rcpc_credit:
            for group_index, group in enumerate(groups):
                run_intervention = (
                    self.rcpc_intervention_max_groups_per_batch < 0
                    or group_index < self.rcpc_intervention_max_groups_per_batch
                )
                group[0]["run_rcpc_intervention"] = (
                    run_intervention and not defer_rcpc_interventions
                )
                group[0]["deferred_rcpc_intervention"] = run_intervention and defer_rcpc_interventions
                if defer_rcpc_interventions:
                    group[0]["defer_group_print"] = True
        call_metrics["reward/group_count"] = float(len(groups))
        call_metrics["reward/group_size/mean"] = (
            sum(len(group) for group in groups) / len(groups) if groups else 0.0
        )
        call_metrics["reward/max_concurrency"] = float(min(self.max_concurrency, max(1, len(groups))))
        section_start = time.perf_counter()
        if self.max_concurrency <= 1 or len(groups) <= 1:
            results = [self._score_group(group) for group in groups]
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(groups))) as executor:
                results = list(executor.map(self._score_group, groups))
        call_metrics["timing_s/reward/score_groups_wall"] = time.perf_counter() - section_start

        if defer_rcpc_interventions:
            section_start = time.perf_counter()
            call_metrics.update(self._run_deferred_rcpc_interventions(groups, results))
            call_metrics["timing_s/reward/deferred_rcpc_wall"] = time.perf_counter() - section_start
            for group, result in zip(groups, results):
                self._maybe_print_group(group[0], result)

        section_start = time.perf_counter()
        for result in results:
            for batch_index, score in result["scores"].items():
                response_length = response_infos[batch_index]["response_length"]
                if response_length > 0:
                    reward_tensor[batch_index, response_length - 1] = float(score)
        call_metrics["timing_s/reward/fill_reward_tensor"] = time.perf_counter() - section_start

        if self.use_criterion_advantage:
            section_start = time.perf_counter()
            criterion_advantage_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            for result in results:
                rcpc_token_advantages = result.get("rcpc_token_advantages", {}) if self.rcpc_enabled else {}
                for batch_index, advantage in result.get("criterion_advantages", {}).items():
                    response_length = response_infos[batch_index]["response_length"]
                    if response_length <= 0:
                        continue
                    token_values = rcpc_token_advantages.get(batch_index)
                    if self.rcpc_enabled and self.rcpc_use_token_advantage and token_values is not None:
                        values = torch.as_tensor(token_values[:response_length], dtype=torch.float32)
                        if values.numel() < response_length:
                            values = torch.nn.functional.pad(values, (0, response_length - values.numel()))
                        criterion_advantage_tensor[batch_index, :response_length] = values[:response_length]
                    else:
                        criterion_advantage_tensor[batch_index, :response_length] = float(advantage)
            data.batch["criterion_advantages"] = criterion_advantage_tensor
            call_metrics["timing_s/reward/build_criterion_tensor"] = time.perf_counter() - section_start
            metrics = self._collect_criterion_metrics(results)
            call_metrics["timing_s/reward/total"] = time.perf_counter() - call_start
            metrics.update(call_metrics)
            metrics = self._filter_logged_metrics(metrics)
            data.meta_info["ropd_metrics"] = metrics

        if self.score_offpolicy:
            for info in response_infos:
                if not info["is_onpolicy"] and info["response_length"] > 0:
                    reward_tensor[info["batch_index"], info["response_length"] - 1] = self._rule_score(info)

        section_start = time.perf_counter()
        self._write_debug(results)
        if self.use_criterion_advantage and "ropd_metrics" in data.meta_info:
            data.meta_info["ropd_metrics"]["timing_s/reward/write_debug"] = time.perf_counter() - section_start
        return reward_tensor

    def _should_defer_rcpc_interventions(self, skip_rcpc_credit: bool) -> bool:
        return (
            self.rcpc_enabled
            and self.rcpc_intervention_enabled
            and not skip_rcpc_credit
            and self.rcpc_intervention_mode == "prefix_regen"
            and self.rcpc_batch_counterfactual
        )

    def _run_deferred_rcpc_interventions(
        self,
        groups: Sequence[Sequence[Dict[str, Any]]],
        results: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        section_start = time.perf_counter()
        plans = []
        all_items = []
        build_errors = []
        empty_plan_count = 0
        eligible_group_count = 0
        for group, result in zip(groups, results):
            first = group[0]
            if not first.get("deferred_rcpc_intervention", False):
                continue
            if not self.rcpc_enabled or not result.get("ok", False):
                continue
            eligible_group_count += 1
            candidates = result.get("rcpc_candidates") or []
            if not candidates:
                continue
            try:
                intervention_items = self._build_rcpc_intervention_items(
                    first,
                    group,
                    result,
                    candidates,
                )
            except Exception as exc:
                error = "{}: {}".format(type(exc).__name__, exc)
                result["rcpc_error"] = error
                build_errors.append((str(result.get("uid", "unknown")), error))
                continue
            if not intervention_items:
                empty_plan_count += 1
                continue
            plans.append(
                {
                    "first": first,
                    "group": group,
                    "result": result,
                    "intervention_items": intervention_items,
                }
            )
            all_items.extend(intervention_items)
        for global_item_index, item in enumerate(all_items):
            # Keep counterfactual seeds identical whether generation runs as
            # one batch or through the generation/judge pipeline.
            item["_rcpc_seed_parent_index"] = global_item_index
        metrics["timing_s/rcpc/build_intervention_plan_wall"] = time.perf_counter() - section_start
        metrics["rcpc/intervention_plan_count"] = float(len(plans))
        metrics["rcpc/intervention_plan_build_error_count"] = float(len(build_errors))
        metrics["rcpc/intervention_plan_empty_count"] = float(empty_plan_count)
        metrics["rcpc/intervention_item_count"] = float(len(all_items))
        arm_count = 2 if self.rcpc_intervention_mode == "prefix_regen" else 1
        requested_samples = len(all_items) * self.rcpc_counterfactual_samples * arm_count
        metrics["rcpc/counterfactual_requested_samples"] = float(requested_samples)
        requested_new_tokens = []
        for item in all_items:
            if self.rcpc_intervention_mode == "prefix_regen":
                requested_new_tokens.extend(
                    [
                        float(item.get("factual_max_new_tokens", 0) or 0.0),
                        float(item.get("control_max_new_tokens", 0) or 0.0),
                    ]
                )
            else:
                requested_new_tokens.append(float(item.get("max_new_tokens", 0) or 0.0))
        metrics["rcpc/counterfactual_max_new_tokens_mean"] = (
            sum(requested_new_tokens) / len(requested_new_tokens)
            if requested_new_tokens
            else 0.0
        )
        metrics["rcpc/counterfactual_max_new_tokens_max"] = (
            max(requested_new_tokens) if requested_new_tokens else 0.0
        )
        if self.rcpc_counterfactual_batch_size > 0 and requested_samples > 0:
            if (
                self.rcpc_overlap_generation_and_judge
                and self._counterfactual_session_factory is not None
                and len(plans) > 1
            ):
                chunk_count = sum(
                    (
                        len(plan["intervention_items"])
                        * self.rcpc_counterfactual_samples
                        * arm_count
                        + self.rcpc_counterfactual_batch_size
                        - 1
                    )
                    // self.rcpc_counterfactual_batch_size
                    for plan in plans
                )
            else:
                chunk_count = (
                    requested_samples + self.rcpc_counterfactual_batch_size - 1
                ) // self.rcpc_counterfactual_batch_size
            metrics["rcpc/counterfactual_generation_chunks"] = float(chunk_count)
        else:
            metrics["rcpc/counterfactual_generation_chunks"] = 1.0 if requested_samples else 0.0

        if not plans:
            metrics["rcpc/intervention_plan_success_count"] = 0.0
            metrics["rcpc/intervention_plan_error_count"] = float(len(build_errors))
            metrics["rcpc/intervention_plan_success_ratio"] = 0.0
            metrics["rcpc/effective_group_ratio"] = 0.0
            if build_errors:
                self._handle_rcpc_plan_errors(build_errors)
            return metrics

        def finish_plan(plan: Mapping[str, Any]) -> Tuple[str, Optional[str]]:
            group = plan["group"]
            result = plan["result"]
            try:
                section_start = time.perf_counter()
                interventions = self._score_rcpc_intervention_items(
                    plan["first"],
                    group,
                    result,
                    plan["intervention_items"],
                )
                result["rcpc_interventions"] = interventions
                token_advantages, metrics = self._build_rcpc_token_advantages(
                    group,
                    result,
                    result.get("rcpc_candidates") or [],
                    interventions,
                )
                verifier_metrics = result.pop("rcpc_verifier_metrics", {})
                request_count = float(verifier_metrics.get("request_count", 0.0))
                retry_count = float(verifier_metrics.get("retry_count", 0.0))
                pair_unit_count = float(verifier_metrics.get("pair_unit_count", 0.0))
                leaf_failure_count = float(verifier_metrics.get("leaf_failure_count", 0.0))
                metrics["rcpc/verifier_retry_ratio"] = (
                    retry_count / request_count if request_count > 0.0 else 0.0
                )
                metrics["rcpc/verifier_leaf_failure_ratio"] = (
                    leaf_failure_count / pair_unit_count if pair_unit_count > 0.0 else 0.0
                )
                metrics["rcpc/counterfactual_batched"] = 1.0
                metrics["rcpc/counterfactual_items"] = float(len(plan["intervention_items"]))
                metrics["rcpc/counterfactual_samples"] = float(self.rcpc_counterfactual_samples)
                metrics["rcpc/counterfactual_batch_size"] = float(self.rcpc_counterfactual_batch_size)
                metrics["timing_s/rcpc/intervention_score_group"] = time.perf_counter() - section_start
                result["rcpc_token_advantages"] = token_advantages
                result["rcpc_metrics"] = metrics
                if interventions:
                    return "success", None
                error = "intervention scoring returned no block effects"
                result["rcpc_error"] = error
                return "empty", error
            except Exception as exc:
                error = "{}: {}".format(type(exc).__name__, exc)
                result["rcpc_error"] = error
                return "error", error

        pipeline_enabled = bool(
            self.rcpc_overlap_generation_and_judge
            and self._counterfactual_session_factory is not None
            and len(plans) > 1
        )
        try:
            if pipeline_enabled:
                pipeline_start = time.perf_counter()
                generation_elapsed = 0.0
                score_start: Optional[float] = None
                plan_executor = ThreadPoolExecutor(
                    max_workers=min(self.max_concurrency, len(plans)),
                    thread_name_prefix="rcpc-plan",
                )
                futures = []
                try:
                    with self._counterfactual_session_factory():
                        for plan in plans:
                            generation_start = time.perf_counter()
                            self._populate_counterfactual_texts(
                                plan["intervention_items"]
                            )
                            generation_elapsed += time.perf_counter() - generation_start
                            if score_start is None:
                                score_start = time.perf_counter()
                            futures.append(plan_executor.submit(finish_plan, plan))
                    outcomes = [future.result() for future in futures]
                finally:
                    plan_executor.shutdown(wait=True)
                pipeline_end = time.perf_counter()
                metrics["timing_s/rcpc/counterfactual_generation_wall"] = generation_elapsed
                metrics["timing_s/rcpc/intervention_score_wall"] = (
                    pipeline_end - score_start if score_start is not None else 0.0
                )
                metrics["timing_s/rcpc/pipeline_wall"] = pipeline_end - pipeline_start
            else:
                generation_start = time.perf_counter()
                self._populate_counterfactual_texts(all_items)
                metrics["timing_s/rcpc/counterfactual_generation_wall"] = (
                    time.perf_counter() - generation_start
                )
                score_start = time.perf_counter()
                if self.max_concurrency <= 1 or len(plans) <= 1:
                    outcomes = [finish_plan(plan) for plan in plans]
                else:
                    with ThreadPoolExecutor(
                        max_workers=min(self.max_concurrency, len(plans)),
                        thread_name_prefix="rcpc-plan",
                    ) as executor:
                        outcomes = list(executor.map(finish_plan, plans))
                metrics["timing_s/rcpc/intervention_score_wall"] = (
                    time.perf_counter() - score_start
                )

            semantic_pair_flags = []
            deleted_content_reappearance_flags = []
            for item in all_items:
                factual_validity = list(item.get("factual_semantic_valid") or [])
                control_validity = list(item.get("control_semantic_valid") or [])
                semantic_pair_flags.extend(
                    bool(factual_validity[index]) and bool(control_validity[index])
                    for index in range(min(len(factual_validity), len(control_validity)))
                )
                reappearance_flags = list(
                    item.get("control_deleted_content_reappeared") or []
                )
                deleted_content_reappearance_flags.extend(
                    1.0 if bool(reappearance_flags[index]) else 0.0
                    for index in range(min(len(reappearance_flags), len(control_validity)))
                    if bool(control_validity[index])
                )
            metrics["rcpc/pair_semantic_valid_ratio"] = (
                sum(semantic_pair_flags) / len(semantic_pair_flags)
                if semantic_pair_flags
                else 0.0
            )
            metrics["rcpc/deleted_content_reappearance_rate"] = (
                sum(deleted_content_reappearance_flags)
                / len(deleted_content_reappearance_flags)
                if deleted_content_reappearance_flags
                else 0.0
            )
        except Exception as exc:
            error = "{}: {}".format(type(exc).__name__, exc)
            generation_errors = []
            for plan in plans:
                plan["result"]["rcpc_error"] = error
                generation_errors.append((str(plan["result"].get("uid", "unknown")), error))
            all_errors = build_errors + generation_errors
            metrics["rcpc/intervention_plan_success_count"] = 0.0
            metrics["rcpc/intervention_plan_error_count"] = float(len(all_errors))
            metrics["rcpc/intervention_plan_success_ratio"] = 0.0
            metrics["rcpc/effective_group_ratio"] = 0.0
            self._handle_rcpc_plan_errors(all_errors)
            return metrics

        success_count = sum(1 for status, _error in outcomes if status == "success")
        scored_empty_count = sum(1 for status, _error in outcomes if status == "empty")
        score_errors = [
            (str(plan["result"].get("uid", "unknown")), str(error))
            for plan, (status, error) in zip(plans, outcomes)
            if status in {"error", "empty"} and error
        ]
        all_errors = build_errors + score_errors
        metrics["rcpc/intervention_plan_success_count"] = float(success_count)
        metrics["rcpc/intervention_plan_scored_empty_count"] = float(scored_empty_count)
        metrics["rcpc/intervention_plan_error_count"] = float(len(all_errors))
        attempted_plan_count = len(plans) + len(build_errors)
        metrics["rcpc/intervention_plan_success_ratio"] = (
            success_count / attempted_plan_count if attempted_plan_count else 0.0
        )
        metrics["rcpc/effective_group_ratio"] = (
            success_count / eligible_group_count if eligible_group_count else 0.0
        )
        if all_errors:
            self._handle_rcpc_plan_errors(all_errors)
        return metrics

    def _handle_rcpc_plan_errors(self, errors: Sequence[Tuple[str, str]]) -> None:
        if not errors:
            return
        error_counts: Dict[str, int] = defaultdict(int)
        for _uid, error in errors:
            error_counts[str(error)] += 1
        summary = {
            "error_count": len(errors),
            "error_types": [
                {"count": count, "error": error}
                for error, count in sorted(error_counts.items(), key=lambda item: (-item[1], item[0]))
            ],
            "examples": [
                {"uid": uid, "error": error}
                for uid, error in list(errors)[:5]
            ],
        }
        with self._print_lock:
            print("[ropd rcpc plan errors]", json.dumps(summary, ensure_ascii=False))
        if self.rcpc_fail_on_intervention_error:
            raise RuntimeError(
                "RCPC intervention failed for {} plan(s); first error: {}".format(
                    len(errors), errors[0][1]
                )
            )

    @staticmethod
    def _filter_logged_metrics(metrics: Mapping[str, float]) -> Dict[str, float]:
        filtered: Dict[str, float] = {}
        static_or_redundant = {
            "reward/zero_score_on_format_error",
            "reward/zero_criteria_on_format_error",
            "reward/max_concurrency",
            "reward/group_count",
            "reward/group_size/mean",
        }
        for key, value in metrics.items():
            key = str(key)
            if key in static_or_redundant:
                continue
            if key.startswith("rcpc/") and key not in _RCPC_LOG_METRICS:
                continue
            if key.startswith("timing_s/rcpc/") and key not in _RCPC_LOG_TIMINGS:
                continue
            if key.startswith("timing_count/"):
                continue
            filtered[key] = float(value)
        return filtered

    def _collect_criterion_metrics(self, results: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        combined_advantages = []
        format_valid_values = []
        clipped_values = []
        raw_scores = []
        final_scores = []
        ok_values = []
        for result in results:
            ok_values.append(1.0 if result.get("ok", False) else 0.0)
            format_valid_values.extend(
                1.0 if value else 0.0 for value in result.get("student_format_valid", [])
            )
            clipped_values.extend(
                1.0 if value else 0.0 for value in result.get("student_response_clipped", [])
            )
            raw_scores.extend(float(value) for value in result.get("student_scores", []))
            scores = result.get("scores", {})
            if isinstance(scores, Mapping):
                final_scores.extend(float(value) for value in scores.values())
            combined_advantages.extend(float(value) for value in result.get("criterion_advantages", {}).values())

        metrics = {
            "criterion_advantage/mean": (
                sum(combined_advantages) / len(combined_advantages) if combined_advantages else 0.0
            ),
            "criterion_advantage/std": _population_std(combined_advantages),
        }
        metrics.update(
            {
                "reward/group_ok_ratio": sum(ok_values) / len(ok_values) if ok_values else 0.0,
                "reward/format_valid_ratio": (
                    sum(format_valid_values) / len(format_valid_values) if format_valid_values else 0.0
                ),
                "reward/zero_score_on_format_error": 1.0 if self.zero_score_on_format_error else 0.0,
                "reward/zero_criteria_on_format_error": 1.0 if self.zero_criteria_on_format_error else 0.0,
                "reward/clipped_ratio": (
                    sum(clipped_values) / len(clipped_values) if clipped_values else 0.0
                ),
                "reward/verifier_raw/mean": sum(raw_scores) / len(raw_scores) if raw_scores else 0.0,
                "reward/verifier_raw/max": max(raw_scores) if raw_scores else 0.0,
                "reward/verifier_raw/min": min(raw_scores) if raw_scores else 0.0,
                "reward/final_nonzero_ratio": (
                    sum(1.0 for value in final_scores if value != 0.0) / len(final_scores)
                    if final_scores
                    else 0.0
                ),
            }
        )
        rcpc_metric_values: Dict[str, List[float]] = defaultdict(list)
        timing_metric_values: Dict[str, List[float]] = defaultdict(list)
        timing_count_values: Dict[str, List[float]] = defaultdict(list)
        for result in results:
            for key, value in result.get("rcpc_metrics", {}).items():
                rcpc_metric_values[str(key)].append(float(value))
            for key, value in result.get("timing_metrics", {}).items():
                timing_metric_values[str(key)].append(float(value))
            for key, value in result.get("timing_counts", {}).items():
                timing_count_values[str(key)].append(float(value))
        for key, values in rcpc_metric_values.items():
            metrics[key] = sum(values) / len(values) if values else 0.0
        for key, values in timing_metric_values.items():
            if not values:
                continue
            metrics[f"{key}/sum"] = sum(values)
            metrics[f"{key}/mean"] = sum(values) / len(values)
            metrics[f"{key}/max"] = max(values)
        for key, values in timing_count_values.items():
            metrics[f"timing_count/{key}"] = sum(values)
        return metrics

    def _has_training_group_keys(self, data: DataProto) -> bool:
        return "uid" in data.non_tensor_batch and "is_onpolicy" in data.non_tensor_batch

    def _response_format_valid(self, info: Mapping[str, Any]) -> bool:
        if not _has_strict_cot_format(str(info.get("response_text", ""))):
            return False
        if bool(info.get("response_clipped", False)):
            return False
        return True

    def _collect_response_infos(self, data: DataProto) -> List[Dict[str, Any]]:
        responses = data.batch["responses"]
        token_entropies = data.batch.get("token_entropies", None)
        response_width = responses.shape[-1]
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_width:]
        valid_response_lengths = response_mask.sum(dim=-1)

        uids = _as_list(data.non_tensor_batch.get("uid"))
        is_onpolicy_values = _as_list(data.non_tensor_batch.get("is_onpolicy"))
        raw_prompts = _as_list(data.non_tensor_batch.get("raw_prompt"))
        raw_prompt_ids_values = _as_list(data.non_tensor_batch.get("raw_prompt_ids"))
        image_paths_values = _as_list(data.non_tensor_batch.get("image_paths"))
        answers = _as_list(data.non_tensor_batch.get("answer"))

        infos = []
        for batch_index in range(len(data)):
            response_length = int(valid_response_lengths[batch_index].item())
            valid_response_ids = responses[batch_index][:response_length].detach().cpu().tolist()
            response_text = self._decode_response_text(responses[batch_index], response_length)
            if token_entropies is not None and response_length > 0:
                response_token_entropies = (
                    token_entropies[batch_index][:response_length].detach().float().cpu()
                ).tolist()
            else:
                response_token_entropies = [0.0] * response_length
            raw_prompt = (
                raw_prompts[batch_index]
                if batch_index < len(raw_prompts)
                else self._decode_prompt_text(data, batch_index)
            )
            image_paths = image_paths_values[batch_index] if batch_index < len(image_paths_values) else []
            infos.append(
                {
                    "batch_index": batch_index,
                    "uid": str(uids[batch_index]) if batch_index < len(uids) else "sample-{}".format(batch_index),
                    "is_onpolicy": bool(is_onpolicy_values[batch_index]) if batch_index < len(is_onpolicy_values) else True,
                    "raw_prompt": str(raw_prompt),
                    "raw_prompt_ids": (
                        [int(token_id) for token_id in _as_list(raw_prompt_ids_values[batch_index])]
                        if batch_index < len(raw_prompt_ids_values)
                        else []
                    ),
                    "image_paths": [str(item) for item in _as_list(image_paths) if item],
                    "ground_truth": str(answers[batch_index]) if batch_index < len(answers) else "",
                    "response_text": response_text,
                    "response_token_ids": valid_response_ids,
                    "response_token_offsets": build_token_offsets(
                        self.tokenizer,
                        valid_response_ids,
                        decoded_text=response_text,
                    ),
                    "response_token_entropies": response_token_entropies,
                    "response_length": response_length,
                    "response_limit": int(response_width),
                    "response_clipped": bool(response_length >= int(response_width)),
                    "global_step": int(getattr(self, "_current_global_step", -1)),
                }
            )
        return infos

    def _decode_response_text(self, response_ids: torch.Tensor, valid_response_length: int) -> str:
        if valid_response_length <= 0:
            return ""
        valid_response_ids = response_ids[:valid_response_length]
        return self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    def _decode_prompt_text(self, data: DataProto, batch_index: int) -> str:
        prompt_ids = data.batch["prompts"][batch_index].clone()
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = int(data.batch["attention_mask"][batch_index][:prompt_length].sum().item())
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        valid_prompt_ids[valid_prompt_ids < 0] = 0
        return self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

    def _fill_rule_rewards(self, reward_tensor: torch.Tensor, response_infos: Sequence[Dict[str, Any]]) -> None:
        for info in response_infos:
            if info["response_length"] <= 0:
                continue
            reward_tensor[info["batch_index"], info["response_length"] - 1] = self._rule_score(info)

    def _group_onpolicy_infos(self, response_infos: Sequence[Dict[str, Any]]) -> "OrderedDict[str, List[Dict[str, Any]]]":
        grouped = OrderedDict()
        for info in response_infos:
            if not info["is_onpolicy"]:
                continue
            grouped.setdefault(info["uid"], []).append(info)
        return grouped

    def _score_group(self, group: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        group_start = time.perf_counter()
        timing_metrics: Dict[str, float] = {}
        timing_counts: Dict[str, float] = {}
        first = group[0]
        raw_teacher_answers = []
        teacher_answers = []
        rubric = None
        try:
            section_start = time.perf_counter()
            raw_teacher_answers = self._generate_teacher_answers(first)
            timing_metrics["timing_s/reward/teacher_group"] = time.perf_counter() - section_start
            timing_counts["teacher_requests"] = float(self.teacher_answer_count)
            teacher_answers = self._filter_teacher_answers(first, raw_teacher_answers)
            if not teacher_answers:
                raw_teacher_labels = [
                    self._extract_final_answer_label(answer) or "UNPARSED"
                    for answer in raw_teacher_answers
                ]
                raise ValueError(
                    "all teacher answers were filtered out by known final label; "
                    "ground_truth={}; raw_teacher_labels={}".format(first["ground_truth"], raw_teacher_labels)
                )
            section_start = time.perf_counter()
            rubric = self._generate_rubric(first, teacher_answers, [item["response_text"] for item in group])
            timing_metrics["timing_s/reward/rubricator_group"] = time.perf_counter() - section_start
            timing_counts["rubricator_requests"] = 1.0
            answer_items = self._build_shuffled_answer_items(
                uid=first["uid"],
                teacher_answers=teacher_answers,
                student_answers=[item["response_text"] for item in group],
            )
            section_start = time.perf_counter()
            verifier_payload = self._verify_answers(first, rubric, [item["text"] for item in answer_items])
            timing_metrics["timing_s/reward/verifier_initial_group"] = time.perf_counter() - section_start
            timing_counts["verifier_initial_requests"] = 1.0
            timing_counts["verifier_initial_answers"] = float(len(answer_items))
            ordered_scores = [float(answer["final_score"]) for answer in verifier_payload["answers"]]
            student_scores = self._restore_student_scores(answer_items, ordered_scores, len(group))
            student_verifier_answers = self._restore_student_verifier_answers(
                answer_items,
                verifier_payload["answers"],
                len(group),
            )
            maximum_score = float(rubric["maximum_score"])
            normalized_scores = [max(0.0, min(1.0, score / maximum_score)) for score in student_scores]
            student_format_valid = [self._response_format_valid(info) for info in group]
            if self.zero_score_on_format_error:
                normalized_scores = [
                    score if student_format_valid[index] else 0.0
                    for index, score in enumerate(normalized_scores)
                ]
            criterion_advantage_values, criterion_stats = _compute_group_criterion_advantages(
                rubric,
                student_verifier_answers,
                student_format_valid,
                require_strict_cot_format=self.zero_criteria_on_format_error,
            )
            scores = {
                info["batch_index"]: normalized_scores[index]
                for index, info in enumerate(group)
            }
            criterion_advantages = {
                info["batch_index"]: criterion_advantage_values[index]
                for index, info in enumerate(group)
            }
            result = {
                "uid": first["uid"],
                "ok": True,
                "scores": scores,
                "criterion_advantages": criterion_advantages,
                "criterion_stats": criterion_stats,
                "student_scores": student_scores,
                "student_format_valid": student_format_valid,
                "student_response_clipped": [bool(info.get("response_clipped", False)) for info in group],
                "student_verifier_answers": student_verifier_answers,
                "student_answers": [item["response_text"] for item in group],
                "student_batch_indices": [item["batch_index"] for item in group],
                "raw_teacher_answers": raw_teacher_answers,
                "teacher_answers": teacher_answers,
                "rubric": rubric,
                "verifier_payload": verifier_payload,
                "shadow_attribution": None,
                "shadow_attribution_error": "",
                "rcpc_candidates": [],
                "rcpc_interventions": {},
                "rcpc_token_advantages": {},
                "rcpc_metrics": {},
                "timing_metrics": timing_metrics,
                "timing_counts": timing_counts,
                "error": "",
            }
            section_start = time.perf_counter()
            self._maybe_add_rcpc_credit(first, group, result)
            timing_metrics["timing_s/rcpc/credit_prepare_group"] = time.perf_counter() - section_start
            section_start = time.perf_counter()
            self._maybe_add_shadow_attribution(first, result)
            timing_metrics["timing_s/reward/shadow_attribution_group"] = time.perf_counter() - section_start
            timing_metrics["timing_s/reward/group_total"] = time.perf_counter() - group_start
            if not first.get("defer_group_print", False):
                self._maybe_print_group(first, result)
            return result
        except Exception as exc:
            timing_metrics["timing_s/reward/group_total"] = time.perf_counter() - group_start
            student_format_valid = [self._response_format_valid(info) for info in group]
            scores = {}
            for index, info in enumerate(group):
                score = self._rule_score(info) if self.fallback_to_ipr else 0.0
                if self.zero_score_on_format_error and not student_format_valid[index]:
                    score = 0.0
                scores[info["batch_index"]] = score
            fallback_advantages = _compute_outcome_group_advantages(
                [float(scores[info["batch_index"]]) for info in group]
            )
            result = {
                "uid": first["uid"],
                "ok": False,
                "scores": scores,
                "criterion_advantages": {
                    info["batch_index"]: fallback_advantages[index]
                    for index, info in enumerate(group)
                },
                "criterion_stats": {},
                "student_scores": [],
                "student_format_valid": student_format_valid,
                "student_response_clipped": [bool(info.get("response_clipped", False)) for info in group],
                "student_verifier_answers": [],
                "student_answers": [info["response_text"] for info in group],
                "student_batch_indices": [info["batch_index"] for info in group],
                "raw_teacher_answers": raw_teacher_answers,
                "teacher_answers": teacher_answers,
                "rubric": rubric,
                "shadow_attribution": None,
                "shadow_attribution_error": "",
                "rcpc_candidates": [],
                "rcpc_interventions": {},
                "rcpc_token_advantages": {},
                "rcpc_metrics": {},
                "timing_metrics": timing_metrics,
                "timing_counts": timing_counts,
                "error": "{}: {}".format(type(exc).__name__, exc),
            }
            if not first.get("defer_group_print", False):
                self._maybe_print_group(first, result)
            return result

    def _maybe_add_rcpc_credit(
        self,
        first: Mapping[str, Any],
        group: Sequence[Dict[str, Any]],
        result: Dict[str, Any],
    ) -> None:
        if first.get("skip_rcpc_credit", False):
            return
        if not self.rcpc_enabled or not result.get("ok", False):
            return
        try:
            candidates = self._build_rcpc_candidates_for_group(group)
            result["rcpc_candidates"] = candidates
            interventions = {}
            if self.rcpc_intervention_enabled and first.get("run_rcpc_intervention", False):
                interventions = self._run_rcpc_interventions(first, group, result, candidates)
            result["rcpc_interventions"] = interventions
            token_advantages, metrics = self._build_rcpc_token_advantages(group, result, candidates, interventions)
            verifier_metrics = result.pop("rcpc_verifier_metrics", {})
            request_count = float(verifier_metrics.get("request_count", 0.0))
            retry_count = float(verifier_metrics.get("retry_count", 0.0))
            pair_unit_count = float(verifier_metrics.get("pair_unit_count", 0.0))
            leaf_failure_count = float(verifier_metrics.get("leaf_failure_count", 0.0))
            metrics["rcpc/verifier_retry_ratio"] = (
                retry_count / request_count if request_count > 0.0 else 0.0
            )
            metrics["rcpc/verifier_leaf_failure_ratio"] = (
                leaf_failure_count / pair_unit_count if pair_unit_count > 0.0 else 0.0
            )
            result["rcpc_token_advantages"] = token_advantages
            result["rcpc_metrics"] = metrics
        except Exception as exc:
            result["rcpc_error"] = "{}: {}".format(type(exc).__name__, exc)

    def _build_rcpc_candidates_for_group(
        self,
        group: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self.rcpc_derive_candidates_from_budget:
            return build_group_candidates(
                group,
                top_actions=self.rcpc_top_actions,
                top_blocks=self.rcpc_top_blocks,
                min_action_chars=self.rcpc_min_action_chars,
                max_action_chars=self.rcpc_max_action_chars,
                max_action_tokens=self.rcpc_max_action_tokens,
                min_robust_denom=self.rcpc_min_robust_denom,
                min_anchor_z=self.rcpc_min_anchor_z,
            )
        return [self._build_rcpc_candidates_for_response(info) for info in group]

    def _build_rcpc_candidates_for_response(self, info: Mapping[str, Any]) -> Dict[str, Any]:
        token_entropies = list(info.get("response_token_entropies") or [])
        response_length = int(info.get("response_length", 0))
        if len(token_entropies) < response_length:
            token_entropies.extend([0.0] * (response_length - len(token_entropies)))
        return build_candidates(
            str(info.get("response_text", "")),
            info.get("response_token_offsets") or [],
            token_entropies,
            top_actions=self.rcpc_top_actions,
            top_blocks=self.rcpc_top_blocks,
            min_action_chars=self.rcpc_min_action_chars,
            max_action_chars=self.rcpc_max_action_chars,
            max_action_tokens=self.rcpc_max_action_tokens,
            min_robust_denom=self.rcpc_min_robust_denom,
            min_anchor_z=self.rcpc_min_anchor_z,
        )

    def _run_rcpc_interventions(
        self,
        first: Mapping[str, Any],
        group: Sequence[Dict[str, Any]],
        result: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
    ) -> Dict[int, Dict[int, Dict[str, Any]]]:
        intervention_items = self._build_rcpc_intervention_items(first, group, result, candidates)
        if not intervention_items:
            return {}
        if self.rcpc_intervention_mode == "prefix_regen":
            self._populate_counterfactual_texts(intervention_items)
        return self._score_rcpc_intervention_items(first, group, result, intervention_items)

    def _build_rcpc_intervention_items(
        self,
        first: Mapping[str, Any],
        group: Sequence[Dict[str, Any]],
        result: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        intervention_items = []
        criterion_points = self._criterion_points_for_result(result)
        for response_index, (info, candidate) in enumerate(zip(group, candidates)):
            if bool(info.get("response_clipped", False)):
                continue
            if not _SUBSTANTIVE_RE.search(str(info.get("response_text", ""))):
                continue
            criterion_advantages = self._criterion_advantages_for_response(result, response_index)
            point_total = sum(max(0.0, value) for value in criterion_points.values())
            response_advantage_scale = 0.0
            for criterion_id, points in criterion_points.items():
                if points <= 0.0:
                    continue
                weight = points / point_total if point_total > 0.0 else 1.0
                response_advantage_scale += weight * abs(float(criterion_advantages.get(criterion_id, 0.0)))
            ranked_blocks = []
            for block in candidate.get("candidate_blocks", []):
                salience = max(
                    0.0,
                    float(block.get("block_max_robust_z", block.get("anchor_robust_z", 0.0))),
                )
                priority = response_advantage_scale * salience
                if priority <= 0.0:
                    continue
                ranked_blocks.append(
                    (
                        priority,
                        {
                            **block,
                            "selection_priority": priority,
                        },
                    )
                )
            ranked_blocks.sort(key=lambda item: item[0], reverse=True)
            if self.rcpc_intervention_max_blocks_per_answer > 0:
                ranked_blocks = ranked_blocks[: self.rcpc_intervention_max_blocks_per_answer]
            blocks = [block for _, block in ranked_blocks]
            for block in blocks:
                item = {
                    "response_index": response_index,
                    "batch_index": info["batch_index"],
                    "block_index": int(block["block_index"]),
                    "block": block,
                }
                if self.rcpc_intervention_mode == "prefix_regen":
                    response_token_ids = list(info.get("response_token_ids", []))
                    control_prefix_end = max(0, int(block.get("token_start", 0)))
                    factual_prefix_end = min(
                        len(response_token_ids),
                        max(control_prefix_end, int(block.get("token_end", -1)) + 1),
                    )
                    control_prefix_ids = response_token_ids[:control_prefix_end]
                    factual_prefix_ids = response_token_ids[:factual_prefix_end]
                    if not info.get("raw_prompt_ids"):
                        raise RuntimeError("prefix_regen requires raw_prompt_ids in the rollout batch")
                    response_limit = max(
                        1,
                        int(info.get("response_limit", info.get("response_length", 0))),
                    )
                    control_max_new_tokens = max(1, response_limit - len(control_prefix_ids))
                    factual_max_new_tokens = max(1, response_limit - len(factual_prefix_ids))
                    item.update(
                        {
                            "raw_prompt_ids": list(info["raw_prompt_ids"]),
                            "control_prefix_response_token_ids": control_prefix_ids,
                            "factual_prefix_response_token_ids": factual_prefix_ids,
                            "control_max_new_tokens": control_max_new_tokens,
                            "factual_max_new_tokens": factual_max_new_tokens,
                        }
                    )
                else:
                    item["text"] = apply_intervention(
                        str(info["response_text"]),
                        block,
                        mode=self.rcpc_intervention_mode,
                    )
                intervention_items.append(item)
        intervention_items.sort(key=lambda item: float(item["block"].get("selection_priority", 0.0)), reverse=True)
        if self.rcpc_intervention_max_blocks_per_group > 0:
            intervention_items = intervention_items[: self.rcpc_intervention_max_blocks_per_group]
        return intervention_items

    def _counterfactual_sample_quality(
        self,
        text: str,
        *,
        prefix_token_ids: Sequence[int],
        max_new_tokens: int,
        generated_token_count: Optional[int] = None,
        clipped: Optional[bool] = None,
    ) -> Dict[str, Any]:
        text = str(text or "")
        if generated_token_count is None:
            token_count = self._count_text_tokens(text)
            generated_token_count = max(0, token_count - len(prefix_token_ids))
        else:
            generated_token_count = max(0, int(generated_token_count))
        # vLLM does not currently expose finish_reason through this rollout
        # path. Reaching the request limit is therefore the safest observable
        # clipping signal; the one-token tolerance absorbs boundary retokenizing.
        if clipped is None:
            clipped = generated_token_count >= max(1, int(max_new_tokens) - 1)
        else:
            clipped = bool(clipped)
        semantic_valid = (
            generated_token_count > 0
            and bool(_SUBSTANTIVE_RE.search(text))
            and not clipped
        )
        return {
            "format_valid": _has_strict_cot_format(text),
            "semantic_valid": semantic_valid,
            "clipped": clipped,
            "generated_token_count": generated_token_count,
        }

    def _populate_counterfactual_texts(self, intervention_items: Sequence[Dict[str, Any]]) -> None:
        if self.rcpc_intervention_mode == "prefix_regen":
            if self._counterfactual_generator is None:
                raise RuntimeError("prefix_regen requires a counterfactual generator from the trainer")
            samples = int(self.rcpc_counterfactual_samples)
            expanded_items = []
            for parent_index, item in enumerate(intervention_items):
                for sample_index in range(samples):
                    seed_parent_index = int(
                        item.get("_rcpc_seed_parent_index", parent_index)
                    )
                    pair_seed_offset = seed_parent_index * samples + sample_index
                    for arm in ("factual", "control"):
                        expanded_item = dict(item)
                        expanded_item["prefix_response_token_ids"] = list(
                            item[f"{arm}_prefix_response_token_ids"]
                        )
                        expanded_item["max_new_tokens"] = int(item[f"{arm}_max_new_tokens"])
                        expanded_item["_rcpc_parent_item_index"] = parent_index
                        expanded_item["_rcpc_sample_index"] = sample_index
                        expanded_item["_rcpc_arm"] = arm
                        expanded_item["_rcpc_pair_seed_offset"] = pair_seed_offset
                        expanded_items.append(expanded_item)
            with self._counterfactual_lock:
                # The trainer owns length bucketing and submits every chunk to
                # one worker-level sharding session. Calling it once here is
                # what prevents a full FSDP-to-vLLM weight sync per chunk.
                generated_answers = self._counterfactual_generator(expanded_items)
            if len(generated_answers) != len(expanded_items):
                raise RuntimeError(
                    "counterfactual generator returned {} answers for {} intervention items".format(
                        len(generated_answers), len(expanded_items)
                    )
                )
            samples_by_parent_arm: Dict[
                Tuple[int, str], List[Tuple[int, str, Dict[str, Any]]]
            ] = defaultdict(list)
            for expanded_item, generated_record in zip(expanded_items, generated_answers):
                if isinstance(generated_record, Mapping):
                    generated_text = str(generated_record.get("text", ""))
                    suffix_text = generated_record.get("suffix_text")
                    generated_metadata = {
                        "suffix_text": None if suffix_text is None else str(suffix_text),
                        "generated_token_count": int(
                            generated_record.get("generated_token_count", 0) or 0
                        ),
                        "clipped": bool(generated_record.get("clipped", False)),
                    }
                else:
                    generated_text = str(generated_record)
                    generated_metadata = {}
                key = (
                    int(expanded_item["_rcpc_parent_item_index"]),
                    str(expanded_item["_rcpc_arm"]),
                )
                samples_by_parent_arm[key].append(
                    (
                        int(expanded_item["_rcpc_sample_index"]),
                        generated_text,
                        generated_metadata,
                    )
                )
            for parent_index, item in enumerate(intervention_items):
                for arm in ("factual", "control"):
                    indexed_samples = sorted(samples_by_parent_arm.get((parent_index, arm), []))
                    texts = [text for _sample_index, text, _metadata in indexed_samples]
                    if len(texts) != samples:
                        raise RuntimeError(
                            "RCPC generator produced {} {} samples for item {}, expected {}".format(
                                len(texts), arm, parent_index, samples
                            )
                        )
                    item[f"{arm}_texts"] = texts
                    qualities = []
                    suffix_texts = []
                    prefix_text = self.tokenizer.decode(
                        item[f"{arm}_prefix_response_token_ids"],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for _sample_index, text, metadata in indexed_samples:
                        suffix_text = metadata.get("suffix_text") if metadata else None
                        if suffix_text is None:
                            suffix_text = text[len(prefix_text) :] if text.startswith(prefix_text) else text
                        suffix_texts.append(str(suffix_text))
                        quality = self._counterfactual_sample_quality(
                            text,
                            prefix_token_ids=item[f"{arm}_prefix_response_token_ids"],
                            max_new_tokens=int(item[f"{arm}_max_new_tokens"]),
                            generated_token_count=(
                                metadata.get("generated_token_count") if metadata else None
                            ),
                            clipped=metadata.get("clipped") if metadata else None,
                        )
                        qualities.append(quality)
                    item[f"{arm}_suffix_texts"] = suffix_texts
                    item[f"{arm}_format_valid"] = [
                        bool(quality["format_valid"]) for quality in qualities
                    ]
                    item[f"{arm}_semantic_valid"] = [
                        bool(quality["semantic_valid"]) for quality in qualities
                    ]
                    item[f"{arm}_clipped"] = [
                        bool(quality["clipped"]) for quality in qualities
                    ]
                reappearance_measurements = [
                    measure_deleted_content_reappearance(
                        str(item.get("block", {}).get("text", "")),
                        suffix_text,
                    )
                    for suffix_text in item.get("control_suffix_texts", [])
                ]
                item["control_deleted_content_reappeared"] = [
                    bool(measurement["reappeared"])
                    for measurement in reappearance_measurements
                ]
                item["control_deleted_content_reappearance_scores"] = [
                    float(measurement["key_term_recall"])
                    for measurement in reappearance_measurements
                ]
                item["control_deleted_content_exact_match"] = [
                    bool(measurement["exact_phrase_match"])
                    for measurement in reappearance_measurements
                ]
                item["texts"] = list(item["control_texts"])
                item["text"] = item["control_texts"][0] if item["control_texts"] else ""
                item["factual_sample_count"] = len(item["factual_texts"])
                item["control_sample_count"] = len(item["control_texts"])
                item["counterfactual_sample_count"] = len(item["control_texts"])

    def _build_rcpc_verifier_flat_items(
        self,
        intervention_items: Sequence[Mapping[str, Any]],
    ) -> List[Tuple[int, str, int, str]]:
        """Flatten only semantically usable factual/control pairs.

        Strict XML validity is intentionally not a semantic filter. Empty or
        likely clipped continuations are excluded before they consume judge
        capacity; a pair is retained only when both potential outcomes are
        usable.
        """
        flat_items: List[Tuple[int, str, int, str]] = []
        for item_index, item in enumerate(intervention_items):
            factual_texts = item.get("factual_texts")
            control_texts = item.get("control_texts")
            if isinstance(factual_texts, list) and isinstance(control_texts, list):
                pair_count = min(len(factual_texts), len(control_texts))
                factual_validity = list(item.get("factual_semantic_valid") or [True] * pair_count)
                control_validity = list(item.get("control_semantic_valid") or [True] * pair_count)
                for sample_index in range(pair_count):
                    factual_valid = (
                        sample_index < len(factual_validity) and bool(factual_validity[sample_index])
                    )
                    control_valid = (
                        sample_index < len(control_validity) and bool(control_validity[sample_index])
                    )
                    if not (factual_valid and control_valid):
                        continue
                    flat_items.append(
                        (item_index, "factual", sample_index, str(factual_texts[sample_index]))
                    )
                    flat_items.append(
                        (item_index, "control", sample_index, str(control_texts[sample_index]))
                    )
                continue

            texts = item.get("texts")
            if not isinstance(texts, list) or not texts:
                texts = [item.get("text", "")]
            for sample_index, text in enumerate(texts):
                if not _SUBSTANTIVE_RE.search(str(text or "")):
                    continue
                flat_items.append((item_index, "control", sample_index, str(text)))
        return flat_items

    def _score_rcpc_intervention_items(
        self,
        first: Mapping[str, Any],
        group: Sequence[Dict[str, Any]],
        result: Mapping[str, Any],
        intervention_items: Sequence[Mapping[str, Any]],
    ) -> Dict[int, Dict[int, Dict[str, Any]]]:
        if not intervention_items:
            return {}

        flat_items = self._build_rcpc_verifier_flat_items(intervention_items)
        answers_by_flat_key, verifier_metrics = self._verify_rcpc_arm_entries(
            first,
            result["rubric"],
            flat_items,
        )
        if isinstance(result, dict):
            result["rcpc_verifier_metrics"] = verifier_metrics
        original_answers = list(result["student_verifier_answers"])
        criterion_ids = [str(item["criterion_id"]) for item in result["rubric"]["rubrics"]]
        answers_by_item_arm: Dict[Tuple[int, str], List[Tuple[int, Mapping[str, Any], str]]] = defaultdict(list)
        for item_index, arm, sample_index, text in flat_items:
            answer = answers_by_flat_key.get((int(item_index), str(arm), int(sample_index)))
            if answer is None:
                continue
            answers_by_item_arm[(int(item_index), arm)].append((sample_index, answer, text))
        restored: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        for item_index, item in enumerate(intervention_items):
            control_entries = sorted(answers_by_item_arm.get((item_index, "control"), []))
            response_index = int(item["response_index"])
            original = original_answers[response_index]
            factual_entries = sorted(answers_by_item_arm.get((item_index, "factual"), []))
            if not factual_entries:
                factual_entries = [
                    (sample_index, original, str(group[response_index]["response_text"]))
                    for sample_index in range(len(control_entries))
                ]
            factual_by_sample = {int(entry[0]): entry for entry in factual_entries}
            control_by_sample = {int(entry[0]): entry for entry in control_entries}
            paired_sample_ids = sorted(set(factual_by_sample) & set(control_by_sample))
            if not paired_sample_ids:
                continue
            factual_entries = [factual_by_sample[index] for index in paired_sample_ids]
            control_entries = [control_by_sample[index] for index in paired_sample_ids]
            factual_answers = [entry[1] for entry in factual_entries]
            control_answers = [entry[1] for entry in control_entries]
            factual_texts = [entry[2] for entry in factual_entries]
            control_texts = [entry[2] for entry in control_entries]
            factual_format_valid = [_has_strict_cot_format(text) for text in factual_texts]
            control_format_valid = [_has_strict_cot_format(text) for text in control_texts]
            pair_semantic_valid = [True] * len(factual_answers)
            source_pair_count = max(
                len(item.get("factual_texts") or []),
                len(item.get("control_texts") or []),
                len(factual_answers),
            )
            factual_semantic_all = list(
                item.get("factual_semantic_valid") or [True] * source_pair_count
            )
            control_semantic_all = list(
                item.get("control_semantic_valid") or [True] * source_pair_count
            )
            all_pair_semantic_valid = [
                bool(factual_semantic_all[index]) and bool(control_semantic_all[index])
                for index in range(min(len(factual_semantic_all), len(control_semantic_all)))
            ]
            criterion_effects = {}
            criterion_variances = {}
            criterion_standard_errors = {}
            criterion_paired_differences = {}
            for criterion_index, criterion_id in enumerate(criterion_ids):
                factual_values = [
                    1.0 if bool(answer["judgement"][criterion_index]) else 0.0
                    for answer in factual_answers
                ]
                control_values = [
                    1.0 if bool(answer["judgement"][criterion_index]) else 0.0
                    for answer in control_answers
                ]
                stats = paired_effect_statistics(
                    factual_values,
                    control_values,
                    pair_validity=pair_semantic_valid,
                    variance_prior=self.rcpc_effect_variance_prior,
                )
                criterion_effects[criterion_id] = float(stats["effect"])
                criterion_variances[criterion_id] = float(stats["estimator_variance"])
                criterion_standard_errors[criterion_id] = float(stats["standard_error"])
                criterion_paired_differences[criterion_id] = list(stats["paired_differences"])
            factual_scores = [float(answer["final_score"]) for answer in factual_answers]
            control_scores = [float(answer["final_score"]) for answer in control_answers]
            score_stats = paired_effect_statistics(
                factual_scores,
                control_scores,
                pair_validity=pair_semantic_valid,
                variance_prior=self.rcpc_effect_variance_prior,
            )
            valid_factual_scores = [
                value for value, valid in zip(factual_scores, pair_semantic_valid) if valid
            ]
            valid_control_scores = [
                value for value, valid in zip(control_scores, pair_semantic_valid) if valid
            ]
            mean_factual_score = (
                sum(valid_factual_scores) / len(valid_factual_scores)
                if valid_factual_scores
                else 0.0
            )
            mean_control_score = (
                sum(valid_control_scores) / len(valid_control_scores)
                if valid_control_scores
                else 0.0
            )
            restored[item["batch_index"]][int(item["block_index"])] = {
                "response_index": response_index,
                "batch_index": int(item["batch_index"]),
                "block_index": int(item["block_index"]),
                "block": item["block"],
                "factual_text": factual_texts[0] if factual_texts else "",
                "factual_texts": factual_texts,
                "control_text": control_texts[0] if control_texts else "",
                "control_texts": control_texts,
                "control_suffix_texts": list(item.get("control_suffix_texts") or []),
                "control_deleted_content_reappeared": list(
                    item.get("control_deleted_content_reappeared") or []
                ),
                "control_deleted_content_reappearance_scores": list(
                    item.get("control_deleted_content_reappearance_scores") or []
                ),
                "control_deleted_content_exact_match": list(
                    item.get("control_deleted_content_exact_match") or []
                ),
                "intervened_text": control_texts[0] if control_texts else "",
                "intervened_texts": control_texts,
                "factual_sample_count": len(factual_answers),
                "control_sample_count": len(control_answers),
                "counterfactual_sample_count": len(control_answers),
                "factual_format_valid": factual_format_valid,
                "control_format_valid": control_format_valid,
                "pair_format_valid": [
                    factual_valid and control_valid
                    for factual_valid, control_valid in zip(
                        factual_format_valid,
                        control_format_valid,
                    )
                ],
                "pair_semantic_valid": all_pair_semantic_valid,
                "valid_pair_count": int(score_stats["valid_pair_count"]),
                "invalid_pair_count": int(score_stats["invalid_pair_count"]),
                "criterion_effects": criterion_effects,
                "criterion_variances": criterion_variances,
                "criterion_standard_errors": criterion_standard_errors,
                "criterion_paired_differences": criterion_paired_differences,
                "original_score": float(original["final_score"]),
                "factual_score": mean_factual_score,
                "factual_scores": factual_scores,
                "control_score": mean_control_score,
                "control_scores": control_scores,
                "intervened_score": mean_control_score,
                "intervened_scores": control_scores,
                "score_effect": float(score_stats["effect"]),
                "score_effect_variance": float(score_stats["estimator_variance"]),
                "score_effect_standard_error": float(score_stats["standard_error"]),
            }
        return {int(batch_index): dict(value) for batch_index, value in restored.items()}

    def _verify_rcpc_arm_entries(
        self,
        first: Mapping[str, Any],
        rubric: Mapping[str, Any],
        flat_items: Sequence[Tuple[int, str, int, str]],
    ) -> Tuple[
        Dict[Tuple[int, str, int], Mapping[str, Any]],
        Dict[str, float],
    ]:
        """Verify paired arms in token-aware batches with local failure isolation.

        Factual/control answers for one intervention sample stay in the same
        request, reducing call-level judge drift. Initial chunks from every
        group share one bounded executor; a failing chunk is recursively split
        inside its task so recovery cannot exceed the global concurrency cap.
        """
        answers_by_key: Dict[Tuple[int, str, int], Mapping[str, Any]] = {}
        diagnostics = {
            "request_count": 0.0,
            "retry_count": 0.0,
            "split_count": 0.0,
            "leaf_failure_count": 0.0,
            "pair_unit_count": 0.0,
        }
        if not flat_items:
            return answers_by_key, diagnostics

        # Keep each factual/control sample pair indivisible during packing and
        # recursive recovery. Single-arm ablations naturally form one-item units.
        units_by_key: "OrderedDict[Tuple[int, int], List[Tuple[int, str, int, str]]]" = OrderedDict()
        for item in flat_items:
            units_by_key.setdefault((int(item[0]), int(item[2])), []).append(item)
        units = list(units_by_key.values())
        diagnostics["pair_unit_count"] = float(len(units))

        def flattened(candidate_units):
            return [item for unit in candidate_units for item in unit]

        def fits(candidate_units) -> bool:
            items = flattened(candidate_units)
            if len(items) > self.rcpc_verifier_batch_size:
                return False
            criterion_count = len(rubric.get("rubrics", []))
            if (
                len(items) * self._rcpc_verifier_output_tokens_per_item(criterion_count)
                > self.verifier_max_output_tokens
            ):
                return False
            prompt = self._render_verifier_prompt(
                first,
                rubric,
                [str(item[3]) for item in items],
            )
            return self._count_text_tokens(prompt) <= self.rcpc_verifier_max_input_tokens

        packed_units = []
        current = []
        for unit in units:
            if current and not fits(current + [unit]):
                packed_units.append(current)
                current = []
            current.append(unit)
        if current:
            packed_units.append(current)

        def verify_units(candidate_units, chunk_label: str):
            local_answers: Dict[Tuple[int, str, int], Mapping[str, Any]] = {}
            local_diagnostics = {
                "request_count": 0.0,
                "retry_count": 0.0,
                "split_count": 0.0,
                "leaf_failure_count": 0.0,
            }
            local_errors = []
            items = flattened(candidate_units)
            last_error: Optional[Exception] = None
            for attempt in range(self.rcpc_verifier_max_retries + 1):
                local_diagnostics["request_count"] += 1.0
                if attempt > 0:
                    local_diagnostics["retry_count"] += 1.0
                try:
                    payload = self._verify_answers(
                        dict(first),
                        dict(rubric),
                        [str(item[3]) for item in items],
                        max_output_tokens=self._rcpc_verifier_output_limit(
                            len(items), len(rubric.get("rubrics", []))
                        ),
                    )
                    answers = payload.get("answers", [])
                    if len(answers) != len(items):
                        raise RuntimeError(
                            "RCPC verifier returned {} answers for chunk size {}".format(
                                len(answers), len(items)
                            )
                        )
                    for item, answer in zip(items, answers):
                        local_answers[(int(item[0]), str(item[1]), int(item[2]))] = answer
                    return local_answers, local_diagnostics, local_errors
                except Exception as exc:
                    last_error = exc
                    # A valid HTTP response with the wrong JSON shape will not
                    # improve by replaying the identical large request. Split
                    # immediately; reserve retries for transient transport or
                    # service failures.
                    if _is_structural_verifier_error(exc):
                        break

            if len(candidate_units) > 1:
                local_diagnostics["split_count"] += 1.0
                midpoint = max(1, len(candidate_units) // 2)
                for child_units, child_label in (
                    (candidate_units[:midpoint], chunk_label + "L"),
                    (candidate_units[midpoint:], chunk_label + "R"),
                ):
                    child_answers, child_diagnostics, child_errors = verify_units(
                        child_units, child_label
                    )
                    local_answers.update(child_answers)
                    for key, value in child_diagnostics.items():
                        local_diagnostics[key] += float(value)
                    local_errors.extend(child_errors)
                return local_answers, local_diagnostics, local_errors

            local_diagnostics["leaf_failure_count"] += 1.0
            local_errors.append(
                "unit={} answers={} error={}: {}".format(
                    chunk_label,
                    len(items),
                    type(last_error).__name__ if last_error is not None else "UnknownError",
                    last_error,
                )
            )
            return local_answers, local_diagnostics, local_errors

        # All groups share this executor, so one slow group cannot monopolize a
        # dedicated thread while other ready verifier chunks wait behind it.
        futures = [
            self._rcpc_verifier_executor.submit(
                verify_units, candidate_units, str(chunk_index)
            )
            for chunk_index, candidate_units in enumerate(packed_units)
        ]
        leaf_errors = []
        for future in futures:
            local_answers, local_diagnostics, local_errors = future.result()
            answers_by_key.update(local_answers)
            for key, value in local_diagnostics.items():
                diagnostics[key] += float(value)
            leaf_errors.extend(local_errors)

        if leaf_errors:
            with self._print_lock:
                print(
                    "[ropd rcpc verifier dropped pairs]",
                    json.dumps(
                        {
                            "count": len(leaf_errors),
                            "examples": leaf_errors[:3],
                        },
                        ensure_ascii=False,
                    ),
                )
        return answers_by_key, diagnostics

    def _criterion_advantages_for_response(
        self,
        result: Mapping[str, Any],
        response_index: int,
    ) -> Dict[str, float]:
        output = {}
        for criterion_id, stats in result.get("criterion_stats", {}).items():
            advantages = stats.get("advantages", [])
            if response_index < len(advantages):
                output[str(criterion_id)] = float(advantages[response_index])
        return output

    def _criterion_points_for_result(self, result: Mapping[str, Any]) -> Dict[str, float]:
        points = {
            str(item["criterion_id"]): float(item["points"])
            for item in result["rubric"]["rubrics"]
        }
        # Fixed-rubric training adds protocol compliance locally rather than
        # asking the LLM judge to evaluate it. Preserve that criterion in CAT.
        for criterion_id, stats in result.get("criterion_stats", {}).items():
            if str(criterion_id) not in points and "points" in stats:
                points[str(criterion_id)] = max(0.0, float(stats["points"]))
        return points

    def _build_rcpc_token_advantages(
        self,
        group: Sequence[Dict[str, Any]],
        result: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        interventions: Mapping[int, Mapping[int, Mapping[str, Any]]],
    ) -> Tuple[Dict[int, List[float]], Dict[str, float]]:
        token_advantages = {}
        metric_values: Dict[str, List[float]] = defaultdict(list)
        successful_metric_values: Dict[str, List[float]] = defaultdict(list)
        successful_response_count = 0
        criterion_points = self._criterion_points_for_result(result)
        for response_index, (info, candidate) in enumerate(zip(group, candidates)):
            batch_index = int(info["batch_index"])
            values, metrics = build_token_advantages(
                response_length=int(info["response_length"]),
                blocks=candidate.get("candidate_blocks", []),
                combined_advantage=float(result["criterion_advantages"].get(batch_index, 0.0)),
                criterion_advantages=self._criterion_advantages_for_response(result, response_index),
                criterion_points=criterion_points,
                intervention_effects=interventions.get(batch_index),
                fallback_to_full_response=self.rcpc_fallback_to_criterion_advantage,
                transport_lambda=self.rcpc_transport_lambda,
            )
            token_advantages[batch_index] = values
            for key, value in metrics.items():
                metric_values[key].append(float(value))
            response_interventions = interventions.get(batch_index, {}) or {}
            has_successful_intervention = any(
                int(payload.get("valid_pair_count", 0)) > 0
                for payload in response_interventions.values()
            )
            if has_successful_intervention:
                successful_response_count += 1
                for key in (
                    "rcpc/advantage_delta_l1_ratio",
                    "rcpc/advantage_cosine_to_baseline",
                    "rcpc/transport_kl",
                ):
                    successful_metric_values[key].append(float(metrics.get(key, 0.0)))
        metrics = {
            key: (sum(values) / len(values) if values else 0.0)
            for key, values in metric_values.items()
        }
        metrics["rcpc/successful_response_ratio"] = (
            successful_response_count / len(group) if group else 0.0
        )
        metrics["rcpc/successful_advantage_delta_l1_ratio"] = (
            sum(successful_metric_values["rcpc/advantage_delta_l1_ratio"])
            / len(successful_metric_values["rcpc/advantage_delta_l1_ratio"])
            if successful_metric_values["rcpc/advantage_delta_l1_ratio"]
            else 0.0
        )
        metrics["rcpc/successful_advantage_cosine_to_baseline"] = (
            sum(successful_metric_values["rcpc/advantage_cosine_to_baseline"])
            / len(successful_metric_values["rcpc/advantage_cosine_to_baseline"])
            if successful_metric_values["rcpc/advantage_cosine_to_baseline"]
            else 1.0
        )
        metrics["rcpc/successful_transport_kl"] = (
            sum(successful_metric_values["rcpc/transport_kl"])
            / len(successful_metric_values["rcpc/transport_kl"])
            if successful_metric_values["rcpc/transport_kl"]
            else 0.0
        )
        intervention_payloads = [
            payload
            for by_block in interventions.values()
            for payload in by_block.values()
        ]
        factual_format_flags = [
            bool(valid)
            for payload in intervention_payloads
            for valid in payload.get("factual_format_valid", [])
        ]
        control_format_flags = [
            bool(valid)
            for payload in intervention_payloads
            for valid in payload.get("control_format_valid", [])
        ]
        pair_format_flags = [
            bool(valid)
            for payload in intervention_payloads
            for valid in payload.get("pair_format_valid", [])
        ]
        pair_semantic_flags = [
            bool(valid)
            for payload in intervention_payloads
            for valid in payload.get("pair_semantic_valid", [])
        ]
        metrics["rcpc/factual_format_valid_ratio"] = (
            sum(factual_format_flags) / len(factual_format_flags)
            if factual_format_flags
            else 0.0
        )
        metrics["rcpc/control_format_valid_ratio"] = (
            sum(control_format_flags) / len(control_format_flags)
            if control_format_flags
            else 0.0
        )
        metrics["rcpc/pair_format_valid_ratio"] = (
            sum(pair_format_flags) / len(pair_format_flags)
            if pair_format_flags
            else 0.0
        )
        metrics["rcpc/pair_semantic_valid_ratio"] = (
            sum(pair_semantic_flags) / len(pair_semantic_flags)
            if pair_semantic_flags
            else 0.0
        )
        metrics["rcpc/valid_counterfactual_pair_count"] = float(sum(pair_format_flags))
        metrics["rcpc/invalid_counterfactual_pair_count"] = float(
            len(pair_format_flags) - sum(pair_format_flags)
        )
        metrics["rcpc/causal_valid_block_ratio"] = (
            sum(
                1.0
                for payload in intervention_payloads
                if int(payload.get("valid_pair_count", 0)) > 0
            )
            / len(intervention_payloads)
            if intervention_payloads
            else 0.0
        )
        return token_advantages, metrics

    def _generate_teacher_answers(self, info: Dict[str, Any]) -> List[str]:
        if self.teacher_template is None:
            raise FileNotFoundError("Missing teacher.txt prompt for dynamic ROPD reward.")
        prompt = _render_template(
            self.teacher_template,
            {
                "question": info["raw_prompt"],
                "ground_truth": info["ground_truth"],
            },
        )
        answers = []
        for _ in range(self.teacher_answer_count):
            answers.append(
                self.client.create_text(
                    model=self.teacher_model,
                    text=prompt,
                    image_paths=info["image_paths"],
                    temperature=self.teacher_temperature,
                    max_output_tokens=self.teacher_max_output_tokens,
                    json_mode=False,
                )
            )
        return list(OrderedDict((answer, None) for answer in answers).keys())

    def _filter_teacher_answers(self, info: Mapping[str, Any], teacher_answers: Sequence[str]) -> List[str]:
        if not self.filter_teacher_by_answer:
            return list(teacher_answers)
        ground_truth = str(info.get("ground_truth", "")).strip().upper()
        if ground_truth not in {"YES", "NO"}:
            return list(teacher_answers)
        return [
            answer
            for answer in teacher_answers
            if self._extract_final_answer_label(answer) == ground_truth
        ]

    def _extract_final_answer_label(self, answer: str) -> str:
        matches = re.findall(r"<answer>(.*?)</answer>", str(answer), flags=re.DOTALL | re.IGNORECASE)
        if matches:
            candidate = matches[-1].strip().upper()
        else:
            candidate = str(answer).strip().upper()
        if candidate in {"YES", "NO"}:
            return candidate
        token_match = re.search(r"\b(YES|NO)\b", candidate)
        return token_match.group(1) if token_match else ""

    def _generate_rubric(
        self,
        info: Dict[str, Any],
        teacher_answers: Sequence[str],
        student_answers: Sequence[str],
    ) -> Dict[str, Any]:
        if self.rubricator_template is None:
            raise FileNotFoundError("Missing rubricator.txt prompt for dynamic ROPD reward.")
        ground_truth = info["ground_truth"] if self.include_ground_truth else "N/A"
        prompt = _render_template(
            self.rubricator_template,
            {
                "question": info["raw_prompt"],
                "ground_truth": ground_truth,
                "teacher_response": _render_answer_block("Reference", teacher_answers),
                "student_response": _render_answer_block("Student", student_answers),
            },
        )
        raw = self.client.create_text(
            model=self.rubricator_model,
            text=prompt,
            image_paths=info["image_paths"],
            temperature=self.rubricator_temperature,
            max_output_tokens=self.rubricator_max_output_tokens,
            json_mode=True,
        )
        return self._validate_rubric(_extract_json_payload(raw))

    def _render_verifier_prompt(
        self,
        info: Mapping[str, Any],
        rubric: Mapping[str, Any],
        answers: Sequence[str],
    ) -> str:
        ground_truth = info["ground_truth"] if self.include_ground_truth else "N/A"
        verifier_rubrics = [
            {
                "criterion_id": str(item["criterion_id"]),
                "criterion": str(
                    item.get("criterion")
                    or item.get("description")
                    or item.get("title")
                    or ""
                ),
                "points": float(item["points"]),
            }
            for item in rubric["rubrics"]
        ]
        return _render_template(
            self.verifier_template,
            {
                "question": info["raw_prompt"],
                "ground_truth": ground_truth,
                "rubrics": json.dumps(
                    verifier_rubrics, ensure_ascii=False, separators=(",", ":")
                ),
                "answers": _render_answer_block("Answer", answers),
            },
        )

    def _count_text_tokens(self, text: str) -> int:
        try:
            encoded = self.tokenizer.encode(str(text), add_special_tokens=False)
            return len(encoded)
        except Exception:
            # Character fallback is deliberately conservative for mixed prose,
            # formulas, and JSON prompt content.
            return max(1, (len(str(text)) + 2) // 3)

    def _rcpc_verifier_output_tokens_per_item(self, criterion_count: int) -> int:
        return max(
            self.rcpc_verifier_output_tokens_per_answer,
            48 + 12 * max(1, int(criterion_count)),
        )

    def _rcpc_verifier_output_limit(self, answer_count: int, criterion_count: int) -> int:
        estimated = max(
            self.rcpc_verifier_min_output_tokens,
            int(answer_count) * self._rcpc_verifier_output_tokens_per_item(criterion_count),
        )
        return min(self.verifier_max_output_tokens, estimated)

    def _verify_answers(
        self,
        info: Dict[str, Any],
        rubric: Dict[str, Any],
        answers: Sequence[str],
        *,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        prompt = self._render_verifier_prompt(info, rubric, answers)
        raw = self.client.create_text(
            model=self.verifier_model,
            text=prompt,
            image_paths=info["image_paths"],
            temperature=self.verifier_temperature,
            max_output_tokens=(
                self.verifier_max_output_tokens
                if max_output_tokens is None
                else max(64, min(self.verifier_max_output_tokens, int(max_output_tokens)))
            ),
            json_mode=True,
        )
        return self._validate_verifier_payload(_extract_json_payload(raw), rubric, expected_count=len(answers))

    def _maybe_add_shadow_attribution(self, info: Mapping[str, Any], result: Dict[str, Any]) -> None:
        if not self.shadow_attribution_enabled or not info.get("run_shadow_attribution", False):
            return
        try:
            result["shadow_attribution"] = self._generate_shadow_attribution(
                info,
                result["rubric"],
                result["student_answers"],
            )
        except Exception as exc:
            result["shadow_attribution_error"] = "{}: {}".format(type(exc).__name__, exc)

    def _generate_shadow_attribution(
        self,
        info: Mapping[str, Any],
        rubric: Mapping[str, Any],
        student_answers: Sequence[str],
    ) -> Dict[str, Any]:
        if self.attributor_template is None:
            raise RuntimeError("shadow attribution template is not loaded")
        ground_truth = info["ground_truth"] if self.include_ground_truth else "N/A"
        prompt = _render_template(
            self.attributor_template,
            {
                "question": str(info["raw_prompt"]),
                "ground_truth": str(ground_truth),
                "rubrics": json.dumps(rubric["rubrics"], ensure_ascii=False, indent=2),
                "answers": _render_answer_block("Student Answer", student_answers),
            },
        )
        raw = self.client.create_text(
            model=self.verifier_model,
            text=prompt,
            image_paths=info["image_paths"],
            temperature=self.verifier_temperature,
            max_output_tokens=self.shadow_attribution_max_output_tokens,
            json_mode=True,
        )
        payload = self._validate_shadow_attribution_payload(
            _extract_json_payload(raw),
            rubric,
            expected_count=len(student_answers),
        )
        return self._resolve_shadow_attribution_spans(payload, student_answers)

    def _validate_shadow_attribution_payload(
        self,
        payload: Dict[str, Any],
        rubric: Mapping[str, Any],
        *,
        expected_count: int,
    ) -> Dict[str, Any]:
        if payload.get("schema_version") != SHADOW_ATTRIBUTION_SCHEMA_VERSION:
            raise ValueError("shadow attribution schema_version mismatch")
        answers = payload.get("answers")
        if not isinstance(answers, list) or len(answers) != expected_count:
            raise ValueError("shadow attribution answer count mismatch")

        criterion_ids = [str(item["criterion_id"]) for item in rubric["rubrics"]]
        allowed_types = {"supporting_span", "explicit_error", "missing", "global"}
        normalized_answers = []
        for answer_index, answer in enumerate(answers, start=1):
            if not isinstance(answer, Mapping) or int(answer.get("answer_index", answer_index)) != answer_index:
                raise ValueError("shadow attribution answer_index must preserve input order")
            attributions = answer.get("attributions")
            if not isinstance(attributions, list):
                raise ValueError("shadow attribution attributions must be a list")
            by_id = {}
            for attribution in attributions:
                if not isinstance(attribution, Mapping):
                    raise ValueError("shadow attribution item must be an object")
                criterion_id = str(attribution.get("criterion_id", ""))
                if criterion_id not in criterion_ids or criterion_id in by_id:
                    raise ValueError("shadow attribution criterion_id mismatch")
                attribution_type = str(attribution.get("attribution_type", ""))
                if attribution_type not in allowed_types:
                    raise ValueError("unsupported shadow attribution_type")
                quote = str(attribution.get("quote") or "")
                by_id[criterion_id] = {
                    "criterion_id": criterion_id,
                    "attribution_type": attribution_type,
                    "quote": quote,
                }
            if set(by_id) != set(criterion_ids):
                raise ValueError("shadow attribution must cover every rubric criterion")
            normalized_answers.append(
                {
                    "answer_index": answer_index,
                    "attributions": [by_id[criterion_id] for criterion_id in criterion_ids],
                }
            )
        return {
            "schema_version": SHADOW_ATTRIBUTION_SCHEMA_VERSION,
            "answers": normalized_answers,
        }

    def _resolve_shadow_attribution_spans(
        self,
        payload: Dict[str, Any],
        student_answers: Sequence[str],
    ) -> Dict[str, Any]:
        resolved_answers = []
        for answer_item, response_text in zip(payload["answers"], student_answers):
            offsets = self._token_offsets(response_text)
            resolved_attributions = []
            for attribution in answer_item["attributions"]:
                quote = attribution["quote"]
                char_start = response_text.find(quote) if quote else -1
                char_end = char_start + len(quote) if char_start >= 0 else -1
                token_start, token_end = self._char_span_to_token_span(offsets, char_start, char_end)
                resolved_attributions.append(
                    {
                        **attribution,
                        "matched": char_start >= 0,
                        "char_start": char_start,
                        "char_end": char_end,
                        "token_start": token_start,
                        "token_end": token_end,
                    }
                )
            resolved_answers.append(
                {
                    "answer_index": answer_item["answer_index"],
                    "attributions": resolved_attributions,
                }
            )
        return {
            "schema_version": SHADOW_ATTRIBUTION_SCHEMA_VERSION,
            "answers": resolved_answers,
        }

    def _token_offsets(self, text: str) -> List[Tuple[int, int]]:
        try:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = encoded["offset_mapping"]
            if hasattr(offsets, "tolist"):
                offsets = offsets.tolist()
            return [(int(start), int(end)) for start, end in offsets]
        except Exception:
            return []

    @staticmethod
    def _char_span_to_token_span(
        offsets: Sequence[Tuple[int, int]],
        char_start: int,
        char_end: int,
    ) -> Tuple[int, int]:
        if char_start < 0 or char_end <= char_start or not offsets:
            return -1, -1
        overlapping = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > char_start and token_start < char_end
        ]
        if not overlapping:
            return -1, -1
        return overlapping[0], overlapping[-1] + 1

    def _validate_rubric(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("schema_version") != RUBRIC_SCHEMA_VERSION:
            raise ValueError("rubric schema_version mismatch")
        rubrics = payload.get("rubrics")
        if not isinstance(rubrics, list) or len(rubrics) == 0:
            raise ValueError("rubric must contain at least one criterion")

        total = 0
        normalized_rubrics = []
        for index, item in enumerate(rubrics, start=1):
            if not isinstance(item, dict):
                raise ValueError("rubric criterion must be an object")
            criterion = str(item.get("criterion", "")).strip()
            if not criterion:
                raise ValueError("rubric criterion must be non-empty")
            points = int(item.get("points", 0))
            if points < 1 or points > 5:
                raise ValueError("rubric points must be in [1, 5]")
            category = str(item.get("category") or "Task")
            points = self._cap_rubric_points(category=category, criterion=criterion, points=points)
            total += points
            normalized_rubrics.append(
                {
                    # Canonicalize IDs so per-position W&B metrics and shadow
                    # attribution stay stable even if the rubricator emits a
                    # missing, duplicate, or out-of-order identifier.
                    "criterion_id": "c{}".format(index),
                    "category": category,
                    "criterion": criterion,
                    "points": points,
                }
            )

        maximum_score = int(payload.get("maximum_score", total))
        if maximum_score != total:
            maximum_score = total
        return {
            "schema_version": RUBRIC_SCHEMA_VERSION,
            "rubrics": normalized_rubrics,
            "maximum_score": maximum_score,
        }

    def _cap_rubric_points(self, *, category: str, criterion: str, points: int) -> int:
        text = "{} {}".format(category, criterion).lower()
        final_answer_patterns = (
            r"final\s+label",
            r"known\s+final",
            r"final\s+answer",
            r"<answer>.*known",
            r"known.*<answer>",
            r"answer.*matches",
            r"matches.*answer",
            r"yes/no",
            r"最终.*标签",
            r"最终.*答案",
            r"答案.*一致",
            r"标签.*一致",
        )
        if any(re.search(pattern, text) for pattern in final_answer_patterns):
            return min(points, max(1, self.final_label_points_cap))

        format_patterns = (
            r"output\s+protocol",
            r"\bformat\b",
            r"tag",
            r"<reasoning>",
            r"<think>",
            r"格式",
            r"标签格式",
        )
        if any(re.search(pattern, text) for pattern in format_patterns):
            return min(points, max(1, self.format_points_cap))

        return points

    def _validate_verifier_payload(
        self,
        payload: Dict[str, Any],
        rubric: Dict[str, Any],
        *,
        expected_count: int,
    ) -> Dict[str, Any]:
        if payload.get("schema_version") != BATCH_VERIFIER_SCHEMA_VERSION:
            raise ValueError("verifier schema_version mismatch")
        answers = payload.get("answers")
        if not isinstance(answers, list) or len(answers) != expected_count:
            raise ValueError("verifier answer count mismatch")

        rubric_points = [int(item["points"]) for item in rubric["rubrics"]]
        normalized_answers = []
        for index, answer in enumerate(answers, start=1):
            if not isinstance(answer, dict):
                raise ValueError("verifier answer item must be an object")
            if int(answer.get("answer_index", index)) != index:
                raise ValueError("verifier answer_index must preserve input order")
            judgement = answer.get("judgement")
            if not isinstance(judgement, list) or len(judgement) != len(rubric_points):
                raise ValueError("verifier judgement length mismatch")
            bool_judgement = [self._parse_verifier_bool(item) for item in judgement]
            final_score = float(sum(point for point, ok in zip(rubric_points, bool_judgement) if ok))
            normalized_answers.append(
                {
                    "answer_index": index,
                    "judgement": bool_judgement,
                    "final_score": final_score,
                }
            )
        return {
            "schema_version": BATCH_VERIFIER_SCHEMA_VERSION,
            "answers": normalized_answers,
        }

    def _parse_verifier_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "true":
                return True
            if normalized == "false":
                return False
        raise ValueError("verifier judgement values must be booleans")

    def _build_shuffled_answer_items(
        self,
        *,
        uid: str,
        teacher_answers: Sequence[str],
        student_answers: Sequence[str],
    ) -> List[Dict[str, Any]]:
        items = []
        for index, answer in enumerate(teacher_answers):
            items.append({"source": "teacher", "source_index": index, "text": answer})
        for index, answer in enumerate(student_answers):
            items.append({"source": "student", "source_index": index, "text": answer})
        return sorted(items, key=lambda item: self._answer_shuffle_key(uid, item))

    def _answer_shuffle_key(self, uid: str, item: Mapping[str, Any]) -> Tuple[str, str, int]:
        digest = hashlib.sha256(
            "{}\x1f{}\x1f{}\x1f{}".format(
                uid,
                item["source"],
                item["source_index"],
                item["text"],
            ).encode("utf-8")
        ).hexdigest()
        return digest, str(item["source"]), int(item["source_index"])

    def _restore_student_scores(
        self,
        answer_items: Sequence[Mapping[str, Any]],
        ordered_scores: Sequence[float],
        student_count: int,
    ) -> List[float]:
        student_scores = [None] * student_count
        for item, score in zip(answer_items, ordered_scores):
            if item["source"] == "student":
                student_scores[int(item["source_index"])] = float(score)
        if any(score is None for score in student_scores):
            raise ValueError("failed to restore all student verifier scores")
        return [float(score) for score in student_scores]

    def _restore_student_verifier_answers(
        self,
        answer_items: Sequence[Mapping[str, Any]],
        verifier_answers: Sequence[Mapping[str, Any]],
        student_count: int,
    ) -> List[Mapping[str, Any]]:
        student_answers = [None] * student_count
        for item, verifier_answer in zip(answer_items, verifier_answers):
            if item["source"] == "student":
                student_answers[int(item["source_index"])] = verifier_answer
        if any(answer is None for answer in student_answers):
            raise ValueError("failed to restore all student verifier judgements")
        return list(student_answers)

    def _rule_score(self, info: Mapping[str, Any]) -> float:
        return float(ipr_compute_score(str(info["response_text"]), str(info["ground_truth"])))

    def _maybe_print_group(self, first_info: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        with self._print_lock:
            self._maybe_print_rcpc_intervention_summary_locked(first_info, result)
            if self.already_print >= self.num_examine:
                return
            self.already_print += 1
            image_paths = first_info.get("image_paths") or []
            print("[ropd uid]", first_info["uid"])
            print("[ropd question]", first_info["raw_prompt"])
            print("[ropd image_count]", len(image_paths))
            print("[ropd teacher_usable_image_count]", self.client.count_usable_image_paths(image_paths))
            skipped_images = self.client.skipped_image_paths(image_paths)
            if skipped_images:
                print("[ropd skipped_images_sample]", skipped_images)
            print("[ropd ground_truth]", first_info["ground_truth"])
            print("[ropd ok]", result["ok"])
            print("[ropd scores]", result["scores"])
            if result.get("criterion_advantages"):
                print("[ropd criterion advantages]", result["criterion_advantages"])
            if self.print_rcpc_outputs and result.get("rcpc_candidates"):
                print("[ropd rcpc metrics]", json.dumps(result.get("rcpc_metrics", {}), ensure_ascii=False))
                print("[ropd rcpc candidates]")
                print(json.dumps(result["rcpc_candidates"], ensure_ascii=False, indent=2)[:20000])
            if self.print_rcpc_outputs and result.get("rcpc_interventions"):
                print("[ropd rcpc interventions]")
                print(json.dumps(result["rcpc_interventions"], ensure_ascii=False, indent=2)[:20000])
            if result.get("rcpc_error"):
                print("[ropd rcpc error]", result["rcpc_error"])
            if self.print_rubric_outputs and result.get("rubric"):
                print("[ropd rubric]")
                print(json.dumps(result["rubric"], ensure_ascii=False, indent=2))
            if self.print_shadow_attributions and result.get("shadow_attribution"):
                print("[ropd shadow attribution]")
                print(json.dumps(result["shadow_attribution"], ensure_ascii=False, indent=2))
            if result.get("shadow_attribution_error"):
                print("[ropd shadow attribution error]", result["shadow_attribution_error"])
            if self.print_teacher_outputs:
                raw_teacher_answers = result.get("raw_teacher_answers") or result.get("teacher_answers") or []
                for index, answer in enumerate(raw_teacher_answers, start=1):
                    print("[ropd teacher raw answer {}]".format(index))
                    print(answer)
                filtered_teacher_answers = result.get("teacher_answers") or []
                if raw_teacher_answers and filtered_teacher_answers != raw_teacher_answers:
                    for index, answer in enumerate(filtered_teacher_answers, start=1):
                        print("[ropd teacher kept answer {}]".format(index))
                        print(answer)
            if self.print_student_outputs:
                student_answers = result.get("student_answers") or []
                student_batch_indices = result.get("student_batch_indices") or list(range(len(student_answers)))
                scores = result.get("scores") or {}
                max_student_outputs = self.print_max_student_outputs
                if max_student_outputs > 0:
                    student_answers = student_answers[:max_student_outputs]
                for index, answer in enumerate(student_answers, start=1):
                    batch_index = student_batch_indices[index - 1] if index - 1 < len(student_batch_indices) else index - 1
                    if isinstance(scores, Mapping):
                        score_value = scores.get(batch_index, "N/A")
                    else:
                        score_value = "N/A"
                    format_valid = result.get("student_format_valid") or []
                    format_value = format_valid[index - 1] if index - 1 < len(format_valid) else "N/A"
                    clipped_values = result.get("student_response_clipped") or []
                    clipped_value = clipped_values[index - 1] if index - 1 < len(clipped_values) else "N/A"
                    print(
                        "[ropd student answer {} score={} format_valid={} clipped={} chars={} "
                        "has_<reasoning>={} has_</reasoning>={} has_<answer>={} has_</answer>={} "
                        "has_legacy_<think>={}]".format(
                            index,
                            score_value,
                            format_value,
                            clipped_value,
                            len(str(answer)),
                            "<reasoning>" in str(answer).lower(),
                            "</reasoning>" in str(answer).lower(),
                            "<answer>" in str(answer).lower(),
                            "</answer>" in str(answer).lower(),
                            "<think>" in str(answer).lower(),
                        )
                    )
                    print(answer)
                    if self.print_verifier_outputs:
                        verifier_answers = result.get("student_verifier_answers") or []
                        if index - 1 < len(verifier_answers):
                            print("[ropd student verifier {}]".format(index))
                            print(json.dumps(verifier_answers[index - 1], ensure_ascii=False))
            if result.get("error"):
                print("[ropd error]", result["error"])

    def _maybe_print_rcpc_intervention_summary_locked(
        self,
        first_info: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> None:
        if not self.rcpc_print_intervention_summary:
            return
        interventions = result.get("rcpc_interventions") or {}
        if not interventions:
            return
        step = int(first_info.get("global_step", getattr(self, "_current_global_step", -1)))
        if step <= 0:
            return
        interval = int(self.rcpc_print_interval)
        if step != 1 and (interval <= 0 or step % interval != 0):
            return
        printed = self._rcpc_summary_printed_by_step.get(step, 0)
        if self.rcpc_print_max_groups >= 0 and printed >= self.rcpc_print_max_groups:
            return
        self._rcpc_summary_printed_by_step[step] = printed + 1

        max_blocks = int(self.rcpc_print_max_blocks)
        student_answers = result.get("student_answers") or []
        student_batch_indices = result.get("student_batch_indices") or list(range(len(student_answers)))
        batch_to_answer = {
            int(batch_index): str(student_answers[index])
            for index, batch_index in enumerate(student_batch_indices)
            if index < len(student_answers)
        }
        trajectory_items = []
        emitted_blocks = 0
        for batch_index, block_map in sorted(interventions.items(), key=lambda item: int(item[0])):
            blocks = []
            for block_index, payload in sorted(block_map.items(), key=lambda item: int(item[0])):
                if max_blocks >= 0 and emitted_blocks >= max_blocks:
                    break
                block = payload.get("block", {}) or {}
                blocks.append(
                    {
                        "block_index": int(block_index),
                        "response_index": int(payload.get("response_index", -1)),
                        "action_ids": block.get("action_ids", []),
                        "token_start": int(block.get("token_start", -1)),
                        "token_end": int(block.get("token_end", -1)),
                        "selection_priority": float(block.get("selection_priority", 0.0)),
                        "anchor_robust_z": float(block.get("anchor_robust_z", 0.0)),
                        "block_mean_robust_z": float(block.get("block_mean_robust_z", 0.0)),
                        "text": str(block.get("text", "")),
                        "factual_text_preview": str(payload.get("factual_text", ""))[:1200],
                        "control_text_preview": str(payload.get("control_text", ""))[:1200],
                        "factual_sample_count": int(payload.get("factual_sample_count", 0)),
                        "control_sample_count": int(payload.get("control_sample_count", 0)),
                        "counterfactual_sample_count": int(payload.get("counterfactual_sample_count", 1)),
                        "factual_format_valid": payload.get("factual_format_valid", []),
                        "control_format_valid": payload.get("control_format_valid", []),
                        "deleted_content_reappeared": payload.get(
                            "control_deleted_content_reappeared", []
                        ),
                        "deleted_content_reappearance_scores": payload.get(
                            "control_deleted_content_reappearance_scores", []
                        ),
                        "deleted_content_exact_match": payload.get(
                            "control_deleted_content_exact_match", []
                        ),
                        "pair_format_valid": payload.get("pair_format_valid", []),
                        "valid_pair_count": int(payload.get("valid_pair_count", 0)),
                        "invalid_pair_count": int(payload.get("invalid_pair_count", 0)),
                        "factual_scores": payload.get("factual_scores", []),
                        "control_scores": payload.get("control_scores", []),
                        "intervened_scores": payload.get("intervened_scores", []),
                        "criterion_effects": payload.get("criterion_effects", {}),
                        "criterion_standard_errors": payload.get("criterion_standard_errors", {}),
                        "criterion_paired_differences": payload.get("criterion_paired_differences", {}),
                        "calibrated_effects": payload.get("calibrated_effects", {}),
                        "score_effect": float(payload.get("score_effect", 0.0)),
                        "original_score": float(payload.get("original_score", 0.0)),
                        "intervened_score": float(payload.get("intervened_score", 0.0)),
                    }
                )
                emitted_blocks += 1
            if blocks:
                answer_text = batch_to_answer.get(int(batch_index), "")
                trajectory_items.append(
                    {
                        "batch_index": int(batch_index),
                        "trajectory_preview": answer_text[:1200],
                        "blocks": blocks,
                    }
                )
            if max_blocks >= 0 and emitted_blocks >= max_blocks:
                break

        payload = {
            "step": step,
            "uid": first_info.get("uid", ""),
            "budget": {
                "B_group": self.rcpc_budget,
                "derive_candidates_from_budget": self.rcpc_derive_candidates_from_budget,
                "group_top_actions": self.rcpc_top_actions,
                "group_top_blocks": self.rcpc_top_blocks,
                "groups_per_batch": self.rcpc_intervention_max_groups_per_batch,
                "blocks_per_answer": self.rcpc_intervention_max_blocks_per_answer,
                "blocks_per_group": self.rcpc_intervention_max_blocks_per_group,
                "paired_samples_per_arm": self.rcpc_counterfactual_samples,
                "counterfactual_batch_size": self.rcpc_counterfactual_batch_size,
                "verifier_max_answers_per_batch": self.rcpc_verifier_batch_size,
                "verifier_max_input_tokens": self.rcpc_verifier_max_input_tokens,
                "print_max_groups": self.rcpc_print_max_groups,
                "print_max_blocks": self.rcpc_print_max_blocks,
            },
            "transport": {
                "lambda": self.rcpc_transport_lambda,
                "calibration": "paired_estimator_variance_with_prior_floor",
                "effect_variance_prior": self.rcpc_effect_variance_prior,
            },
            "scores": result.get("scores", {}),
            "criterion_advantages": result.get("criterion_advantages", {}),
            "rcpc_metrics": result.get("rcpc_metrics", {}),
            "trajectories": trajectory_items,
        }
        print("[ropd rcpc intervention summary]")
        print(json.dumps(payload, ensure_ascii=False, indent=2)[:30000])

    def _write_debug(self, results: Iterable[Mapping[str, Any]]) -> None:
        if self.debug_path is None:
            return
        self.debug_path.parent.mkdir(parents=True, exist_ok=True)
        with self.debug_path.open("a", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
