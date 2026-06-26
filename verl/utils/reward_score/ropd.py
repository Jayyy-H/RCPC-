import base64
import hashlib
import json
import mimetypes
import os
import re
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from verl import DataProto
from verl.utils.reward_score.ipr import ipr_compute_score
from verl.utils.reward_score.rcpc import (
    apply_intervention,
    build_candidates,
    build_token_advantages,
    build_token_offsets,
)


RUBRIC_SCHEMA_VERSION = "ropd.rubric.v1"
BATCH_VERIFIER_SCHEMA_VERSION = "ropd.batch_verifier.v2"
SHADOW_ATTRIBUTION_SCHEMA_VERSION = "ropd.shadow_attribution.v1"


_PLACEHOLDER_RE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_SUBSTANTIVE_RE = re.compile(r"[\w\u4e00-\u9fff]", re.UNICODE)


def _has_strict_cot_format(text: str) -> bool:
    think_matches = list(_THINK_RE.finditer(str(text)))
    answer_matches = list(_ANSWER_RE.finditer(str(text)))
    if len(think_matches) != 1 or len(answer_matches) != 1:
        return False

    think_match = think_matches[0]
    answer_match = answer_matches[0]
    if think_match.end() > answer_match.start():
        return False
    if not _SUBSTANTIVE_RE.search(think_match.group(1).strip()):
        return False

    return answer_match.group(1).strip().upper() in {"YES", "NO"}


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
    # Some internal modelhub configs store the full responses endpoint.
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
        client_kwargs = {"api_key": api_key, "timeout": timeout}
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

        base_model = os.getenv("ROPD_MODEL") or _cfg(reward_config, "ropd_model", "gpt-5.4")
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
        self.include_ground_truth = bool(_cfg(reward_config, "ropd_include_ground_truth", True))
        self.filter_teacher_by_answer = bool(_cfg(reward_config, "ropd_filter_teacher_by_answer", True))
        self.print_teacher_outputs = bool(_cfg(reward_config, "ropd_print_teacher_outputs", True))
        self.print_student_outputs = bool(_cfg(reward_config, "ropd_print_student_outputs", False))
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
        self.rcpc_top_actions = max(1, int(_cfg(reward_config, "ropd_rcpc_top_actions", 12)))
        self.rcpc_top_blocks = max(1, int(_cfg(reward_config, "ropd_rcpc_top_blocks", 6)))
        self.rcpc_min_action_chars = max(1, int(_cfg(reward_config, "ropd_rcpc_min_action_chars", 12)))
        self.rcpc_max_action_chars = max(1, int(_cfg(reward_config, "ropd_rcpc_max_action_chars", 260)))
        self.rcpc_max_action_tokens = max(1, int(_cfg(reward_config, "ropd_rcpc_max_action_tokens", 24)))
        self.rcpc_min_robust_denom = float(_cfg(reward_config, "ropd_rcpc_min_robust_denom", 0.05))
        self.rcpc_min_anchor_z = float(_cfg(reward_config, "ropd_rcpc_min_anchor_z", 0.5))
        self.rcpc_intervention_enabled = bool(_cfg(reward_config, "ropd_rcpc_intervention_enabled", False))
        self.rcpc_intervention_max_groups_per_batch = max(
            0, int(_cfg(reward_config, "ropd_rcpc_intervention_max_groups_per_batch", 1))
        )
        self.rcpc_intervention_max_blocks_per_answer = max(
            0, int(_cfg(reward_config, "ropd_rcpc_intervention_max_blocks_per_answer", 2))
        )
        self.rcpc_intervention_mode = str(_cfg(reward_config, "ropd_rcpc_intervention_mode", "mask"))
        self.rcpc_fallback_to_criterion_advantage = bool(
            _cfg(reward_config, "ropd_rcpc_fallback_to_criterion_advantage", True)
        )
        self.print_rcpc_outputs = bool(_cfg(reward_config, "ropd_print_rcpc_outputs", False))
        self.require_strict_cot_format = bool(_cfg(reward_config, "ropd_require_strict_cot_format", True))
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
        prompt_dir_value = str(_cfg(reward_config, "ropd_prompt_dir", "prompts/ropd"))
        prompt_dir = Path(prompt_dir_value)
        if not prompt_dir.is_absolute():
            prompt_dir = repo_root / prompt_dir
        self.teacher_template = (prompt_dir / "teacher.txt").read_text(encoding="utf-8")
        self.rubricator_template = (prompt_dir / "rubricator.txt").read_text(encoding="utf-8")
        self.verifier_template = (prompt_dir / "verifier.txt").read_text(encoding="utf-8")
        self.attributor_template = None
        if self.shadow_attribution_enabled:
            self.attributor_template = (prompt_dir / "attributor.txt").read_text(encoding="utf-8")

        debug_path_value = _cfg(reward_config, "ropd_debug_path", None)
        self.debug_path = Path(debug_path_value) if debug_path_value else None
        if self.debug_path is not None and not self.debug_path.is_absolute():
            self.debug_path = repo_root / self.debug_path

        self.client = RopdOpenAIClient(
            api_key_env=str(_cfg(reward_config, "ropd_api_key_env", "OPENAI_API_KEY")),
            base_url=_cfg(reward_config, "ropd_base_url", None),
            base_url_env=str(_cfg(reward_config, "ropd_base_url_env", "OPENAI_BASE_URL")),
            api_style=str(_cfg(reward_config, "ropd_api_style", "responses")),
            timeout=float(_cfg(reward_config, "ropd_request_timeout", 120.0)),
            include_images=bool(_cfg(reward_config, "ropd_include_images", True)),
            max_image_bytes=int(_cfg(reward_config, "ropd_max_image_bytes", 8388608)),
        )

    def __call__(self, data: DataProto) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        response_infos = self._collect_response_infos(data)

        if not self._has_training_group_keys(data):
            self._fill_rule_rewards(reward_tensor, response_infos)
            return reward_tensor

        grouped = self._group_onpolicy_infos(response_infos)
        groups = list(grouped.values())
        if self.shadow_attribution_enabled:
            for group_index, group in enumerate(groups):
                group[0]["run_shadow_attribution"] = (
                    group_index < self.shadow_attribution_max_groups_per_batch
                )
        if self.rcpc_enabled and self.rcpc_intervention_enabled:
            for group_index, group in enumerate(groups):
                group[0]["run_rcpc_intervention"] = (
                    group_index < self.rcpc_intervention_max_groups_per_batch
                )
        if self.max_concurrency <= 1 or len(groups) <= 1:
            results = [self._score_group(group) for group in groups]
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(groups))) as executor:
                results = list(executor.map(self._score_group, groups))

        for result in results:
            for batch_index, score in result["scores"].items():
                response_length = response_infos[batch_index]["response_length"]
                reward_tensor[batch_index, response_length - 1] = float(score)

        if self.use_criterion_advantage:
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
            data.meta_info["ropd_metrics"] = self._collect_criterion_metrics(results)

        if self.score_offpolicy:
            for info in response_infos:
                if not info["is_onpolicy"] and info["response_length"] > 0:
                    reward_tensor[info["batch_index"], info["response_length"] - 1] = self._rule_score(info)

        self._write_debug(results)
        return reward_tensor

    def _collect_criterion_metrics(self, results: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        criterion_values: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {"judgements": [], "advantages": []}
        )
        combined_advantages = []
        for result in results:
            combined_advantages.extend(float(value) for value in result.get("criterion_advantages", {}).values())
            for criterion_id, stats in result.get("criterion_stats", {}).items():
                criterion_values[str(criterion_id)]["judgements"].extend(
                    float(value) for value in stats.get("judgements", [])
                )
                criterion_values[str(criterion_id)]["advantages"].extend(
                    float(value) for value in stats.get("advantages", [])
                )

        metrics = {
            "criterion_advantage/mean": (
                sum(combined_advantages) / len(combined_advantages) if combined_advantages else 0.0
            ),
            "criterion_advantage/std": _population_std(combined_advantages),
        }
        rcpc_metric_values: Dict[str, List[float]] = defaultdict(list)
        for result in results:
            for key, value in result.get("rcpc_metrics", {}).items():
                rcpc_metric_values[str(key)].append(float(value))
        for key, values in rcpc_metric_values.items():
            metrics[key] = sum(values) / len(values) if values else 0.0
        for criterion_id, values in sorted(criterion_values.items()):
            safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", criterion_id)
            judgements = values["judgements"]
            advantages = values["advantages"]
            metrics["rubric/{}/pass_ratio".format(safe_id)] = (
                sum(judgements) / len(judgements) if judgements else 0.0
            )
            metrics["rubric/{}/adv_std".format(safe_id)] = _population_std(advantages)
        return metrics

    def _has_training_group_keys(self, data: DataProto) -> bool:
        return "uid" in data.non_tensor_batch and "is_onpolicy" in data.non_tensor_batch

    def _collect_response_infos(self, data: DataProto) -> List[Dict[str, Any]]:
        responses = data.batch["responses"]
        old_log_probs = data.batch.get("old_log_probs", None)
        response_width = responses.shape[-1]
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_width:]
        valid_response_lengths = response_mask.sum(dim=-1)

        uids = _as_list(data.non_tensor_batch.get("uid"))
        is_onpolicy_values = _as_list(data.non_tensor_batch.get("is_onpolicy"))
        raw_prompts = _as_list(data.non_tensor_batch.get("raw_prompt"))
        image_paths_values = _as_list(data.non_tensor_batch.get("image_paths"))
        answers = _as_list(data.non_tensor_batch.get("answer"))

        infos = []
        for batch_index in range(len(data)):
            response_length = int(valid_response_lengths[batch_index].item())
            valid_response_ids = responses[batch_index][:response_length].detach().cpu().tolist()
            response_text = self._decode_response_text(responses[batch_index], response_length)
            if old_log_probs is not None and response_length > 0:
                token_uncertainties = (
                    -old_log_probs[batch_index][:response_length].detach().float().cpu()
                ).tolist()
            else:
                token_uncertainties = [0.0] * response_length
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
                    "image_paths": [str(item) for item in _as_list(image_paths) if item],
                    "ground_truth": str(answers[batch_index]) if batch_index < len(answers) else "",
                    "response_text": response_text,
                    "response_token_ids": valid_response_ids,
                    "response_token_offsets": build_token_offsets(self.tokenizer, valid_response_ids),
                    "response_token_uncertainties": token_uncertainties,
                    "response_length": response_length,
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
        first = group[0]
        raw_teacher_answers = []
        teacher_answers = []
        rubric = None
        try:
            raw_teacher_answers = self._generate_teacher_answers(first)
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
            rubric = self._generate_rubric(first, teacher_answers, [item["response_text"] for item in group])
            answer_items = self._build_shuffled_answer_items(
                uid=first["uid"],
                teacher_answers=teacher_answers,
                student_answers=[item["response_text"] for item in group],
            )
            verifier_payload = self._verify_answers(first, rubric, [item["text"] for item in answer_items])
            ordered_scores = [float(answer["final_score"]) for answer in verifier_payload["answers"]]
            student_scores = self._restore_student_scores(answer_items, ordered_scores, len(group))
            student_verifier_answers = self._restore_student_verifier_answers(
                answer_items,
                verifier_payload["answers"],
                len(group),
            )
            maximum_score = float(rubric["maximum_score"])
            normalized_scores = [max(0.0, min(1.0, score / maximum_score)) for score in student_scores]
            student_format_valid = [
                _has_strict_cot_format(info["response_text"])
                for info in group
            ]
            if self.require_strict_cot_format:
                normalized_scores = [
                    score if student_format_valid[index] else 0.0
                    for index, score in enumerate(normalized_scores)
                ]
            criterion_advantage_values, criterion_stats = _compute_group_criterion_advantages(
                rubric,
                student_verifier_answers,
                student_format_valid,
                require_strict_cot_format=self.require_strict_cot_format,
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
                "error": "",
            }
            self._maybe_add_rcpc_credit(first, group, result)
            self._maybe_add_shadow_attribution(first, result)
            self._maybe_print_group(first, result)
            return result
        except Exception as exc:
            scores = {}
            for info in group:
                scores[info["batch_index"]] = self._rule_score(info) if self.fallback_to_ipr else 0.0
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
                "student_format_valid": [
                    _has_strict_cot_format(info["response_text"])
                    for info in group
                ],
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
                "error": "{}: {}".format(type(exc).__name__, exc),
            }
            self._maybe_print_group(first, result)
            return result

    def _maybe_add_rcpc_credit(
        self,
        first: Mapping[str, Any],
        group: Sequence[Dict[str, Any]],
        result: Dict[str, Any],
    ) -> None:
        if not self.rcpc_enabled or not result.get("ok", False):
            return
        try:
            candidates = [
                self._build_rcpc_candidates_for_response(info)
                for info in group
            ]
            result["rcpc_candidates"] = candidates
            interventions = {}
            if self.rcpc_intervention_enabled and first.get("run_rcpc_intervention", False):
                interventions = self._run_rcpc_interventions(first, group, result, candidates)
            result["rcpc_interventions"] = interventions
            token_advantages, metrics = self._build_rcpc_token_advantages(group, result, candidates, interventions)
            result["rcpc_token_advantages"] = token_advantages
            result["rcpc_metrics"] = metrics
        except Exception as exc:
            result["rcpc_error"] = "{}: {}".format(type(exc).__name__, exc)

    def _build_rcpc_candidates_for_response(self, info: Mapping[str, Any]) -> Dict[str, Any]:
        token_uncertainties = list(info.get("response_token_uncertainties") or [])
        response_length = int(info.get("response_length", 0))
        if len(token_uncertainties) < response_length:
            token_uncertainties.extend([0.0] * (response_length - len(token_uncertainties)))
        return build_candidates(
            str(info.get("response_text", "")),
            info.get("response_token_offsets") or [],
            token_uncertainties,
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
        intervention_items = []
        for response_index, (info, candidate) in enumerate(zip(group, candidates)):
            blocks = list(candidate.get("candidate_blocks", []))[: self.rcpc_intervention_max_blocks_per_answer]
            for block in blocks:
                intervention_items.append(
                    {
                        "response_index": response_index,
                        "batch_index": info["batch_index"],
                        "block_index": int(block["block_index"]),
                        "block": block,
                        "text": apply_intervention(
                            str(info["response_text"]),
                            block,
                            mode=self.rcpc_intervention_mode,
                        ),
                    }
                )
        if not intervention_items:
            return {}

        payload = self._verify_answers(
            dict(first),
            result["rubric"],
            [item["text"] for item in intervention_items],
        )
        original_answers = list(result["student_verifier_answers"])
        original_format_valid = list(result.get("student_format_valid", []))
        criterion_ids = [str(item["criterion_id"]) for item in result["rubric"]["rubrics"]]
        restored: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        for item, intervened_answer in zip(intervention_items, payload["answers"]):
            response_index = int(item["response_index"])
            original = original_answers[response_index]
            criterion_effects = {}
            for criterion_index, criterion_id in enumerate(criterion_ids):
                original_judgement = bool(original["judgement"][criterion_index])
                if self.require_strict_cot_format and not original_format_valid[response_index]:
                    original_judgement = False
                intervened_judgement = bool(intervened_answer["judgement"][criterion_index])
                criterion_effects[criterion_id] = (
                    (1.0 if original_judgement else 0.0)
                    - (1.0 if intervened_judgement else 0.0)
                )
            restored[item["batch_index"]][int(item["block_index"])] = {
                "criterion_effects": criterion_effects,
                "original_score": float(original["final_score"]),
                "intervened_score": float(intervened_answer["final_score"]),
                "score_effect": float(original["final_score"]) - float(intervened_answer["final_score"]),
            }
        return {int(batch_index): dict(value) for batch_index, value in restored.items()}

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

    def _build_rcpc_token_advantages(
        self,
        group: Sequence[Dict[str, Any]],
        result: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        interventions: Mapping[int, Mapping[int, Mapping[str, Any]]],
    ) -> Tuple[Dict[int, List[float]], Dict[str, float]]:
        token_advantages = {}
        metric_values: Dict[str, List[float]] = defaultdict(list)
        criterion_points = {
            str(item["criterion_id"]): float(item["points"])
            for item in result["rubric"]["rubrics"]
        }
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
            )
            token_advantages[batch_index] = values
            for key, value in metrics.items():
                metric_values[key].append(float(value))
        metrics = {
            key: (sum(values) / len(values) if values else 0.0)
            for key, values in metric_values.items()
        }
        metrics["rcpc/enabled"] = 1.0
        metrics["rcpc/intervention_enabled"] = 1.0 if self.rcpc_intervention_enabled else 0.0
        metrics["rcpc/intervention_groups"] = 1.0 if interventions else 0.0
        return token_advantages, metrics

    def _generate_teacher_answers(self, info: Dict[str, Any]) -> List[str]:
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

    def _verify_answers(
        self,
        info: Dict[str, Any],
        rubric: Dict[str, Any],
        answers: Sequence[str],
    ) -> Dict[str, Any]:
        ground_truth = info["ground_truth"] if self.include_ground_truth else "N/A"
        prompt = _render_template(
            self.verifier_template,
            {
                "question": info["raw_prompt"],
                "ground_truth": ground_truth,
                "rubrics": json.dumps(rubric["rubrics"], ensure_ascii=False, indent=2),
                "answers": _render_answer_block("Answer", answers),
            },
        )
        raw = self.client.create_text(
            model=self.verifier_model,
            text=prompt,
            image_paths=info["image_paths"],
            temperature=self.verifier_temperature,
            max_output_tokens=self.verifier_max_output_tokens,
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
            for index, answer in enumerate(student_answers, start=1):
                batch_index = student_batch_indices[index - 1] if index - 1 < len(student_batch_indices) else index - 1
                if isinstance(scores, Mapping):
                    score_value = scores.get(batch_index, "N/A")
                else:
                    score_value = "N/A"
                format_valid = result.get("student_format_valid") or []
                format_value = format_valid[index - 1] if index - 1 < len(format_valid) else "N/A"
                print("[ropd student answer {} score={} format_valid={}]".format(index, score_value, format_value))
                print(answer)
                if self.print_verifier_outputs:
                    verifier_answers = result.get("student_verifier_answers") or []
                    if index - 1 < len(verifier_answers):
                        print("[ropd student verifier {}]".format(index))
                        print(json.dumps(verifier_answers[index - 1], ensure_ascii=False))
        if result.get("error"):
            print("[ropd error]", result["error"])

    def _write_debug(self, results: Iterable[Mapping[str, Any]]) -> None:
        if self.debug_path is None:
            return
        self.debug_path.parent.mkdir(parents=True, exist_ok=True)
        with self.debug_path.open("a", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
