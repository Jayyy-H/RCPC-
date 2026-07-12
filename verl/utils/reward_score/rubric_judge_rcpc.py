"""Fixed-rubric LLM-as-judge reward with RCPC credit shaping.

This scorer is a lower-cost alternative to ROPD. It does not ask a teacher or
rubricator to generate per-prompt rubrics. Instead, every training sample must
carry a structured `rubric` field, typically copied from RaR-Science. The judge
only evaluates each rollout response against those fixed criteria.
"""

from __future__ import annotations

import ast
import json
import time
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from verl.utils.reward_score.ropd import (
    BATCH_VERIFIER_SCHEMA_VERSION,
    RopdIPRRewardScorer,
    _as_list,
    _compute_outcome_group_advantages,
    _extract_json_payload,
    _render_answer_block,
    _render_template,
    _sample_std,
)


FIXED_RUBRIC_SCHEMA_VERSION = "rcpc.fixed_rubric.v1"


def _parse_raw_rubric(value: Any) -> List[Mapping[str, Any]]:
    if value is None:
        raise ValueError("sample is missing required `rubric` field")
    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray, str)):
        value = value.tolist()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("sample `rubric` field is empty")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = ast.literal_eval(text)
    if isinstance(value, Mapping):
        if "rubrics" in value:
            value = value["rubrics"]
        elif "rubric" in value:
            value = value["rubric"]
    if not isinstance(value, list) or not value:
        raise ValueError("sample `rubric` must be a non-empty list")
    output = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("each rubric item must be an object")
        output.append(item)
    return output


def _format_fixed_criterion(title: str, description: str, signed_weight: float) -> str:
    base = f"{title}: {description}".strip(": ").strip()
    if signed_weight < 0:
        return (
            "Penalty/Pitfall criterion. Mark TRUE only if the response exhibits this pitfall, "
            "violates this avoidance requirement, or makes the described mistake. "
            "If the original wording says 'must not ...', 'should not ...', 'do not ...', "
            "or 'does not ...', invert it: TRUE means the answer does the forbidden thing "
            f"or fails to satisfy that avoidance. Rubric text: {base}"
        )
    return (
        "Positive criterion. Mark TRUE only if the response substantially satisfies this requirement. "
        f"Rubric text: {base}"
    )


def _safe_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _quality_value(judgement: bool, signed_weight: float) -> float:
    """Convert raw judge boolean into a quality-direction criterion value.

    Positive criteria: TRUE is good -> 1, FALSE -> 0.
    Negative criteria: TRUE means pitfall present -> bad, so quality is 0;
    FALSE means pitfall absent -> good, so quality is 1.
    """
    if signed_weight < 0:
        return 0.0 if judgement else 1.0
    return 1.0 if judgement else 0.0


def _compute_fixed_group_criterion_advantages(
    rubric: Mapping[str, Any],
    verifier_answers: Sequence[Mapping[str, Any]],
    format_valid: Sequence[bool],
    *,
    require_strict_cot_format: bool,
    epsilon: float = 1e-6,
) -> Tuple[List[float], Dict[str, Dict[str, List[float]]]]:
    rubrics = list(rubric["rubrics"])
    response_count = len(verifier_answers)
    combined = [0.0] * response_count
    active_weight = 0.0
    criterion_stats: Dict[str, Dict[str, List[float]]] = {}

    for criterion_index, criterion in enumerate(rubrics):
        signed_weight = float(criterion.get("raw_weight", criterion.get("points", 1.0)))
        point_weight = max(0.0, abs(signed_weight))
        values = []
        for response_index, answer in enumerate(verifier_answers):
            judgement = bool(answer["judgement"][criterion_index])
            value = _quality_value(judgement, signed_weight)
            if require_strict_cot_format and not format_valid[response_index]:
                value = 0.0
            values.append(value)

        mean = sum(values) / len(values) if values else 0.0
        std = _sample_std(values)
        if std > epsilon and point_weight > 0.0:
            advantages = [(value - mean) / (std + epsilon) for value in values]
            active_weight += point_weight
            for response_index, advantage in enumerate(advantages):
                combined[response_index] += point_weight * advantage
        else:
            advantages = [0.0] * response_count

        criterion_stats[str(criterion["criterion_id"])] = {
            # This is a quality-direction pass ratio. For negative/pitfall
            # criteria, 1 means the pitfall was avoided.
            "judgements": values,
            "advantages": advantages,
        }

    if active_weight > 0.0:
        combined = [value / active_weight for value in combined]
        combined_std = _sample_std(combined)
        if combined_std > epsilon:
            combined = [value / (combined_std + epsilon) for value in combined]
    return combined, criterion_stats


class FixedRubricRCPCRewardScorer(RopdIPRRewardScorer):
    """Use sample-provided rubrics, LLM judge scoring, and RCPC credit shaping."""

    def _collect_response_infos(self, data):
        infos = super()._collect_response_infos(data)
        rubric_values = _as_list(data.non_tensor_batch.get("rubric"))
        for batch_index, info in enumerate(infos):
            if batch_index < len(rubric_values):
                info["fixed_rubric_raw"] = rubric_values[batch_index]
            else:
                info["fixed_rubric_raw"] = None
        return infos

    def _build_fixed_rubric(self, raw_rubric: Any) -> Dict[str, Any]:
        raw_items = _parse_raw_rubric(raw_rubric)
        normalized = []
        positive_total = 0.0
        negative_total = 0.0
        for index, item in enumerate(raw_items, start=1):
            title = str(item.get("title") or f"Criterion {index}").strip()
            description = str(item.get("description") or item.get("criterion") or "").strip()
            if not description:
                raise ValueError(f"rubric item {index} is missing description")
            signed_weight = _safe_float(item.get("weight", item.get("points", 1.0)), 1.0)
            if signed_weight == 0.0:
                continue
            if signed_weight > 0:
                positive_total += signed_weight
            else:
                negative_total += signed_weight
            normalized.append(
                {
                    "criterion_id": f"c{len(normalized) + 1}",
                    "category": "FixedRubric",
                    "title": title,
                    "criterion": _format_fixed_criterion(title, description, signed_weight),
                    "description": description,
                    "points": abs(signed_weight),
                    "raw_weight": signed_weight,
                }
            )
        if not normalized:
            raise ValueError("rubric has no non-zero-weight criteria")
        if positive_total <= 0.0:
            positive_total = sum(float(item["points"]) for item in normalized)
        return {
            "schema_version": FIXED_RUBRIC_SCHEMA_VERSION,
            "rubrics": normalized,
            "maximum_score": positive_total,
            "minimum_score": negative_total,
        }

    @staticmethod
    def _normalize_signed_score(raw_score: float, rubric: Mapping[str, Any]) -> float:
        maximum = max(1e-6, float(rubric.get("maximum_score", 1.0)))
        return max(0.0, min(1.0, float(raw_score) / maximum))

    def _score_group(self, group: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        group_start = time.perf_counter()
        timing_metrics: Dict[str, float] = {}
        timing_counts: Dict[str, float] = {}
        first = group[0]
        rubric = None
        try:
            rubric = self._build_fixed_rubric(first.get("fixed_rubric_raw"))
            section_start = time.perf_counter()
            verifier_payload = self._verify_answers(
                dict(first),
                rubric,
                [item["response_text"] for item in group],
            )
            timing_metrics["timing_s/reward/verifier_initial_group"] = time.perf_counter() - section_start
            timing_counts["verifier_initial_requests"] = 1.0
            timing_counts["verifier_initial_answers"] = float(len(group))
            student_scores = [float(answer["final_score"]) for answer in verifier_payload["answers"]]
            normalized_scores = [
                self._normalize_signed_score(score, rubric)
                for score in student_scores
            ]
            student_format_valid = [self._response_format_valid(info) for info in group]
            if self.zero_score_on_format_error:
                normalized_scores = [
                    score if student_format_valid[index] else 0.0
                    for index, score in enumerate(normalized_scores)
                ]

            criterion_advantage_values, criterion_stats = _compute_fixed_group_criterion_advantages(
                rubric,
                verifier_payload["answers"],
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
                "student_verifier_answers": verifier_payload["answers"],
                "student_answers": [item["response_text"] for item in group],
                "student_batch_indices": [item["batch_index"] for item in group],
                "raw_teacher_answers": [],
                "teacher_answers": [],
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
            timing_metrics["timing_s/reward/group_total"] = time.perf_counter() - group_start
            if not first.get("defer_group_print", False):
                self._maybe_print_group(first, result)
            return result
        except Exception as exc:
            timing_metrics["timing_s/reward/group_total"] = time.perf_counter() - group_start
            scores = {info["batch_index"]: 0.0 for info in group}
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
                "student_format_valid": [self._response_format_valid(info) for info in group],
                "student_response_clipped": [bool(info.get("response_clipped", False)) for info in group],
                "student_verifier_answers": [],
                "student_answers": [info["response_text"] for info in group],
                "student_batch_indices": [info["batch_index"] for info in group],
                "raw_teacher_answers": [],
                "teacher_answers": [],
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

        rubric_weights = [float(item.get("raw_weight", item["points"])) for item in rubric["rubrics"]]
        normalized_answers = []
        for index, answer in enumerate(answers, start=1):
            if not isinstance(answer, dict):
                raise ValueError("verifier answer item must be an object")
            if int(answer.get("answer_index", index)) != index:
                raise ValueError("verifier answer_index must preserve input order")
            judgement = answer.get("judgement")
            if not isinstance(judgement, list) or len(judgement) != len(rubric_weights):
                raise ValueError("verifier judgement length mismatch")
            bool_judgement = [self._parse_verifier_bool(item) for item in judgement]
            final_score = float(
                sum(weight for weight, ok in zip(rubric_weights, bool_judgement) if ok)
            )
            normalized_answers.append(
                {
                    "answer_index": index,
                    "judgement": bool_judgement,
                    "final_score": final_score,
                    "normalized_score": self._normalize_signed_score(final_score, rubric),
                }
            )
        return {
            "schema_version": BATCH_VERIFIER_SCHEMA_VERSION,
            "answers": normalized_answers,
        }

    def _score_rcpc_intervention_items(
        self,
        first: Mapping[str, Any],
        group: Sequence[Dict[str, Any]],
        result: Mapping[str, Any],
        intervention_items: Sequence[Mapping[str, Any]],
    ) -> Dict[int, Dict[int, Dict[str, Any]]]:
        if not intervention_items:
            return {}

        flat_items = []
        flat_texts = []
        for item_index, item in enumerate(intervention_items):
            texts = item.get("texts")
            if not isinstance(texts, list) or not texts:
                texts = [item.get("text", "")]
            for sample_index, text in enumerate(texts):
                flat_items.append((item_index, item, sample_index))
                flat_texts.append(str(text))
        payload = self._verify_answers(
            dict(first),
            result["rubric"],
            flat_texts,
        )
        original_answers = list(result["student_verifier_answers"])
        original_format_valid = list(result.get("student_format_valid", []))
        criteria = list(result["rubric"]["rubrics"])
        answers_by_item: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
        for (item_index, _item, _sample_index), intervened_answer in zip(flat_items, payload["answers"]):
            answers_by_item[int(item_index)].append(intervened_answer)
        restored: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        for item_index, item in enumerate(intervention_items):
            intervened_answers = answers_by_item.get(item_index, [])
            if not intervened_answers:
                continue
            response_index = int(item["response_index"])
            original = original_answers[response_index]
            criterion_effects = {}
            for criterion_index, criterion in enumerate(criteria):
                criterion_id = str(criterion["criterion_id"])
                signed_weight = float(criterion.get("raw_weight", criterion.get("points", 1.0)))
                original_judgement = bool(original["judgement"][criterion_index])
                original_quality = _quality_value(original_judgement, signed_weight)
                intervened_qualities = [
                    _quality_value(bool(answer["judgement"][criterion_index]), signed_weight)
                    for answer in intervened_answers
                ]
                mean_intervened_quality = (
                    sum(intervened_qualities) / len(intervened_qualities)
                    if intervened_qualities
                    else 0.0
                )
                if self.zero_criteria_on_format_error and not original_format_valid[response_index]:
                    original_quality = 0.0
                criterion_effects[criterion_id] = original_quality - mean_intervened_quality
            intervened_scores = [float(answer["final_score"]) for answer in intervened_answers]
            mean_intervened_score = (
                sum(intervened_scores) / len(intervened_scores) if intervened_scores else 0.0
            )
            intervened_texts = item.get("texts")
            if not isinstance(intervened_texts, list) or not intervened_texts:
                intervened_texts = [str(item.get("text", ""))]
            restored[item["batch_index"]][int(item["block_index"])] = {
                "response_index": response_index,
                "batch_index": int(item["batch_index"]),
                "block_index": int(item["block_index"]),
                "block": item["block"],
                "intervened_text": str(item.get("text", "")),
                "intervened_texts": [str(text) for text in intervened_texts],
                "counterfactual_sample_count": len(intervened_answers),
                "criterion_effects": criterion_effects,
                "original_score": float(original["final_score"]),
                "intervened_score": mean_intervened_score,
                "intervened_scores": intervened_scores,
                "score_effect": float(original["final_score"]) - mean_intervened_score,
            }
        return {int(batch_index): dict(value) for batch_index, value in restored.items()}

    def _maybe_print_group(self, first_info: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        # Reuse the existing compact ROPD/RCPC debug printer. The tag remains
        # `[ropd ...]` because downstream log parsing already expects it.
        return super()._maybe_print_group(first_info, result)
