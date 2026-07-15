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
    _has_strict_cot_format,
    _render_answer_block,
    _render_template,
    _sample_std,
)
from verl.utils.reward_score.rcpc import paired_effect_statistics


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
    format_weight: float = 1.0,
    epsilon: float = 1e-6,
) -> Tuple[List[float], Dict[str, Dict[str, Any]]]:
    rubrics = list(rubric["rubrics"])
    response_count = len(verifier_answers)
    combined = [0.0] * response_count
    active_weight = 0.0
    criterion_stats: Dict[str, Dict[str, Any]] = {}

    for criterion_index, criterion in enumerate(rubrics):
        signed_weight = float(criterion.get("raw_weight", criterion.get("points", 1.0)))
        point_weight = max(0.0, abs(signed_weight))
        values = []
        for response_index, answer in enumerate(verifier_answers):
            judgement = bool(answer["judgement"][criterion_index])
            value = _quality_value(judgement, signed_weight)
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
            "points": point_weight,
        }

    # XML/protocol compliance is an independent deterministic criterion. It
    # must not erase judgeable semantic evidence from every task rubric.
    format_values = [1.0 if bool(value) else 0.0 for value in format_valid]
    format_mean = sum(format_values) / len(format_values) if format_values else 0.0
    format_std = _sample_std(format_values)
    format_weight = max(0.0, float(format_weight))
    if format_std > epsilon and format_weight > 0.0:
        format_advantages = [
            (value - format_mean) / (format_std + epsilon)
            for value in format_values
        ]
        active_weight += format_weight
        for response_index, advantage in enumerate(format_advantages):
            combined[response_index] += format_weight * advantage
    else:
        format_advantages = [0.0] * response_count
    criterion_stats["format"] = {
        "judgements": format_values,
        "advantages": format_advantages,
        "points": format_weight,
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
                format_weight=float(self.format_points_cap),
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

        flat_items = self._build_rcpc_verifier_flat_items(intervention_items)
        answers_by_flat_key, verifier_metrics = self._verify_rcpc_arm_entries(
            first,
            result["rubric"],
            flat_items,
        )
        if isinstance(result, dict):
            result["rcpc_verifier_metrics"] = verifier_metrics
        original_answers = list(result["student_verifier_answers"])
        criteria = list(result["rubric"]["rubrics"])
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
            pair_format_valid = [
                factual_valid and control_valid
                for factual_valid, control_valid in zip(
                    factual_format_valid,
                    control_format_valid,
                )
            ]
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
            for criterion_index, criterion in enumerate(criteria):
                criterion_id = str(criterion["criterion_id"])
                signed_weight = float(criterion.get("raw_weight", criterion.get("points", 1.0)))
                factual_qualities = [
                    _quality_value(bool(answer["judgement"][criterion_index]), signed_weight)
                    for answer in factual_answers
                ]
                control_qualities = [
                    _quality_value(bool(answer["judgement"][criterion_index]), signed_weight)
                    for answer in control_answers
                ]
                stats = paired_effect_statistics(
                    factual_qualities,
                    control_qualities,
                    pair_validity=pair_semantic_valid,
                    variance_prior=self.rcpc_effect_variance_prior,
                )
                criterion_effects[criterion_id] = float(stats["effect"])
                criterion_variances[criterion_id] = float(stats["estimator_variance"])
                criterion_standard_errors[criterion_id] = float(stats["standard_error"])
                criterion_paired_differences[criterion_id] = list(stats["paired_differences"])
            if "format" in result.get("criterion_stats", {}):
                format_stats = paired_effect_statistics(
                    [1.0 if valid else 0.0 for valid in factual_format_valid],
                    [1.0 if valid else 0.0 for valid in control_format_valid],
                    pair_validity=pair_semantic_valid,
                    variance_prior=self.rcpc_effect_variance_prior,
                )
                criterion_effects["format"] = float(format_stats["effect"])
                criterion_variances["format"] = float(format_stats["estimator_variance"])
                criterion_standard_errors["format"] = float(format_stats["standard_error"])
                criterion_paired_differences["format"] = list(
                    format_stats["paired_differences"]
                )
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
                "intervened_text": control_texts[0] if control_texts else "",
                "intervened_texts": control_texts,
                "factual_sample_count": len(factual_answers),
                "control_sample_count": len(control_answers),
                "counterfactual_sample_count": len(control_answers),
                "factual_format_valid": factual_format_valid,
                "control_format_valid": control_format_valid,
                "pair_format_valid": pair_format_valid,
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

    def _maybe_print_group(self, first_info: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        # Reuse the existing compact ROPD/RCPC debug printer. The tag remains
        # `[ropd ...]` because downstream log parsing already expects it.
        return super()._maybe_print_group(first_info, result)
