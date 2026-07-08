"""Utilities for RCPC action anchoring and span-level credit shaping.

This module is intentionally model-agnostic. It consumes a decoded response,
the generated response token ids, and a per-token uncertainty signal. In the
training pipeline the uncertainty signal is usually `-old_log_prob`, because
vLLM generation logits are not kept in the rollout batch.
"""

import math
import re
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


_REASONING_BLOCK_RE = re.compile(
    r"<(?:reasoning|think)\b[^>]*>(.*?)</(?:reasoning|think)>",
    re.DOTALL | re.IGNORECASE,
)
_EDGE_STRUCTURAL_TAG_RE = re.compile(r"\s*</?(?:reasoning|think|answer)\b[^>]*>\s*", re.IGNORECASE)


def build_token_offsets(tokenizer: Any, token_ids: Sequence[int]) -> List[Tuple[int, int]]:
    """Map generated token positions to character spans in decoded response text."""
    offsets: List[Tuple[int, int]] = []
    prev_text = ""
    ids = [int(token_id) for token_id in token_ids]
    for index in range(len(ids)):
        prefix_text = tokenizer.decode(
            ids[: index + 1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        offsets.append((len(prev_text), len(prefix_text)))
        prev_text = prefix_text
    return offsets


def _trim_edge_structural_tags(raw: str, abs_start: int) -> Tuple[int, int, str]:
    local_start = 0
    local_end = len(raw)
    while local_start < local_end:
        match = _EDGE_STRUCTURAL_TAG_RE.match(raw, local_start, local_end)
        if match is None:
            break
        local_start = match.end()

    while local_start < local_end:
        tail = raw[local_start:local_end]
        match = re.search(r"\s*</?(?:reasoning|think|answer)\b[^>]*>\s*$", tail, re.IGNORECASE)
        if match is None:
            break
        local_end = local_start + match.start()

    while local_start < local_end and raw[local_start].isspace():
        local_start += 1
    while local_end > local_start and raw[local_end - 1].isspace():
        local_end -= 1

    text = raw[local_start:local_end].strip()
    return abs_start + local_start, abs_start + local_end, text


def split_micro_actions(text: str, min_chars: int, max_chars: int) -> List[Tuple[int, int, str]]:
    """Split a response into comma-level micro-sentence reasoning actions.

    The splitter is deliberately conservative: it cuts natural reasoning
    pivots such as commas, sentence boundaries, bullets, reasoning tags, and
    discourse markers, but keeps short marker-led clauses (for example
    "so the answer is ...") as standalone actions instead of merging them back
    into the previous step. This matches the SFT outputs where important
    self-corrections often begin with "Wait", "So", or "Therefore".
    """
    reasoning_match = _REASONING_BLOCK_RE.search(text)
    if reasoning_match is not None:
        working_text = reasoning_match.group(1)
        base_offset = reasoning_match.start(1)
    else:
        working_text = text
        base_offset = 0

    spans: List[Tuple[int, int, str]] = []
    boundaries = set()
    start = 0
    boundary_patterns = [
        r"(?<!\d)[,，](?!\d)",
        r"[\n;；]",
        r"(?<!\d\.)(?<=[.!?。！？])\s+",
        r"(?=</?(?:reasoning|think|answer)\b)",
        r"(?m)(?=^\s*(?:Step\s*)?\d+[.)]\s+)",
        r"(?i)(?=\b(?:therefore|however|so|if|then|because|but|thus|hence|next|now|wait|check|conclude|finally|let's|let us|using|substituting)\b)",
    ]
    for pattern in boundary_patterns:
        for match in re.finditer(pattern, working_text):
            boundary = match.end() if match.end() > match.start() else match.start()
            if 0 < boundary < len(working_text):
                boundaries.add(boundary)

    for boundary in sorted(boundaries | {len(working_text)}):
        chunk_start, chunk_end = start, boundary
        if chunk_end <= chunk_start:
            continue
        raw = working_text[chunk_start:chunk_end]
        span_start, span_end, stripped = _trim_edge_structural_tags(raw, base_offset + chunk_start)
        if stripped:
            display_text = stripped.rstrip(",，;；").strip() or stripped
            spans.append((span_start, span_end, display_text))
        start = boundary

    singleton_connectives = {
        "so",
        "then",
        "but",
        "however",
        "therefore",
        "thus",
        "hence",
        "if",
        "because",
        "next",
        "now",
        "wait",
        "check",
        "conclude",
        "finally",
        "using",
        "substituting",
    }
    connective_merged: List[Tuple[int, int, str]] = []
    index = 0
    while index < len(spans):
        s, e, span_text = spans[index]
        normalized = span_text.strip().lower().rstrip(",，;；")
        if normalized in singleton_connectives and index + 1 < len(spans):
            _, ne, _ = spans[index + 1]
            merged_text = text[s:ne].strip().rstrip(",，;；").strip()
            connective_merged.append((s, ne, merged_text or text[s:ne].strip()))
            index += 2
        else:
            connective_merged.append((s, e, span_text))
            index += 1
    spans = connective_merged

    protected_short_prefixes = (
        "so",
        "then",
        "but",
        "however",
        "therefore",
        "thus",
        "hence",
        "if",
        "because",
        "next",
        "now",
        "wait",
        "check",
        "conclude",
        "finally",
        "using",
        "substituting",
        "<answer>",
    )

    def is_protected_short_span(span_text: str) -> bool:
        normalized = span_text.strip().lower()
        return any(normalized == prefix or normalized.startswith(prefix + " ") for prefix in protected_short_prefixes)

    merged: List[Tuple[int, int, str]] = []
    for span in spans:
        span_text = span[2]
        is_answer_span = "answer:" in span_text.lower() or "<answer>" in span_text.lower()
        if merged and len(span_text) < min_chars and not is_answer_span and not is_protected_short_span(span_text):
            ps, _, _ = merged[-1]
            _, ne, _ = span
            merged_text = text[ps:ne].strip().rstrip(",，;；").strip()
            merged[-1] = (ps, ne, merged_text or text[ps:ne].strip())
        else:
            merged.append(span)

    final_spans: List[Tuple[int, int, str]] = []
    for s, e, t in merged:
        if len(t) <= max_chars:
            final_spans.append((s, e, t.rstrip(",，;；").strip() or t))
            continue
        local_start = s
        for match in re.finditer(r"(?<=[.!?。！？])\s+", text[s:e]):
            local_end = s + match.end()
            piece = text[local_start:local_end].strip()
            if piece:
                final_spans.append((local_start, local_end, piece.rstrip(",，;；").strip() or piece))
            local_start = local_end
        if local_start < e:
            piece = text[local_start:e].strip()
            if piece:
                final_spans.append((local_start, e, piece.rstrip(",，;；").strip() or piece))
    return final_spans


def _median_abs_deviation(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    median = statistics.median(values)
    return statistics.median([abs(value - median) for value in values])


def _overlapping_tokens(
    offsets: Sequence[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> List[int]:
    return [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > char_start and token_start < char_end and token_end > token_start
    ]


def _cap_spans_by_token_count(
    response_text: str,
    spans: Sequence[Tuple[int, int, str]],
    offsets: Sequence[Tuple[int, int]],
    max_action_tokens: int,
) -> List[Tuple[int, int, str]]:
    if max_action_tokens <= 0:
        return list(spans)
    capped: List[Tuple[int, int, str]] = []
    for char_start, char_end, action_text in spans:
        token_indices = _overlapping_tokens(offsets, char_start, char_end)
        if len(token_indices) <= max_action_tokens:
            capped.append((char_start, char_end, action_text))
            continue
        for offset in range(0, len(token_indices), max_action_tokens):
            chunk = token_indices[offset : offset + max_action_tokens]
            if not chunk:
                continue
            s = max(char_start, offsets[chunk[0]][0])
            e = min(char_end, offsets[chunk[-1]][1])
            piece = response_text[s:e].strip()
            if piece:
                capped.append((s, e, piece.rstrip(",，;；").strip() or piece))
    return capped


def score_actions(
    response_text: str,
    token_offsets: Sequence[Tuple[int, int]],
    token_uncertainties: Sequence[float],
    *,
    min_action_chars: int,
    max_action_chars: int,
    max_action_tokens: int,
    min_robust_denom: float,
    eps: float = 1e-6,
) -> List[Dict[str, Any]]:
    spans = split_micro_actions(response_text, min_chars=min_action_chars, max_chars=max_action_chars)
    spans = _cap_spans_by_token_count(response_text, spans, token_offsets, max_action_tokens)

    actions: List[Dict[str, Any]] = []
    for action_index, (char_start, char_end, text) in enumerate(spans):
        token_indices = _overlapping_tokens(token_offsets, char_start, char_end)
        values = [
            float(token_uncertainties[index])
            for index in token_indices
            if index < len(token_uncertainties)
        ]
        if values:
            top_r = max(1, math.ceil(math.sqrt(len(values))))
            top_values = sorted(values, reverse=True)[:top_r]
            uncertainty_topr_mean = sum(top_values) / len(top_values)
            token_start = min(token_indices)
            token_end = max(token_indices)
        else:
            uncertainty_topr_mean = 0.0
            token_start = -1
            token_end = -1
        actions.append(
            {
                "action_index": action_index,
                "text": text,
                "char_start": char_start,
                "char_end": char_end,
                "token_indices": token_indices,
                "token_start": token_start,
                "token_end": token_end,
                "token_count": len(token_indices),
                "uncertainty_topr_mean": uncertainty_topr_mean,
                "uncertainty_robust_z": 0.0,
            }
        )

    raw_values = [float(action["uncertainty_topr_mean"]) for action in actions]
    median = statistics.median(raw_values) if raw_values else 0.0
    mad = _median_abs_deviation(raw_values)
    std = statistics.pstdev(raw_values) if len(raw_values) > 1 else 0.0
    denom = max(mad, std if mad < min_robust_denom else 0.0, min_robust_denom, eps)
    for action in actions:
        action["uncertainty_robust_z"] = (float(action["uncertainty_topr_mean"]) - median) / denom
    return actions


def aggregate_blocks(
    actions: Sequence[Mapping[str, Any]],
    candidate_action_ids: Sequence[int],
    *,
    top_blocks: int,
    min_anchor_z: float,
) -> List[Dict[str, Any]]:
    candidate_set = {int(action_id) for action_id in candidate_action_ids}
    peak_indices = []
    for index, action in enumerate(actions):
        z = float(action["uncertainty_robust_z"])
        if z < min_anchor_z or int(action["action_index"]) not in candidate_set:
            continue
        left = float(actions[index - 1]["uncertainty_robust_z"]) if index > 0 else float("-inf")
        right = float(actions[index + 1]["uncertainty_robust_z"]) if index + 1 < len(actions) else float("-inf")
        if z >= left and z > right:
            peak_indices.append(index)

    peak_indices = sorted(peak_indices, key=lambda idx: float(actions[idx]["uncertainty_robust_z"]), reverse=True)
    blocks: List[Dict[str, Any]] = []
    used_action_ids = set()
    for anchor in peak_indices:
        anchor_action_id = int(actions[anchor]["action_index"])
        if anchor_action_id in used_action_ids:
            continue
        left = anchor
        right = anchor
        if (
            anchor - 1 >= 0
            and int(actions[anchor - 1]["action_index"]) in candidate_set
            and int(actions[anchor - 1]["action_index"]) not in used_action_ids
        ):
            left = anchor - 1
        if (
            anchor + 1 < len(actions)
            and int(actions[anchor + 1]["action_index"]) in candidate_set
            and int(actions[anchor + 1]["action_index"]) not in used_action_ids
        ):
            right = anchor + 1

        block_actions = list(actions[left : right + 1])
        action_ids = [int(action["action_index"]) for action in block_actions]
        token_indices = [
            token_index
            for action in block_actions
            for token_index in action.get("token_indices", [])
        ]
        z_values = [float(action["uncertainty_robust_z"]) for action in block_actions]
        text = " ".join(str(action["text"]).strip() for action in block_actions if str(action["text"]).strip())
        blocks.append(
            {
                "block_index": len(blocks),
                "anchor_action_id": int(actions[anchor]["action_index"]),
                "action_ids": action_ids,
                "action_start": action_ids[0],
                "action_end": action_ids[-1],
                "text": text,
                "char_start": min(int(action["char_start"]) for action in block_actions),
                "char_end": max(int(action["char_end"]) for action in block_actions),
                "token_start": min(token_indices) if token_indices else -1,
                "token_end": max(token_indices) if token_indices else -1,
                "token_count": len(token_indices),
                "anchor_robust_z": float(actions[anchor]["uncertainty_robust_z"]),
                "block_mean_robust_z": sum(z_values) / len(z_values) if z_values else 0.0,
                "block_max_robust_z": max(z_values) if z_values else 0.0,
            }
        )
        used_action_ids.update(action_ids)
        if len(blocks) >= top_blocks:
            break
    return blocks


def build_candidates(
    response_text: str,
    token_offsets: Sequence[Tuple[int, int]],
    token_uncertainties: Sequence[float],
    *,
    top_actions: int,
    top_blocks: int,
    min_action_chars: int,
    max_action_chars: int,
    max_action_tokens: int,
    min_robust_denom: float,
    min_anchor_z: float,
) -> Dict[str, Any]:
    actions = score_actions(
        response_text,
        token_offsets,
        token_uncertainties,
        min_action_chars=min_action_chars,
        max_action_chars=max_action_chars,
        max_action_tokens=max_action_tokens,
        min_robust_denom=min_robust_denom,
    )
    top_action_items = sorted(actions, key=lambda action: float(action["uncertainty_robust_z"]), reverse=True)[
        :top_actions
    ]
    blocks = aggregate_blocks(
        actions,
        [int(action["action_index"]) for action in top_action_items],
        top_blocks=top_blocks,
        min_anchor_z=min_anchor_z,
    )
    return {
        "actions": actions,
        "top_actions": top_action_items,
        "candidate_blocks": blocks,
    }


def apply_intervention(response_text: str, block: Mapping[str, Any], mode: str) -> str:
    char_start = int(block.get("char_start", -1))
    char_end = int(block.get("char_end", -1))
    if char_start < 0 or char_end <= char_start or char_end > len(response_text):
        return response_text
    if mode == "remove":
        replacement = ""
    elif mode == "neutral":
        replacement = " [The reasoning step here is omitted.] "
    else:
        replacement = " [RCPC_MASKED_REASONING_ACTION] "
    return response_text[:char_start] + replacement + response_text[char_end:]


def _sample_variance(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _softmax_from_log_weights(log_weights: Sequence[float]) -> List[float]:
    if not log_weights:
        return []
    max_log_weight = max(log_weights)
    exp_values = [math.exp(value - max_log_weight) for value in log_weights]
    denom = sum(exp_values)
    if denom <= 0:
        return [1.0 / len(exp_values)] * len(exp_values)
    return [value / denom for value in exp_values]


def _build_transport_units(
    response_length: int,
    blocks: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Build a non-overlapping token partition for CAT.

    Candidate blocks keep their original identity. Any uncovered generated
    tokens become neutral units with causal potential 0, which lets CAT
    conserve response-level advantage while leaving non-causal regions close to
    the original uniform allocation.
    """
    length = max(0, int(response_length))
    if length <= 0:
        return []

    candidate_units = []
    for block in blocks:
        token_start = int(block.get("token_start", -1))
        token_end = int(block.get("token_end", -1))
        if token_start < 0 or token_end < token_start or token_start >= length:
            continue
        candidate_units.append(
            {
                "kind": "candidate",
                "block_index": int(block.get("block_index", -1)),
                "token_start": max(0, token_start),
                "token_end": min(length - 1, token_end),
                "block": block,
            }
        )

    candidate_units.sort(key=lambda item: (item["token_start"], item["token_end"]))
    units: List[Dict[str, Any]] = []
    cursor = 0
    for unit in candidate_units:
        token_start = max(unit["token_start"], cursor)
        token_end = unit["token_end"]
        if token_end < token_start:
            continue
        if token_start > cursor:
            units.append(
                {
                    "kind": "background",
                    "block_index": None,
                    "token_start": cursor,
                    "token_end": token_start - 1,
                    "block": None,
                }
            )
        units.append({**unit, "token_start": token_start, "token_end": token_end})
        cursor = token_end + 1

    if cursor < length:
        units.append(
            {
                "kind": "background",
                "block_index": None,
                "token_start": cursor,
                "token_end": length - 1,
                "block": None,
            }
        )

    if not units:
        units.append(
            {
                "kind": "background",
                "block_index": None,
                "token_start": 0,
                "token_end": length - 1,
                "block": None,
            }
        )
    return units


def _calibrate_causal_potentials(
    *,
    blocks: Sequence[Mapping[str, Any]],
    intervention_effects: Optional[Mapping[int, Mapping[str, Any]]],
    noise_floor: float,
    eps: float,
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    """Noise-aware empirical-Bayes calibration of local causal effects.

    Current training code estimates local effects by masking/removing a block
    and asking the verifier to rescore the answer. That gives a deletion-effect
    proxy for tau. When future prefix-rollout estimates are added, the same
    interface can pass per-block/per-criterion variances in `criterion_variances`.
    """
    if not intervention_effects:
        return {}, {
            "rcpc/calibrated_effect_abs_mean": 0.0,
            "rcpc/shrinkage_mean": 0.0,
            "rcpc/effect_signal_variance_mean": 0.0,
            "rcpc/effect_noise_mean": 0.0,
        }

    block_indices = {int(block.get("block_index", -1)) for block in blocks}
    by_criterion: Dict[str, List[Tuple[int, float, float]]] = {}
    for block_index, payload in intervention_effects.items():
        block_index = int(block_index)
        if block_index not in block_indices:
            continue
        criterion_effects = payload.get("criterion_effects", {}) or {}
        criterion_variances = payload.get("criterion_variances", {}) or {}
        for criterion_id, raw_effect in criterion_effects.items():
            criterion_id = str(criterion_id)
            tau_hat = float(raw_effect)
            variance = float(criterion_variances.get(criterion_id, noise_floor))
            by_criterion.setdefault(criterion_id, []).append((block_index, tau_hat, max(0.0, variance)))

    calibrated: Dict[int, Dict[str, float]] = {}
    abs_values = []
    shrinkages = []
    signal_variances = []
    noise_values = []

    for criterion_id, entries in by_criterion.items():
        tau_values = [entry[1] for entry in entries]
        noise = [entry[2] for entry in entries]
        observed_variance = _sample_variance(tau_values)
        mean_noise = sum(noise) / len(noise) if noise else 0.0
        signal_variance = max(0.0, observed_variance - mean_noise)
        scale = math.sqrt(max(observed_variance, 0.0)) + eps
        signal_variances.append(signal_variance)
        noise_values.append(mean_noise)

        for block_index, tau_hat, variance in entries:
            shrinkage = signal_variance / (signal_variance + variance + eps)
            tau_eb = shrinkage * tau_hat
            tau_bar = math.tanh(tau_eb / scale) if signal_variance > 0.0 else 0.0
            calibrated.setdefault(block_index, {})[criterion_id] = tau_bar
            abs_values.append(abs(tau_bar))
            shrinkages.append(shrinkage)

    metrics = {
        "rcpc/calibrated_effect_abs_mean": sum(abs_values) / len(abs_values) if abs_values else 0.0,
        "rcpc/shrinkage_mean": sum(shrinkages) / len(shrinkages) if shrinkages else 0.0,
        "rcpc/effect_signal_variance_mean": (
            sum(signal_variances) / len(signal_variances) if signal_variances else 0.0
        ),
        "rcpc/effect_noise_mean": sum(noise_values) / len(noise_values) if noise_values else 0.0,
    }
    return calibrated, metrics


def build_token_advantages(
    *,
    response_length: int,
    blocks: Sequence[Mapping[str, Any]],
    combined_advantage: float,
    criterion_advantages: Mapping[str, float],
    criterion_points: Mapping[str, float],
    intervention_effects: Optional[Mapping[int, Mapping[str, Any]]] = None,
    fallback_to_full_response: bool = True,
    transport_lambda: float = 1.0,
    effect_noise_floor: float = 0.05,
    eps: float = 1e-6,
) -> Tuple[List[float], Dict[str, float]]:
    length = max(0, int(response_length))
    token_advantages = [0.0] * length
    if length <= 0:
        return token_advantages, {
            "rcpc/nonzero_blocks": 0.0,
            "rcpc/token_coverage": 0.0,
            "rcpc/transport_lambda": float(transport_lambda),
            "rcpc/conservation_error": 0.0,
        }

    units = _build_transport_units(length, blocks)
    calibrated_effects, calibration_metrics = _calibrate_causal_potentials(
        blocks=blocks,
        intervention_effects=intervention_effects,
        noise_floor=max(0.0, float(effect_noise_floor)),
        eps=eps,
    )
    if intervention_effects:
        for block_index, criterion_map in calibrated_effects.items():
            payload = intervention_effects.get(block_index)
            if isinstance(payload, dict):
                payload["calibrated_effects"] = dict(criterion_map)

    active_criteria = []
    for criterion_id, advantage in criterion_advantages.items():
        criterion_id = str(criterion_id)
        points = max(0.0, float(criterion_points.get(criterion_id, 0.0)))
        if points <= 0.0:
            continue
        active_criteria.append((criterion_id, float(advantage), points))

    if not active_criteria:
        if fallback_to_full_response:
            token_advantages = [float(combined_advantage)] * length
        return token_advantages, {
            "rcpc/nonzero_blocks": 0.0,
            "rcpc/token_coverage": 1.0 if fallback_to_full_response else 0.0,
            "rcpc/transport_lambda": float(transport_lambda),
            "rcpc/conservation_error": 0.0,
            **calibration_metrics,
        }

    point_total = sum(points for _, _, points in active_criteria)
    unit_lengths = [max(1, int(unit["token_end"]) - int(unit["token_start"]) + 1) for unit in units]
    total_unit_tokens = sum(unit_lengths)
    base_distribution = [unit_length / total_unit_tokens for unit_length in unit_lengths]
    causal_units = 0

    for criterion_id, criterion_advantage, points in active_criteria:
        alpha = points / point_total if point_total > 0.0 else 1.0 / len(active_criteria)
        sign = 1.0 if criterion_advantage > 0 else (-1.0 if criterion_advantage < 0 else 0.0)
        log_weights = []
        for unit, base_mass in zip(units, base_distribution):
            block_index = unit.get("block_index")
            tau_bar = 0.0
            if block_index is not None:
                tau_bar = float(calibrated_effects.get(int(block_index), {}).get(criterion_id, 0.0))
                if abs(tau_bar) > 0.0:
                    causal_units += 1
            log_weights.append(math.log(max(base_mass, eps)) + float(transport_lambda) * sign * tau_bar)

        transported_distribution = _softmax_from_log_weights(log_weights)
        for unit, base_mass, transported_mass in zip(units, base_distribution, transported_distribution):
            if base_mass <= 0.0:
                continue
            unit_advantage = criterion_advantage * transported_mass / base_mass
            for token_index in range(int(unit["token_start"]), int(unit["token_end"]) + 1):
                token_advantages[token_index] += alpha * unit_advantage

    mean_advantage = sum(token_advantages) / len(token_advantages) if token_advantages else 0.0
    target_advantage = float(combined_advantage)
    conservation_error = mean_advantage - target_advantage
    if abs(mean_advantage) > eps and abs(target_advantage) > eps:
        scale = target_advantage / mean_advantage
        token_advantages = [value * scale for value in token_advantages]
        mean_advantage = sum(token_advantages) / len(token_advantages)
        conservation_error = mean_advantage - target_advantage
    elif abs(target_advantage) > eps and fallback_to_full_response:
        token_advantages = [target_advantage] * len(token_advantages)
        mean_advantage = target_advantage
        conservation_error = 0.0

    covered_tokens = set()
    for unit in units:
        if unit.get("kind") == "candidate":
            covered_tokens.update(range(int(unit["token_start"]), int(unit["token_end"]) + 1))
    coverage = len(covered_tokens) / len(token_advantages) if token_advantages else 0.0
    metrics = {
        "rcpc/nonzero_blocks": float(sum(1 for unit in units if unit.get("kind") == "candidate")),
        "rcpc/token_coverage": float(coverage),
        "rcpc/causal_units": float(causal_units),
        "rcpc/transport_lambda": float(transport_lambda),
        "rcpc/conservation_error": float(conservation_error),
        **calibration_metrics,
    }
    return token_advantages, metrics
