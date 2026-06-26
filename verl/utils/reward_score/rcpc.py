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


def split_micro_actions(text: str, min_chars: int, max_chars: int) -> List[Tuple[int, int, str]]:
    """Split a response into comma-level micro-sentence reasoning actions."""
    spans: List[Tuple[int, int, str]] = []
    boundaries = set()
    start = 0
    boundary_patterns = [
        r"(?<!\d)[,，](?!\d)",
        r"[\n;；]",
        r"(?<!\d\.)(?<=[.!?。！？])\s+",
        r"(?=\n?\s*(?:Step\s*)?\d+[.)]\s+)",
        r"(?i)(?=\b(?:therefore|however|so|if|then|because|but|thus|hence|next|now|check|conclude|finally|let's|let us)\b)",
    ]
    for pattern in boundary_patterns:
        for match in re.finditer(pattern, text):
            boundary = match.end() if match.end() > match.start() else match.start()
            if 0 < boundary < len(text):
                boundaries.add(boundary)

    for boundary in sorted(boundaries | {len(text)}):
        chunk_start, chunk_end = start, boundary
        if chunk_end <= chunk_start:
            continue
        raw = text[chunk_start:chunk_end]
        stripped = raw.strip()
        if stripped:
            lead = len(raw) - len(raw.lstrip())
            trail = len(raw.rstrip())
            display_text = stripped.rstrip(",，;；").strip() or stripped
            spans.append((chunk_start + lead, chunk_start + trail, display_text))
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
        "check",
        "conclude",
        "finally",
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
        "check",
        "conclude",
        "finally",
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
        left = anchor
        right = anchor
        if anchor - 1 >= 0 and int(actions[anchor - 1]["action_index"]) in candidate_set:
            left = anchor - 1
        if anchor + 1 < len(actions) and int(actions[anchor + 1]["action_index"]) in candidate_set:
            right = anchor + 1

        block_actions = list(actions[left : right + 1])
        action_ids = [int(action["action_index"]) for action in block_actions]
        if any(action_id in used_action_ids for action_id in action_ids):
            continue
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


def build_token_advantages(
    *,
    response_length: int,
    blocks: Sequence[Mapping[str, Any]],
    combined_advantage: float,
    criterion_advantages: Mapping[str, float],
    criterion_points: Mapping[str, float],
    intervention_effects: Optional[Mapping[int, Mapping[str, Any]]] = None,
    fallback_to_full_response: bool = True,
) -> Tuple[List[float], Dict[str, float]]:
    token_advantages = [0.0] * max(0, int(response_length))
    assigned = [False] * len(token_advantages)
    nonzero_blocks = 0
    effect_values = []

    for block in blocks:
        block_index = int(block.get("block_index", -1))
        token_start = int(block.get("token_start", -1))
        token_end = int(block.get("token_end", -1))
        if token_start < 0 or token_end < token_start or token_start >= len(token_advantages):
            continue
        token_end = min(token_end, len(token_advantages) - 1)

        block_credit = 0.0
        block_effect = None
        if intervention_effects is not None:
            block_effect = intervention_effects.get(block_index)
        if block_effect:
            criterion_effects = block_effect.get("criterion_effects", {})
            for criterion_id, effect in criterion_effects.items():
                effect = float(effect)
                if effect == 0:
                    continue
                effect_values.append(abs(effect))
                block_credit += (
                    float(criterion_advantages.get(str(criterion_id), 0.0))
                    * abs(effect)
                    * float(criterion_points.get(str(criterion_id), 1.0))
                )

        if block_credit == 0.0:
            # Fallback before causal intervention is stable: use the response
            # criterion advantage and prioritize stronger uncertainty peaks.
            peak = max(0.0, float(block.get("block_max_robust_z", block.get("anchor_robust_z", 0.0))))
            block_credit = float(combined_advantage) * (1.0 + peak)

        if block_credit == 0.0:
            continue
        nonzero_blocks += 1
        span_len = max(1, token_end - token_start + 1)
        per_token_credit = block_credit / math.sqrt(span_len)
        for token_index in range(token_start, token_end + 1):
            token_advantages[token_index] += per_token_credit
            assigned[token_index] = True

    if fallback_to_full_response and token_advantages and not any(assigned):
        token_advantages = [float(combined_advantage)] * len(token_advantages)
        assigned = [True] * len(token_advantages)

    coverage = sum(1 for value in assigned if value) / len(assigned) if assigned else 0.0
    metrics = {
        "rcpc/nonzero_blocks": float(nonzero_blocks),
        "rcpc/token_coverage": float(coverage),
        "rcpc/intervention_abs_effect_mean": (
            sum(effect_values) / len(effect_values) if effect_values else 0.0
        ),
    }
    return token_advantages, metrics
