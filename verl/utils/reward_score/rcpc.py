"""Utilities for RCPC action anchoring and span-level credit shaping.

This module is intentionally model-agnostic. It consumes a decoded response,
the generated response token ids, and the policy's per-token entropy computed
while old log probabilities are recomputed.
"""

import math
import re
import statistics
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


_REASONING_BLOCK_RE = re.compile(
    r"<(?:reasoning|think)\b[^>]*>(.*?)</(?:reasoning|think)>",
    re.DOTALL | re.IGNORECASE,
)
_EDGE_STRUCTURAL_TAG_RE = re.compile(r"\s*</?(?:reasoning|think|answer)\b[^>]*>\s*", re.IGNORECASE)
_KEY_CONTENT_TOKEN_RE = re.compile(
    r"[a-z]+(?:['’-][a-z]+)*|\d+(?:\.\d+)?|[\u4e00-\u9fff]",
    re.IGNORECASE,
)
_KEY_CONTENT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "answer",
        "are",
        "as",
        "at",
        "be",
        "because",
        "but",
        "by",
        "check",
        "conclude",
        "finally",
        "for",
        "from",
        "hence",
        "however",
        "if",
        "in",
        "is",
        "it",
        "let",
        "next",
        "now",
        "of",
        "on",
        "or",
        "result",
        "so",
        "step",
        "substituting",
        "than",
        "that",
        "the",
        "then",
        "therefore",
        "this",
        "thus",
        "to",
        "using",
        "wait",
        "we",
        "which",
        "with",
    }
)
_FORMULA_FRAGMENT_RE = re.compile(
    r"[a-z0-9.]+(?:\s*[=+*/^<>≈≠≤≥∝]\s*[a-z0-9.]+)+",
    re.IGNORECASE,
)
_PROTECTED_SPAN_PATTERNS = (
    re.compile(r"```.*?```", re.DOTALL),
    re.compile(r"(?<!`)`[^`\n]+`(?!`)"),
    re.compile(r"\$\$.*?\$\$", re.DOTALL),
    re.compile(
        r"(?<!\\)\$(?![\s$])(?:\\.|[^$\\])+?(?<!\\)\$(?![\w$])",
        re.DOTALL,
    ),
    re.compile(r"\\\(.*?\\\)", re.DOTALL),
    re.compile(r"\\\[.*?\\\]", re.DOTALL),
    re.compile(
        r"\\begin\{(?P<environment>[^{}]+)\}.*?\\end\{(?P=environment)\}",
        re.DOTALL,
    ),
)
_SEMANTIC_BOUNDARY_PATTERNS = (
    (3, r"[\n;；]"),
    (3, r"(?<!\d\.)(?<=[.!?。！？])\s+"),
    (3, r"(?m)(?=^\s*(?:Step\s*)?\d+[.)]\s+)"),
    (2, r"(?=</?(?:reasoning|think|answer)\b)"),
    (1, r"(?:(?<!\d)[,，]|[,，](?!\d))"),
)
_DISCOURSE_MARKER_RE = re.compile(
    r"\b(?:therefore|however|so|if|then|because|but|thus|hence|next|now|wait|check|conclude|finally|let's|let us|using|substituting)\b",
    re.IGNORECASE,
)
_SOFT_CAP_ONLY_BOUNDARY_PATTERNS = (
    (2, r"(?:(?<!\d)[:：]|[:：](?!\d))"),
)


def _merge_character_ranges(ranges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[List[int]] = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([int(start), int(end)])
        else:
            merged[-1][1] = max(merged[-1][1], int(end))
    return [(start, end) for start, end in merged]


def _protected_character_ranges(text: str) -> List[Tuple[int, int]]:
    """Return regions whose internal punctuation is not an action boundary."""
    ranges: List[Tuple[int, int]] = []
    for pattern in _PROTECTED_SPAN_PATTERNS:
        ranges.extend((match.start(), match.end()) for match in pattern.finditer(text))

    # Parenthesized expressions often contain formula arguments or local
    # qualifications. Keep their punctuation internal while still allowing a
    # boundary immediately after the closing delimiter.
    matching_open = {")": "(", "]": "[", "}": "{"}
    stack: List[Tuple[str, int]] = []
    for index, char in enumerate(text):
        if index > 0 and text[index - 1] == "\\":
            continue
        if char in "([{":
            stack.append((char, index))
        elif char in matching_open and stack and stack[-1][0] == matching_open[char]:
            _opening, start = stack.pop()
            ranges.append((start, index + 1))
    return _merge_character_ranges(ranges)


def _position_is_protected(position: int, ranges: Sequence[Tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in ranges)


def _semantic_boundaries(
    text: str,
    *,
    include_soft_cap_only: bool = False,
) -> List[Tuple[int, int]]:
    """Collect legal ``(character boundary, priority)`` pairs."""
    protected_ranges = _protected_character_ranges(text)
    priorities: Dict[int, int] = {}
    patterns = list(_SEMANTIC_BOUNDARY_PATTERNS)
    if include_soft_cap_only:
        patterns.extend(_SOFT_CAP_ONLY_BOUNDARY_PATTERNS)
    for priority, pattern in patterns:
        for match in re.finditer(pattern, text):
            if _position_is_protected(match.start(), protected_ranges):
                continue
            boundary = match.end() if match.end() > match.start() else match.start()
            if 0 < boundary < len(text):
                priorities[boundary] = max(priorities.get(boundary, 0), int(priority))

    # A discourse word is a boundary only at an actual clause edge. Matching
    # every occurrence would fragment phrases such as "the next conclusion"
    # or "is therefore equal". Punctuation/newline remains part of the prior
    # action while the marker stays attached to the following clause.
    for match in _DISCOURSE_MARKER_RE.finditer(text):
        position = match.start()
        if _position_is_protected(position, protected_ranges):
            continue
        prefix = text[:position].rstrip(" \t")
        if not prefix or prefix[-1] not in ",，;；.!?。！？:：\n":
            continue
        if 0 < position < len(text):
            priorities[position] = max(priorities.get(position, 0), 1)
    return sorted(priorities.items())


def build_token_offsets(
    tokenizer: Any,
    token_ids: Sequence[int],
    decoded_text: Optional[str] = None,
) -> List[Tuple[int, int]]:
    """Map generated token positions to decoded character spans.

    Fast tokenizers can recover all offsets in one linear pass. We only trust
    that path when re-tokenization exactly reproduces the non-special token
    ids; otherwise the original prefix-decoding implementation is retained as
    a correctness-preserving fallback.
    """
    ids = [int(token_id) for token_id in token_ids]
    if not ids:
        return []

    text = decoded_text
    if text is None:
        text = tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    special_ids = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", [])}
    content_ids = [token_id for token_id in ids if token_id not in special_ids]
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        encoded_ids = encoded["input_ids"]
        encoded_offsets = encoded["offset_mapping"]
        if encoded_ids and isinstance(encoded_ids[0], (list, tuple)):
            encoded_ids = encoded_ids[0]
            encoded_offsets = encoded_offsets[0]
        encoded_ids = [int(token_id) for token_id in encoded_ids]
        encoded_offsets = [tuple(map(int, offset)) for offset in encoded_offsets]
        if encoded_ids == content_ids and len(encoded_offsets) == len(content_ids):
            offsets: List[Tuple[int, int]] = []
            content_index = 0
            cursor = 0
            for token_id in ids:
                if token_id in special_ids:
                    offsets.append((cursor, cursor))
                    continue
                start, end = encoded_offsets[content_index]
                offsets.append((start, end))
                cursor = end
                content_index += 1
            return offsets
    except Exception:
        pass

    # Preserve historical behavior when decode/encode normalization is not
    # exactly reversible.
    offsets: List[Tuple[int, int]] = []
    prev_text = ""
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
    start = 0
    boundaries = [boundary for boundary, _priority in _semantic_boundaries(working_text)]
    for boundary in sorted(set(boundaries) | {len(working_text)}):
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


def _compact_key_content(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    normalized = re.sub(r"</?(?:reasoning|think|answer)\b[^>]*>", " ", normalized)
    return "".join(
        char
        for char in normalized
        if char.isalnum() or "\u4e00" <= char <= "\u9fff" or char in "=+*/^<>≈≠≤≥∝√%"
    )


def _key_content_terms(text: str) -> List[str]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    normalized = re.sub(r"</?(?:reasoning|think|answer)\b[^>]*>", " ", normalized)
    terms: List[str] = []
    seen = set()
    for token in _KEY_CONTENT_TOKEN_RE.findall(normalized):
        token = token.lower()
        is_cjk = len(token) == 1 and "\u4e00" <= token <= "\u9fff"
        is_number = token[0].isdigit()
        if not is_cjk and not is_number:
            if len(token) < 3 or token in _KEY_CONTENT_STOPWORDS:
                continue
        if token not in seen:
            seen.add(token)
            terms.append(token)

    for match in _FORMULA_FRAGMENT_RE.finditer(normalized):
        formula = _compact_key_content(match.group(0))
        if len(formula) >= 3 and formula not in seen:
            seen.add(formula)
            terms.append(formula)
    return terms


def measure_deleted_content_reappearance(
    deleted_text: str,
    regenerated_suffix: str,
    *,
    min_key_term_recall: float = 0.6,
) -> Dict[str, Any]:
    """Measure conservative lexical reappearance of a deleted reasoning block.

    This is intentionally a no-model diagnostic. It ignores generic reasoning
    connectives, rewards an exact normalized phrase match, and otherwise
    requires multiple content terms to reappear. Semantic paraphrases can be
    missed, so the result should be interpreted as a lower-bound proxy for
    prefix-regeneration self-repair rather than a semantic equivalence test.
    """
    source_terms = _key_content_terms(deleted_text)
    target_terms = set(_key_content_terms(regenerated_suffix))
    compact_source = _compact_key_content(deleted_text)
    compact_target = _compact_key_content(regenerated_suffix)
    exact_phrase_match = bool(
        source_terms
        and len(compact_source) >= 8
        and compact_source in compact_target
    )
    matched_terms = [term for term in source_terms if term in target_terms]
    recall = len(matched_terms) / len(source_terms) if source_terms else 0.0

    if len(source_terms) >= 3:
        required_matches = max(2, math.ceil(len(source_terms) * float(min_key_term_recall)))
        term_match = len(matched_terms) >= required_matches
    elif len(source_terms) == 2:
        term_match = len(matched_terms) == 2
    elif len(source_terms) == 1:
        only_term = source_terms[0]
        term_match = len(only_term) >= 6 and only_term in target_terms
    else:
        term_match = False

    return {
        "reappeared": bool(exact_phrase_match or term_match),
        "key_term_recall": float(recall),
        "exact_phrase_match": bool(exact_phrase_match),
        "key_term_count": len(source_terms),
        "matched_key_term_count": len(matched_terms),
    }


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
    """Apply a semantic soft cap without cutting an indivisible action.

    ``max_action_tokens`` is a target, not a hard slicing interval. We first
    use the last legal boundary before the target; if none exists, we allow the
    next legal boundary up to twice the target. A span with no such boundary is
    preserved intact so formulas and single inference clauses cannot be split
    at arbitrary token positions.
    """
    if max_action_tokens <= 0:
        return list(spans)

    def token_count(start: int, end: int) -> int:
        return len(_overlapping_tokens(offsets, start, end))

    def append_piece(output: List[Tuple[int, int, str]], start: int, end: int) -> None:
        piece_start, piece_end, piece = _trim_edge_structural_tags(
            response_text[start:end], start
        )
        if piece:
            display = piece.rstrip(",，;；").strip() or piece
            output.append((piece_start, piece_end, display))

    capped: List[Tuple[int, int, str]] = []
    for char_start, char_end, action_text in spans:
        token_indices = _overlapping_tokens(offsets, char_start, char_end)
        if len(token_indices) <= max_action_tokens:
            capped.append((char_start, char_end, action_text))
            continue

        local_text = response_text[char_start:char_end]
        legal_boundaries = [
            char_start + boundary
            for boundary, _priority in _semantic_boundaries(
                local_text, include_soft_cap_only=True
            )
        ]
        cursor = char_start
        while token_count(cursor, char_end) > max_action_tokens:
            candidates = [
                boundary
                for boundary in legal_boundaries
                if cursor < boundary < char_end
            ]
            before_target = [
                boundary
                for boundary in candidates
                if 0 < token_count(cursor, boundary) <= max_action_tokens
                and token_count(boundary, char_end) > 0
            ]
            if before_target:
                cut = max(before_target)
            else:
                after_target = [
                    boundary
                    for boundary in candidates
                    if max_action_tokens
                    < token_count(cursor, boundary)
                    <= 2 * max_action_tokens
                    and token_count(boundary, char_end) > 0
                ]
                cut = min(after_target) if after_target else None

            if cut is None:
                append_piece(capped, cursor, char_end)
                cursor = char_end
                break
            append_piece(capped, cursor, cut)
            cursor = cut

        if cursor < char_end:
            append_piece(capped, cursor, char_end)
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


def build_group_candidates(
    responses: Sequence[Mapping[str, Any]],
    *,
    top_actions: int,
    top_blocks: int,
    min_action_chars: int,
    max_action_chars: int,
    max_action_tokens: int,
    min_robust_denom: float,
    min_anchor_z: float,
) -> List[Dict[str, Any]]:
    """Build one candidate pool shared by every trajectory in a prompt group.

    Robust entropy normalization remains response-local, but the Top-K action
    and block caps are applied once to the complete group. This keeps a method
    budget ``B_group`` from accidentally becoming ``B_group`` per response.
    """
    outputs: List[Dict[str, Any]] = []
    ranked_actions: List[Tuple[float, int, Mapping[str, Any]]] = []
    for response_index, response in enumerate(responses):
        actions = score_actions(
            str(response.get("response_text", "")),
            response.get("response_token_offsets") or [],
            response.get("response_token_entropies") or [],
            min_action_chars=min_action_chars,
            max_action_chars=max_action_chars,
            max_action_tokens=max_action_tokens,
            min_robust_denom=min_robust_denom,
        )
        outputs.append({"actions": actions, "top_actions": [], "candidate_blocks": []})
        for action in actions:
            ranked_actions.append(
                (float(action["uncertainty_robust_z"]), response_index, action)
            )

    ranked_actions.sort(key=lambda item: item[0], reverse=True)
    selected_actions = ranked_actions[: max(0, int(top_actions))]
    selected_ids_by_response: Dict[int, List[int]] = {}
    for _score, response_index, action in selected_actions:
        selected_ids_by_response.setdefault(response_index, []).append(int(action["action_index"]))
        outputs[response_index]["top_actions"].append(action)

    ranked_blocks: List[Tuple[float, int, Dict[str, Any]]] = []
    for response_index, output in enumerate(outputs):
        actions = output["actions"]
        candidate_action_ids = selected_ids_by_response.get(response_index, [])
        if not actions or not candidate_action_ids:
            continue
        response_blocks = aggregate_blocks(
            actions,
            candidate_action_ids,
            top_blocks=len(actions),
            min_anchor_z=min_anchor_z,
        )
        for block in response_blocks:
            ranked_blocks.append(
                (float(block.get("anchor_robust_z", 0.0)), response_index, block)
            )

    ranked_blocks.sort(key=lambda item: item[0], reverse=True)
    for _score, response_index, block in ranked_blocks[: max(0, int(top_blocks))]:
        outputs[response_index]["candidate_blocks"].append(block)
    return outputs


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


def paired_effect_statistics(
    factual_values: Sequence[float],
    control_values: Sequence[float],
    *,
    pair_validity: Optional[Sequence[bool]] = None,
    variance_prior: float = 0.0,
) -> Dict[str, Any]:
    """Estimate a paired factual-minus-control effect and its uncertainty.

    Invalid factual/control pairs are excluded instead of being converted into
    semantic failures. This keeps malformed counterfactual generations from
    creating a dense, artificial effect across every rubric criterion.
    """
    if len(factual_values) != len(control_values):
        raise ValueError(
            "paired RCPC samples must have equal sizes: factual={} control={}".format(
                len(factual_values), len(control_values)
            )
        )
    if not factual_values:
        raise ValueError("paired RCPC effect requires at least one sample per arm")
    if pair_validity is None:
        pair_validity = [True] * len(factual_values)
    elif len(pair_validity) != len(factual_values):
        raise ValueError(
            "paired RCPC validity mask must match sample size: validity={} samples={}".format(
                len(pair_validity), len(factual_values)
            )
        )
    paired_differences = [
        float(factual) - float(control)
        for factual, control, valid in zip(factual_values, control_values, pair_validity)
        if bool(valid)
    ]
    if paired_differences:
        effect = sum(paired_differences) / len(paired_differences)
        sample_variance = _sample_variance(paired_differences)
        estimator_variance = sample_variance / len(paired_differences)
        # A tiny paired sample can look spuriously certain when every observed
        # difference is identical. Treat variance_prior as a weak prior on one
        # paired outcome and use it only as a floor; it therefore vanishes as
        # the number of valid pairs grows and never overwrites larger empirical
        # uncertainty.
        prior_floor = max(0.0, float(variance_prior)) / len(paired_differences)
        estimator_variance = max(estimator_variance, prior_floor)
    else:
        effect = 0.0
        sample_variance = 0.0
        estimator_variance = 0.0
    return {
        "effect": effect,
        "paired_differences": paired_differences,
        "sample_variance": sample_variance,
        "estimator_variance": estimator_variance,
        "standard_error": math.sqrt(max(0.0, estimator_variance)),
        "total_pair_count": len(factual_values),
        "valid_pair_count": len(paired_differences),
        "invalid_pair_count": len(factual_values) - len(paired_differences),
    }


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
    eps: float,
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    """Calibrate paired effects using their own estimator uncertainty.

    No across-block variance is used: equal non-zero effects must remain
    informative rather than being collapsed to zero. The reliability factor is
    an effect-to-uncertainty ratio computed independently for each block and
    criterion.
    """
    if not intervention_effects:
        return {}, {
            "rcpc/calibrated_effect_abs_mean": 0.0,
            "rcpc/effect_reliability_mean": 0.0,
            "rcpc/effect_nonzero_ratio": 0.0,
            "rcpc/effect_standard_error_mean": 0.0,
        }

    block_indices = {int(block.get("block_index", -1)) for block in blocks}
    calibrated: Dict[int, Dict[str, float]] = {}
    abs_values = []
    reliabilities = []
    standard_errors = []
    raw_nonzero = []
    for block_index, payload in intervention_effects.items():
        block_index = int(block_index)
        if block_index not in block_indices:
            continue
        criterion_effects = payload.get("criterion_effects", {}) or {}
        criterion_variances = payload.get("criterion_variances", {}) or {}
        for criterion_id, raw_effect in criterion_effects.items():
            criterion_id = str(criterion_id)
            tau_hat = float(raw_effect)
            variance = max(0.0, float(criterion_variances.get(criterion_id, 0.0)))
            signal_power = tau_hat * tau_hat
            if signal_power <= eps and variance <= eps:
                reliability = 0.0
            else:
                reliability = signal_power / (signal_power + variance + eps)
            tau_bar = max(-1.0, min(1.0, tau_hat)) * reliability
            calibrated.setdefault(block_index, {})[criterion_id] = tau_bar
            abs_values.append(abs(tau_bar))
            reliabilities.append(reliability)
            standard_errors.append(math.sqrt(variance))
            raw_nonzero.append(1.0 if abs(tau_hat) > eps else 0.0)

    metrics = {
        "rcpc/calibrated_effect_abs_mean": sum(abs_values) / len(abs_values) if abs_values else 0.0,
        "rcpc/effect_reliability_mean": (
            sum(reliabilities) / len(reliabilities) if reliabilities else 0.0
        ),
        "rcpc/effect_nonzero_ratio": sum(raw_nonzero) / len(raw_nonzero) if raw_nonzero else 0.0,
        "rcpc/effect_standard_error_mean": (
            sum(standard_errors) / len(standard_errors) if standard_errors else 0.0
        ),
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
    eps: float = 1e-6,
) -> Tuple[List[float], Dict[str, float]]:
    length = max(0, int(response_length))
    token_advantages = [0.0] * length
    if length <= 0:
        return token_advantages, {
            "rcpc/candidate_block_count": 0.0,
            "rcpc/nonzero_blocks": 0.0,
            "rcpc/token_coverage": 0.0,
            "rcpc/transport_lambda": float(transport_lambda),
            "rcpc/conservation_error": 0.0,
            "rcpc/advantage_delta_abs_mean": 0.0,
            "rcpc/advantage_delta_l1_ratio": 0.0,
            "rcpc/advantage_cosine_to_baseline": 1.0,
            "rcpc/transport_kl": 0.0,
            "rcpc/candidate_multiplier_mean": 1.0,
            "rcpc/background_multiplier_mean": 1.0,
            "rcpc/calibrated_effect_abs_mean": 0.0,
            "rcpc/effect_reliability_mean": 0.0,
            "rcpc/effect_nonzero_ratio": 0.0,
            "rcpc/effect_standard_error_mean": 0.0,
        }

    units = _build_transport_units(length, blocks)
    calibrated_effects, calibration_metrics = _calibrate_causal_potentials(
        blocks=blocks,
        intervention_effects=intervention_effects,
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
            "rcpc/candidate_block_count": float(
                sum(1 for unit in units if unit.get("kind") == "candidate")
            ),
            "rcpc/nonzero_blocks": 0.0,
            "rcpc/token_coverage": 1.0 if fallback_to_full_response else 0.0,
            "rcpc/transport_lambda": float(transport_lambda),
            "rcpc/conservation_error": 0.0,
            "rcpc/advantage_delta_abs_mean": 0.0,
            "rcpc/advantage_delta_l1_ratio": 0.0,
            "rcpc/advantage_cosine_to_baseline": 1.0,
            "rcpc/transport_kl": 0.0,
            "rcpc/candidate_multiplier_mean": 1.0,
            "rcpc/background_multiplier_mean": 1.0,
            **calibration_metrics,
        }

    point_total = sum(points for _, _, points in active_criteria)
    unit_lengths = [max(1, int(unit["token_end"]) - int(unit["token_start"]) + 1) for unit in units]
    total_unit_tokens = sum(unit_lengths)
    base_distribution = [unit_length / total_unit_tokens for unit_length in unit_lengths]
    causal_units = 0
    transport_kls = []
    candidate_multipliers = []
    background_multipliers = []

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
        transport_kls.append(
            sum(
                transported_mass * math.log(max(transported_mass, eps) / max(base_mass, eps))
                for base_mass, transported_mass in zip(base_distribution, transported_distribution)
            )
        )
        for kind in ("candidate", "background"):
            indices = [index for index, unit in enumerate(units) if unit.get("kind") == kind]
            base_total = sum(base_distribution[index] for index in indices)
            if base_total <= 0.0:
                continue
            transported_total = sum(transported_distribution[index] for index in indices)
            multiplier = transported_total / base_total
            if kind == "candidate":
                candidate_multipliers.append(multiplier)
            else:
                background_multipliers.append(multiplier)
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
    baseline_advantages = [target_advantage] * len(token_advantages)
    advantage_deltas = [
        value - baseline
        for value, baseline in zip(token_advantages, baseline_advantages)
    ]
    delta_abs_mean = sum(abs(value) for value in advantage_deltas) / len(advantage_deltas)
    baseline_l1 = sum(abs(value) for value in baseline_advantages)
    transported_l1 = sum(abs(value) for value in token_advantages)
    delta_l1 = sum(abs(value) for value in advantage_deltas)
    # Symmetric relative L1 change. Unlike delta / ||baseline||_1, this stays
    # well-defined when criterion advantages cancel and the scalar baseline is
    # near zero. The value is bounded in [0, 2].
    symmetric_l1_scale = 0.5 * (baseline_l1 + transported_l1)
    delta_l1_ratio = delta_l1 / max(symmetric_l1_scale, eps)
    delta_l1_ratio = max(0.0, min(2.0, delta_l1_ratio))
    dot = sum(value * baseline for value, baseline in zip(token_advantages, baseline_advantages))
    value_norm = math.sqrt(sum(value * value for value in token_advantages))
    baseline_norm = math.sqrt(sum(value * value for value in baseline_advantages))
    if value_norm <= eps or baseline_norm <= eps:
        cosine = 1.0 if delta_abs_mean <= eps else 0.0
    else:
        cosine = max(-1.0, min(1.0, dot / (value_norm * baseline_norm)))
    nonzero_block_indices = {
        int(block_index)
        for block_index, criterion_map in calibrated_effects.items()
        if any(abs(float(value)) > eps for value in criterion_map.values())
    }
    metrics = {
        "rcpc/candidate_block_count": float(sum(1 for unit in units if unit.get("kind") == "candidate")),
        "rcpc/nonzero_blocks": float(len(nonzero_block_indices)),
        "rcpc/token_coverage": float(coverage),
        "rcpc/causal_units": float(causal_units),
        "rcpc/transport_lambda": float(transport_lambda),
        "rcpc/conservation_error": float(conservation_error),
        "rcpc/advantage_delta_abs_mean": float(delta_abs_mean),
        "rcpc/advantage_delta_l1_ratio": float(delta_l1_ratio),
        "rcpc/advantage_cosine_to_baseline": float(cosine),
        "rcpc/transport_kl": sum(transport_kls) / len(transport_kls) if transport_kls else 0.0,
        "rcpc/candidate_multiplier_mean": (
            sum(candidate_multipliers) / len(candidate_multipliers) if candidate_multipliers else 1.0
        ),
        "rcpc/background_multiplier_mean": (
            sum(background_multipliers) / len(background_multipliers) if background_multipliers else 1.0
        ),
        **calibration_metrics,
    }
    return token_advantages, metrics
