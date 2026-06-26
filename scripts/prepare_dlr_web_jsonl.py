#!/usr/bin/env python3
"""Prepare DLR-Web records for the RCPC/ROPD training pipeline.

The trainer expects local JSONL files with at least:
  - problem: the student prompt
  - solution: the supervised target trajectory
  - answer: the concise final answer

This script converts Attention1115/DLR-Web rows into that format while
preserving the original metadata for debugging and future filtering.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truncate(text: str, max_chars: int) -> str:
    text = _as_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def _matches(row: Dict[str, Any], allowed: Optional[Sequence[str]], key: str) -> bool:
    if not allowed:
        return True
    value = _as_text(row.get(key)).lower()
    return value in {item.lower() for item in allowed}


def _extract_answer_from_response(response: str) -> str:
    match = ANSWER_RE.search(response)
    if match and match.group(1).strip():
        return match.group(1).strip()

    boxed_matches = BOXED_RE.findall(response)
    if boxed_matches:
        return boxed_matches[-1].strip()

    after_think = THINK_CLOSE_RE.split(response, maxsplit=1)
    if len(after_think) == 2:
        tail_lines = [line.strip() for line in after_think[1].splitlines() if line.strip()]
        if tail_lines:
            return tail_lines[-1]

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize_solution(response: str, final_answer: str) -> str:
    """Normalize DLR-Web thinking traces into `<think>...</think><answer>...</answer>`."""
    response = _as_text(response)
    answer = _as_text(final_answer) or _extract_answer_from_response(response)

    think_match = THINK_RE.search(response)
    if think_match:
        think = think_match.group(1).strip()
    else:
        close_match = THINK_CLOSE_RE.search(response)
        if close_match:
            think = response[: close_match.start()].strip()
        else:
            think = ANSWER_RE.sub("", response).strip()

    if not think:
        think = "I solve the problem using the provided facts and reasoning constraints."
    if not answer:
        answer = "N/A"

    return f"<think>\n{think}\n</think><answer>{answer}</answer>"


def build_problem(row: Dict[str, Any], args: argparse.Namespace) -> str:
    parts = [
        "Solve the following multidisciplinary reasoning problem.",
        "Use the provided context and design logic when they are relevant.",
        "Return exactly one response in the format:",
        "<think>...</think><answer>...</answer>",
        "",
    ]

    discipline = _as_text(row.get("discipline"))
    difficulty = _as_text(row.get("difficulty"))
    question_type = _as_text(row.get("type"))
    if discipline:
        parts.append(f"[Discipline]\n{discipline}\n")
    if difficulty:
        parts.append(f"[Difficulty]\n{difficulty}\n")
    if question_type:
        parts.append(f"[Question Type]\n{question_type}\n")

    if args.include_original_document:
        original_document = _truncate(row.get("original_document"), args.max_document_chars)
        if original_document:
            parts.append(f"[Source Document]\n{original_document}\n")

    if args.include_design_logic:
        design_logic = _truncate(row.get("design_logic"), args.max_design_logic_chars)
        if design_logic:
            parts.append(f"[Design Logic]\n{design_logic}\n")

    question = _truncate(row.get("question"), args.max_question_chars)
    parts.append(f"[Question]\n{question}")
    return "\n".join(parts).strip()


def convert_row(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    final_answer = _as_text(row.get("final_answer"))
    return {
        "problem": build_problem(row, args),
        "solution": normalize_solution(_as_text(row.get("response")), final_answer),
        "answer": final_answer or _extract_answer_from_response(_as_text(row.get("response"))),
        "question": _as_text(row.get("question")),
        "reference_answer": _as_text(row.get("reference_answer")),
        "final_answer": final_answer,
        "response": _as_text(row.get("response")),
        "original_document": _as_text(row.get("original_document")),
        "design_logic": _as_text(row.get("design_logic")),
        "discipline": _as_text(row.get("discipline")),
        "difficulty": _as_text(row.get("difficulty")),
        "type": _as_text(row.get("type")),
    }


def iter_filtered_rows(dataset: Iterable[Dict[str, Any]], args: argparse.Namespace) -> Iterator[Dict[str, Any]]:
    for row in dataset:
        row = dict(row)
        if not _matches(row, args.discipline, "discipline"):
            continue
        if not _matches(row, args.difficulty, "difficulty"):
            continue
        if not _matches(row, args.question_type, "type"):
            continue
        if args.require_final_answer and not _as_text(row.get("final_answer")):
            continue
        if args.require_response and not _as_text(row.get("response")):
            continue
        yield row


def load_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    load_kwargs: Dict[str, Any] = {}
    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir

    if args.streaming:
        if args.max_samples <= 0:
            raise ValueError("--streaming requires --max-samples to avoid unbounded in-memory collection.")
        dataset = load_dataset(args.dataset, split=args.split, streaming=True, **load_kwargs)
        rows = iter_filtered_rows(dataset, args)
        records: List[Dict[str, Any]] = []
        for row in rows:
            records.append(convert_row(row, args))
            if len(records) >= args.max_samples:
                break
        return records

    dataset = load_dataset(args.dataset, split=args.split, **load_kwargs)
    if args.discipline or args.difficulty or args.question_type or args.require_final_answer or args.require_response:
        dataset = dataset.filter(
            lambda row: _matches(row, args.discipline, "discipline")
            and _matches(row, args.difficulty, "difficulty")
            and _matches(row, args.question_type, "type")
            and (not args.require_final_answer or bool(_as_text(row.get("final_answer"))))
            and (not args.require_response or bool(_as_text(row.get("response"))))
        )

    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    return [convert_row(row, args) for row in dataset]


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Attention1115/DLR-Web")
    parser.add_argument("--split", default="sample", help="DLR-Web split, e.g. sample or full.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--streaming", action="store_true", help="Stream from Hugging Face for bounded sampling.")
    parser.add_argument("--max-samples", type=int, default=10000, help="<=0 means all rows for non-streaming mode.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--train-output", default="data/dlr_web/train.jsonl")
    parser.add_argument("--val-output", default="data/dlr_web/val.jsonl")
    parser.add_argument("--discipline", action="append", help="Optional exact discipline filter; repeatable.")
    parser.add_argument("--difficulty", action="append", help="Optional exact difficulty filter; repeatable.")
    parser.add_argument("--question-type", action="append", help="Optional exact type filter; repeatable.")
    parser.add_argument("--include-original-document", action="store_true")
    parser.add_argument("--include-design-logic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-document-chars", type=int, default=3000)
    parser.add_argument("--max-design-logic-chars", type=int, default=2000)
    parser.add_argument("--max-question-chars", type=int, default=6000)
    parser.add_argument("--require-final-answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-response", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    records = load_records(args)
    if not records:
        raise RuntimeError("No records matched the requested filters.")

    random.Random(args.seed).shuffle(records)
    val_size = min(max(0, args.val_size), max(0, len(records) - 1))
    val_records = records[:val_size]
    train_records = records[val_size:]

    write_jsonl(Path(args.train_output), train_records)
    write_jsonl(Path(args.val_output), val_records)
    print(f"Wrote {len(train_records)} train records to {args.train_output}")
    print(f"Wrote {len(val_records)} val records to {args.val_output}")


if __name__ == "__main__":
    main()
