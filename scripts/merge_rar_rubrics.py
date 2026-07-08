#!/usr/bin/env python3
"""Merge RaR-Science rubric field back into a QA/CoT JSONL file.

The downloaded RaR-Science records contain fields such as:
  question, reference_answer, question_source, rubric, rubric_list, rubric_count

Some processed training files only keep:
  question, cot, answer

This script uses the question text as the join key and copies the structured
rubric field from the raw JSONL into the processed training JSONL. Extra fields
can still be copied explicitly with --copy-fields when needed.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


DEFAULT_COPY_FIELDS = ("rubric",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy RaR-Science rubric fields from raw JSONL into a processed training JSONL by matching question."
    )
    parser.add_argument("--raw", required=True, help="Raw RaR-Science JSONL containing rubric fields.")
    parser.add_argument("--input", required=True, help="Processed JSONL to augment, e.g. {question,cot,answer}.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--raw-question-key", default="question", help="Question key in the raw JSONL.")
    parser.add_argument("--input-question-key", default="question", help="Question key in the processed JSONL.")
    parser.add_argument(
        "--copy-fields",
        nargs="+",
        default=list(DEFAULT_COPY_FIELDS),
        help="Fields copied from raw records. Default: rubric.",
    )
    parser.add_argument(
        "--strict-question-match",
        action="store_true",
        help="Use exact stripped question text instead of whitespace/tag normalized question text.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write unmatched input records unchanged instead of failing.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite fields that already exist in the processed input records.",
    )
    parser.add_argument(
        "--max-missing-examples",
        type=int,
        default=10,
        help="How many unmatched examples to print in the summary.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("first", "last", "error"),
        default="first",
        help="How to handle duplicate normalized questions in raw JSONL.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Useful for quick debug dumps that were written as Python repr.
                record = ast.literal_eval(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: expected an object, got {type(record).__name__}")
            yield line_no, record


_QUESTION_TAG_RE = re.compile(r"<question>\s*(.*?)\s*</question>", flags=re.DOTALL | re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


def normalize_question(value: Any, *, strict: bool) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    if strict:
        return text
    match = _QUESTION_TAG_RE.search(text)
    if match:
        text = match.group(1)
    text = text.replace("\u00a0", " ")
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def build_raw_index(
    raw_path: Path,
    *,
    question_key: str,
    copy_fields: List[str],
    strict_question_match: bool,
    duplicate_policy: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int], int]:
    index: Dict[str, Dict[str, Any]] = {}
    source_lines: Dict[str, int] = {}
    duplicate_count = 0

    for line_no, record in load_jsonl(raw_path):
        key = normalize_question(record.get(question_key), strict=strict_question_match)
        if not key:
            continue
        missing_copy_fields = [field for field in copy_fields if field not in record]
        if missing_copy_fields:
            raise KeyError(f"{raw_path}:{line_no}: missing copy fields {missing_copy_fields}")

        payload = {field: record[field] for field in copy_fields}
        if key in index:
            duplicate_count += 1
            if duplicate_policy == "error":
                first_line = source_lines[key]
                raise ValueError(
                    f"duplicate question after normalization: raw lines {first_line} and {line_no}; "
                    f"question={key[:200]!r}"
                )
            if duplicate_policy == "first":
                continue
        index[key] = payload
        source_lines[key] = line_no

    return index, source_lines, duplicate_count


def merge_records(args: argparse.Namespace) -> None:
    raw_path = Path(args.raw).expanduser()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    if not raw_path.exists():
        raise FileNotFoundError(f"raw file not found: {raw_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_index, raw_source_lines, duplicate_count = build_raw_index(
        raw_path,
        question_key=args.raw_question_key,
        copy_fields=args.copy_fields,
        strict_question_match=args.strict_question_match,
        duplicate_policy=args.duplicate_policy,
    )

    total = 0
    matched = 0
    missing: List[Tuple[int, str]] = []
    overwritten = 0
    added_field_count = 0

    with output_path.open("w", encoding="utf-8") as writer:
        for line_no, record in load_jsonl(input_path):
            total += 1
            key = normalize_question(record.get(args.input_question_key), strict=args.strict_question_match)
            payload = raw_index.get(key)
            if payload is None:
                missing.append((line_no, key))
                if not args.allow_missing:
                    continue
            else:
                matched += 1
                for field, value in payload.items():
                    if args.no_overwrite and field in record:
                        continue
                    if field in record:
                        overwritten += 1
                    else:
                        added_field_count += 1
                    record[field] = value
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    if missing and not args.allow_missing:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass
        examples = "\n".join(
            f"  input line {line_no}: {question[:300]!r}"
            for line_no, question in missing[: args.max_missing_examples]
        )
        raise RuntimeError(
            "Some processed records could not be matched to raw rubrics. "
            "No output file was kept. Use --allow-missing to write unmatched records unchanged, "
            "or inspect question normalization.\n"
            f"matched={matched}, missing={len(missing)}, total={total}\n"
            f"Missing examples:\n{examples}"
        )

    print("Merge complete.")
    print(f"  raw_file: {raw_path}")
    print(f"  input_file: {input_path}")
    print(f"  output_file: {output_path}")
    print(f"  raw_unique_questions: {len(raw_index)}")
    print(f"  raw_duplicate_questions: {duplicate_count}")
    print(f"  input_total: {total}")
    print(f"  matched: {matched}")
    print(f"  missing: {len(missing)}")
    print(f"  added_fields: {added_field_count}")
    print(f"  overwritten_fields: {overwritten}")
    if missing:
        print("  missing_examples:")
        for line_no, question in missing[: args.max_missing_examples]:
            print(f"    input line {line_no}: {question[:300]!r}")


def main() -> int:
    args = parse_args()
    try:
        merge_records(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
