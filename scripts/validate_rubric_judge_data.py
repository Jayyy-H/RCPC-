#!/usr/bin/env python3
"""Validate JSONL files for fixed-rubric RCPC training."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate RaR-style rubric JSONL for RCPC rubric-judge training.")
    parser.add_argument("files", nargs="+", help="JSONL file(s) to validate.")
    parser.add_argument("--max-examples", type=int, default=20, help="Maximum bad examples to print.")
    return parser.parse_args()


def load_line(line: str) -> Mapping[str, Any]:
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        item = ast.literal_eval(line)
    if not isinstance(item, Mapping):
        raise ValueError("line is not a JSON object")
    return item


def validate_record(record: Mapping[str, Any]) -> None:
    question = str(record.get("question") or "").strip()
    if not question:
        raise ValueError("missing non-empty `question`")
    answer = str(record.get("answer") or record.get("reference_answer") or record.get("final_answer") or "").strip()
    if not answer:
        raise ValueError("missing `answer`, `reference_answer`, or `final_answer`")
    rubric = record.get("rubric")
    if isinstance(rubric, str):
        try:
            rubric = json.loads(rubric)
        except json.JSONDecodeError:
            rubric = ast.literal_eval(rubric)
    if not isinstance(rubric, list) or not rubric:
        raise ValueError("missing non-empty list `rubric`")
    for index, item in enumerate(rubric, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"rubric[{index}] is not an object")
        if not str(item.get("description") or item.get("criterion") or "").strip():
            raise ValueError(f"rubric[{index}] missing description")
        try:
            float(item.get("weight", item.get("points")))
        except (TypeError, ValueError):
            raise ValueError(f"rubric[{index}] missing numeric weight")


def validate_file(path: Path, max_examples: int) -> int:
    total = 0
    bad = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                validate_record(load_line(line))
            except Exception as exc:
                if len(bad) < max_examples:
                    bad.append((line_no, f"{type(exc).__name__}: {exc}"))
    print(f"{path}: total={total}, bad={len(bad)}")
    for line_no, message in bad:
        print(f"  line {line_no}: {message}")
    return len(bad)


def main() -> int:
    args = parse_args()
    bad_total = 0
    for filename in args.files:
        path = Path(filename).expanduser()
        if not path.exists():
            print(f"{path}: missing file", file=sys.stderr)
            bad_total += 1
            continue
        bad_total += validate_file(path, args.max_examples)
    return 1 if bad_total else 0


if __name__ == "__main__":
    raise SystemExit(main())

