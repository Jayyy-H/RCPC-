import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


YES_NO_INT_MAP = {
    "Yes": 1,
    "No": 0,
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0,
    1: 1,
    0: 0,
    True: 1,
    False: 0,
}


def normalize_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    if value in YES_NO_INT_MAP:
        return YES_NO_INT_MAP[value]
    text = str(value).strip()
    if not text:
        return None
    head = text.split("<", 1)[0].strip()
    if head in YES_NO_INT_MAP:
        return YES_NO_INT_MAP[head]
    lowered = head.lower()
    if lowered in YES_NO_INT_MAP:
        return YES_NO_INT_MAP[lowered]
    return None


def get_nested(row: Dict[str, Any], dotted_key: str) -> Any:
    value: Any = row
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def read_jsonl(path: Path, score_key: str, label_keys: List[str]) -> Tuple[np.ndarray, np.ndarray, Counter]:
    scores = []
    labels = []
    sources: Counter = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw_score = get_nested(row, score_key)
            if raw_score is None:
                raise KeyError(f"line {line_no}: missing score key {score_key!r}")

            label = None
            for key in label_keys:
                label = normalize_label(get_nested(row, key))
                if label is not None:
                    break
            if label is None:
                continue

            scores.append(float(raw_score))
            labels.append(label)
            sources[row.get("yes_logit_source", "missing_source")] += 1

    if not scores:
        raise RuntimeError("no valid rows with both score and label were found")

    return np.array(scores, dtype=np.float64), np.array(labels, dtype=np.int64), sources


def prob_to_log10_odds(values: np.ndarray, eps: float) -> np.ndarray:
    clipped = np.clip(values, eps, 1.0 - eps)
    return np.log10(clipped / (1.0 - clipped))


def prob_ticks(eps: float) -> Tuple[List[float], List[str]]:
    probs = [
        eps,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        0.1,
        0.5,
        0.9,
        0.99,
        0.999,
        0.9999,
        0.99999,
        0.999999,
        1.0 - eps,
    ]
    filtered = []
    labels = []
    for p in probs:
        if p < eps or p > 1.0 - eps:
            continue
        if filtered and abs(p - filtered[-1]) < 1e-15:
            continue
        filtered.append(p)
        if p == 0.5:
            labels.append("0.5")
        elif p < 0.001:
            labels.append(f"{p:.0e}")
        elif p > 0.999:
            labels.append(f"{p:.6f}".rstrip("0"))
        else:
            labels.append(f"{p:g}")
    ticks = prob_to_log10_odds(np.array(filtered), eps).tolist()
    return ticks, labels


def print_summary(scores: np.ndarray, labels: np.ndarray, sources: Counter, eps: float) -> None:
    print("n", len(scores), "source", sources)
    for name, mask in [
        ("all", np.ones_like(labels, dtype=bool)),
        ("label0_approve", labels == 0),
        ("label1_reject", labels == 1),
    ]:
        arr = scores[mask]
        if arr.size == 0:
            continue
        print()
        print(name, "n", arr.size)
        print("min/max", float(arr.min()), float(arr.max()))
        print("quantiles", np.quantile(arr, [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]))
        print("near0", int((arr <= eps).sum()), "near1", int((arr >= 1.0 - eps).sum()))


def add_threshold_line(ax: plt.Axes, threshold: Optional[float], eps: float, label: str, color: str) -> None:
    if threshold is None or math.isnan(threshold):
        return
    y = prob_to_log10_odds(np.array([threshold], dtype=np.float64), eps)[0]
    ax.axhline(y, color=color, linestyle="--", linewidth=1.6, label=f"{label}={threshold:g}")


def gaussian_kernel(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0])
    radius = max(1, int(round(4 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def smooth_histogram(values: np.ndarray, bins: int, sigma: float, value_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(values, bins=bins, range=value_range)
    centers = (edges[:-1] + edges[1:]) / 2
    smooth_counts = np.convolve(counts.astype(np.float64), gaussian_kernel(sigma), mode="same")
    return centers, smooth_counts


def transform_scores(scores: np.ndarray, x_scale: str, eps: float) -> np.ndarray:
    if x_scale == "raw":
        return np.clip(scores, 0.0, 1.0)
    if x_scale == "log10_odds":
        return prob_to_log10_odds(scores, eps)
    raise ValueError(f"unknown x_scale: {x_scale}")


def plot_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    output: Path,
    eps: float,
    title: str,
    bins: int,
    smooth_sigma: float,
    x_scale: str,
) -> None:
    x_values = transform_scores(scores, x_scale, eps)
    value_range = (0.0, 1.0) if x_scale == "raw" else (
        float(prob_to_log10_odds(np.array([eps]), eps)[0]),
        float(prob_to_log10_odds(np.array([1.0 - eps]), eps)[0]),
    )

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    for label, name, color in [
        (0, "label 0 approve", "#1f77b4"),
        (1, "label 1 reject", "#d62728"),
    ]:
        arr = x_values[labels == label]
        if arr.size == 0:
            continue
        x, y = smooth_histogram(arr, bins=bins, sigma=smooth_sigma, value_range=value_range)
        ax.plot(x, y, color=color, linewidth=2.3, label=f"{name} (n={arr.size})")
        ax.fill_between(x, y, color=color, alpha=0.18)

    if x_scale == "raw":
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("yes_logit")
    else:
        ticks, tick_labels = prob_ticks(eps)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=35, ha="right")
        ax.set_xlabel("yes_logit on log10-odds scale")

    ax.set_ylabel("smoothed sample count")
    ax.set_title("Smoothed yes_logit count distribution")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    print(f"saved figure: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize extreme yes_logit distributions from an IPR metrics jsonl file."
    )
    parser.add_argument("--file", required=True, help="Input result jsonl file.")
    parser.add_argument("--output", default=None, help="Output PNG path. Defaults to <file>.yes_logit_distribution.png.")
    parser.add_argument("--score_key", default="yes_logit", help="Score field, supports dotted keys.")
    parser.add_argument(
        "--label_keys",
        default="label,gt_label,answer,solution",
        help="Comma-separated label fields to try, supports dotted keys.",
    )
    parser.add_argument("--eps", type=float, default=1e-8, help="Clipping epsilon for log-odds transform.")
    parser.add_argument("--title", default="yes_logit distribution", help="Figure title.")
    parser.add_argument("--bins", type=int, default=300, help="Histogram bin count before smoothing.")
    parser.add_argument("--smooth_sigma", type=float, default=2.5, help="Gaussian smoothing sigma in histogram bins.")
    parser.add_argument(
        "--x_scale",
        choices=("raw", "log10_odds"),
        default="raw",
        help="raw plots yes_logit in [0, 1]; log10_odds expands the near-0 and near-1 tails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.file)
    output_path = Path(args.output) if args.output else Path(f"{args.file}.yes_logit_distribution.png")
    label_keys = [key.strip() for key in args.label_keys.split(",") if key.strip()]

    scores, labels, sources = read_jsonl(input_path, args.score_key, label_keys)
    print_summary(scores, labels, sources, args.eps)
    plot_distribution(
        scores=scores,
        labels=labels,
        output=output_path,
        eps=args.eps,
        title=args.title,
        bins=args.bins,
        smooth_sigma=args.smooth_sigma,
        x_scale=args.x_scale,
    )


if __name__ == "__main__":
    main()
