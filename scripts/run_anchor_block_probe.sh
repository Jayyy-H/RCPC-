#!/usr/bin/env bash
set -euo pipefail

# Fill these paths before running, or pass them as environment variables:
# MODEL_PATH=/path/to/qwen3 DATA_PATH=/path/to/data.jsonl bash scripts/run_anchor_block_probe.sh
MODEL_PATH="${MODEL_PATH:-}"
DATA_PATH="${DATA_PATH:-}"

OUT_DIR="${OUT_DIR:-runs/anchor_block_probe}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TOP_ACTIONS="${TOP_ACTIONS:-12}"
TOP_BLOCKS="${TOP_BLOCKS:-6}"
DEVICE="${DEVICE:-auto}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Please set MODEL_PATH in scripts/run_anchor_block_probe.sh or pass MODEL_PATH=/path/to/model." >&2
  exit 1
fi

if [[ -z "${DATA_PATH}" ]]; then
  echo "Please set DATA_PATH in scripts/run_anchor_block_probe.sh or pass DATA_PATH=/path/to/data.jsonl." >&2
  exit 1
fi

python scripts/run_anchor_block_probe.py \
  --model-path "${MODEL_PATH}" \
  --data-path "${DATA_PATH}" \
  --out-dir "${OUT_DIR}" \
  --num-samples "${NUM_SAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --top-actions "${TOP_ACTIONS}" \
  --top-blocks "${TOP_BLOCKS}" \
  --device "${DEVICE}"
