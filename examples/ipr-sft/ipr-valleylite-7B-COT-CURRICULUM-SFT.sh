#!/usr/bin/env bash
set -euo pipefail

# Stage 2: smooth Long CoT -> short CoT -> answer-only curriculum.
# MODEL_PATH should point to the checkpoint produced by SFT(weight decay)+GRPO(ROPD).

set -x

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the post-GRPO checkpoint}"
TRAIN_FILE="${TRAIN_FILE:?Set TRAIN_FILE to the multi-view JSONL}"
VAL_FILE="${VAL_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/bn/chenhaobo-va-data/lrj/checkpoints/valleylite-7b-cot-curriculum-sft}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/examples/ipr-sft/deepspeed_zero1_offload.json}"

export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NPROC_PER_NODE

export MODEL_PATH TRAIN_FILE VAL_FILE OUTPUT_DIR DEEPSPEED_CONFIG
export PROMPT_KEY="${PROMPT_KEY:-problem}"
export TARGET_KEY="${TARGET_KEY:-solution}"
export CURRICULUM_PRESET="${CURRICULUM_PRESET:-stage2}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-768}"
export MAX_PIXELS="${MAX_PIXELS:-100352}"
export MIN_PIXELS="${MIN_PIXELS:-50176}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1.0}"
export MAX_STEPS="${MAX_STEPS:--1}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export EVAL_STEPS="${EVAL_STEPS:-200}"
export SAVE_STEPS="${SAVE_STEPS:-200}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
export MAX_SAMPLES="${MAX_SAMPLES:-0}"
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
export EVAL_OUTPUT_MODE="${EVAL_OUTPUT_MODE:-ANSWER_ONLY}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-ipr-cot-internalization}"
export RUN_NAME="${RUN_NAME:-valleylite-7b-ipr-cot-curriculum-sft}"
export FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-true}"
export FREEZE_MM_PROJECTOR="${FREEZE_MM_PROJECTOR:-false}"
export SFT_GRADIENT_CHECKPOINTING="${SFT_GRADIENT_CHECKPOINTING:-true}"
export SFT_SKIP_BAD_SAMPLES="${SFT_SKIP_BAD_SAMPLES:-true}"
export SFT_EVAL_SKIP_BAD_SAMPLES="${SFT_EVAL_SKIP_BAD_SAMPLES:-false}"
export SFT_MAX_SAMPLE_RETRIES="${SFT_MAX_SAMPLE_RETRIES:-16}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${PROJECT_ROOT}"
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/ipr_cot_curriculum_sft_train.py "$@"
