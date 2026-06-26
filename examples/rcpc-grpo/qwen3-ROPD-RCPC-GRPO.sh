#!/usr/bin/env bash
set -euo pipefail

export GPT5_4_API_KEY="${GPT5_4_API_KEY:-AmfJ2xJ8ToUQ0Lk0tBPMCVgw50bdtwWu_GPT_AK}"
export GPT5_4_BASE_URL="${GPT5_4_BASE_URL:-https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/responses}"
export ROPD_MODEL="${ROPD_MODEL:-gpt-5.4-2026-03-05}"

set -x

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export VERL_PRECOMPUTE_MASTER="${VERL_PRECOMPUTE_MASTER:-1}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export VLLM_USE_NCCL_SYMM_MEM="${VLLM_USE_NCCL_SYMM_MEM:-0}"
export VERL_DISABLE_FLASH_ATTN_CE="${VERL_DISABLE_FLASH_ATTN_CE:-1}"
export ROPD_SYNC_AFTER_BACKWARD="${ROPD_SYNC_AFTER_BACKWARD:-1}"
export ROPD_SYNC_AFTER_OPTIM="${ROPD_SYNC_AFTER_OPTIM:-1}"
export ROPD_SYNC_VLLM_PHASES="${ROPD_SYNC_VLLM_PHASES:-1}"
export ROPD_VALIDATE_ACTOR_BATCH="${ROPD_VALIDATE_ACTOR_BATCH:-0}"
export ROPD_SKIP_BAD_SAMPLES="${ROPD_SKIP_BAD_SAMPLES:-true}"
export ROPD_MAX_SAMPLE_RETRIES="${ROPD_MAX_SAMPLE_RETRIES:-16}"

MODEL_PATH="${MODEL_PATH:-}"
TRAIN_FILE="${TRAIN_FILE:-}"
VAL_FILE="${VAL_FILE:-}"

if [[ -z "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH is required, e.g. MODEL_PATH=/path/to/Qwen3 bash $0" >&2
    exit 1
fi
if [[ -z "${TRAIN_FILE}" ]]; then
    echo "TRAIN_FILE is required, e.g. TRAIN_FILE=/path/to/train.jsonl bash $0" >&2
    exit 1
fi
if [[ -z "${VAL_FILE}" ]]; then
    echo "VAL_FILE is required, e.g. VAL_FILE=/path/to/val.jsonl bash $0" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-10}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-1}"

EXTRA_ARGS=()
if [[ "${ROPD_SMOKE:-1}" == "1" ]]; then
    EXTRA_ARGS+=(
        trainer.max_steps=2
        trainer.logger='["console"]'
        trainer.save_freq=-1
        trainer.test_freq=-1
        data.rollout_batch_size=1
        data.val_batch_size=1
        worker.rollout.n=2
        worker.rollout.tensor_parallel_size=1
        worker.reward.num_examine=3
        worker.reward.ropd_max_concurrency=1
        worker.reward.ropd_rcpc_intervention_max_groups_per_batch=1
        worker.reward.ropd_rcpc_intervention_max_blocks_per_answer=1
    )
fi

if [[ "${ROPD_SMOKE:-1}" != "1" && -z "${WANDB_API_KEY:-}" && "${WANDB_MODE:-}" != "offline" && "${WANDB_MODE:-}" != "disabled" ]]; then
    EXTRA_ARGS+=(trainer.logger='["console"]')
fi

if [[ "${VLLM_ENFORCE_EAGER:-0}" == "1" ]]; then
    EXTRA_ARGS+=(worker.rollout.enforce_eager=true)
fi

if [[ -n "${ACTOR_ATTN_IMPLEMENTATION:-}" ]]; then
    EXTRA_ARGS+=(worker.actor.model.attn_implementation="${ACTOR_ATTN_IMPLEMENTATION}")
fi

python3 -m verl.trainer.main \
    config=./examples/rcpc-grpo/qwen3-ROPD-RCPC-GRPO.yaml \
    worker.actor.model.model_path="${MODEL_PATH}" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    worker.reward.ropd_model="${ROPD_MODEL}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.val_before_train=false \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    trainer.max_val_batches="${MAX_VAL_BATCHES}" \
    trainer.experiment_name="${EXPERIMENT_NAME:-qwen3_ropd_rcpc}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
