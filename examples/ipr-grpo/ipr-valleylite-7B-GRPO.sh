#!/usr/bin/env bash
set -euo pipefail

set -x

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export VERL_PRECOMPUTE_MASTER="${VERL_PRECOMPUTE_MASTER:-1}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
case "${PYTORCH_CUDA_ALLOC_CONF:-}" in
    *expandable_segments:True*|*expandable_segments:true*)
        unset PYTORCH_CUDA_ALLOC_CONF
        ;;
esac
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export VLLM_USE_NCCL_SYMM_MEM="${VLLM_USE_NCCL_SYMM_MEM:-0}"
export ROPD_SYNC_AFTER_BACKWARD="${ROPD_SYNC_AFTER_BACKWARD:-0}"
export ROPD_SYNC_AFTER_OPTIM="${ROPD_SYNC_AFTER_OPTIM:-1}"
export ROPD_SYNC_VLLM_PHASES="${ROPD_SYNC_VLLM_PHASES:-1}"
export ROPD_SKIP_BAD_SAMPLES="${ROPD_SKIP_BAD_SAMPLES:-true}"
export ROPD_MAX_SAMPLE_RETRIES="${ROPD_MAX_SAMPLE_RETRIES:-16}"

MODEL_PATH="${MODEL_PATH:-/mnt/bn/chenhaobo-va-data/lrj/checkpoints/valleylite-7b-ipr-cot-sft-0524/final/}"
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

EXTRA_ARGS=()
if [[ "${GRPO_SMOKE:-1}" == "1" ]]; then
    EXTRA_ARGS+=(
        trainer.max_steps=2
        trainer.logger='["console"]'
        trainer.save_freq=-1
        trainer.test_freq=-1
        data.rollout_batch_size=1
        data.val_batch_size=1
        worker.rollout.n=2
        worker.rollout.tensor_parallel_size=1
    )
fi

if [[ "${GRPO_SMOKE:-1}" != "1" && -z "${WANDB_API_KEY:-}" && "${WANDB_MODE:-}" != "offline" && "${WANDB_MODE:-}" != "disabled" ]]; then
    EXTRA_ARGS+=(trainer.logger='["console"]')
fi

if [[ "${VLLM_ENFORCE_EAGER:-0}" == "1" ]]; then
    EXTRA_ARGS+=(worker.rollout.enforce_eager=true)
fi

if [[ -n "${ACTOR_ATTN_IMPLEMENTATION:-}" ]]; then
    EXTRA_ARGS+=(worker.actor.model.attn_implementation="${ACTOR_ATTN_IMPLEMENTATION}")
fi

python3 -m verl.trainer.main \
    config=./examples/ipr-grpo/ipr-valleylite-7B-GRPO.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.val_before_train=false \
    data.val_batch_size=64 \
    trainer.experiment_name=valleylite_grpo_reward \
    "${EXTRA_ARGS[@]}" \
    "$@"
