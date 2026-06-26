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
case "${PYTORCH_CUDA_ALLOC_CONF:-}" in
    *expandable_segments:True*|*expandable_segments:true*)
        unset PYTORCH_CUDA_ALLOC_CONF
        ;;
esac
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
#export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export VERL_DISABLE_FLASH_ATTN_CE="${VERL_DISABLE_FLASH_ATTN_CE:-1}"
export VLLM_USE_NCCL_SYMM_MEM="${VLLM_USE_NCCL_SYMM_MEM:-0}"
export ROPD_SYNC_AFTER_BACKWARD="${ROPD_SYNC_AFTER_BACKWARD:-1}"
export ROPD_SYNC_AFTER_OPTIM="${ROPD_SYNC_AFTER_OPTIM:-1}"
#下方为训练batch合法性检查 我默认设置了关闭
export ROPD_VALIDATE_ACTOR_BATCH="${ROPD_VALIDATE_ACTOR_BATCH:-0}"
export ROPD_SYNC_VLLM_PHASES="${ROPD_SYNC_VLLM_PHASES:-1}"
export ROPD_SKIP_BAD_SAMPLES="${ROPD_SKIP_BAD_SAMPLES:-true}"
export ROPD_MAX_SAMPLE_RETRIES="${ROPD_MAX_SAMPLE_RETRIES:-16}"
#MODEL_PATH="${MODEL_PATH:-/mnt/bn/yangmin-priv/czh/checkpoints/valley/valleylite_ecom_vl_7B_Gthinker_omni_data_v121_150k_opensourse_150k/}"
#MODEL_PATH="${MODEL_PATH:-/mnt/bn/chenhaobo-va-data/lrj/checkpoints/valleylite-7b-ipr-cot-sft-0524/final/}"
MODEL_PATH="${MODEL_PATH:-/mnt/bn/chenhaobo-va-data/lrj/checkpoints/valleylite-7b-ipr-cot-sft-0607/checkpoint-20/}"
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

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
    )
fi
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-2}"

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
    config=./examples/ipr-grpo/ipr-valleylite-7B-ROPD-GRPO.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.ropd_model=${ROPD_MODEL} \
	trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
	trainer.val_before_train=false \
	data.val_batch_size=${VAL_BATCH_SIZE} \
	trainer.max_val_batches=${MAX_VAL_BATCHES} \
	trainer.experiment_name=valleylite_ropd_fromsft20\
    "${EXTRA_ARGS[@]}" \
    "$@"
