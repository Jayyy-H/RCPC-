#!/usr/bin/env bash
set -euo pipefail

export OPENAI_API_KEY="AmfJ2xJ8ToUQ0Lk0tBPMCVgw50bdtwWu_GPT_AK"
export ROPD_MODEL="${ROPD_MODEL:-gpt-5.4}"
# If your Linux gateway requires a custom endpoint, uncomment and fill this.
# export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
MODEL_PATH=/mnt/bn/yangmin-priv/czh/models/Valley_B7_v3_navit_cot_AutoModel_clean/
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"

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
        worker.reward.num_examine=3
        worker.reward.ropd_max_concurrency=1
    )
fi

python3 -m verl.trainer.main \
    config=./examples/ipr-grpo/ipr-7B-ROPD-GRPO.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.ropd_model=${ROPD_MODEL} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.val_before_train=false \
    data.val_batch_size=128 \
    trainer.experiment_name=sftrl_ropd_reward \
    "${EXTRA_ARGS[@]}" \
    "$@"
