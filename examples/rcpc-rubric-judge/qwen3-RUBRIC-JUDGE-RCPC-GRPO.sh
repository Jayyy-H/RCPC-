#!/usr/bin/env bash
set -euo pipefail

JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
JUDGE_API_STYLE="${JUDGE_API_STYLE:-responses}"

set -x

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export VERL_PRECOMPUTE_MASTER="${VERL_PRECOMPUTE_MASTER:-1}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-/mnt/bn/chenhaobo-va-data/lrj/log/wandb/rcpc-rubric-judge}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${WANDB_DIR}/data}"
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
export RCPC_WRAP_QUESTION_TAGS="${RCPC_WRAP_QUESTION_TAGS:-true}"

MODEL_PATH="${MODEL_PATH:-/mnt/bn/chenhaobo-va-data/lrj/checkpoints/rcpc/qwen3-4b-RaRScience-sft/final}"
TRAIN_FILE="${TRAIN_FILE:-/mnt/bn/chenhaobo-va-data/lrj/data/RaR-Science-20k-o3-mini/splits/rl_rubrics_train.jsonl}"
VAL_FILE="${VAL_FILE:-/mnt/bn/chenhaobo-va-data/lrj/data/RaR-Science-20k-o3-mini/splits/val_rubrics.jsonl}"
PROMPT_KEY="${PROMPT_KEY:-question}"

export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE_FOR_UPDATE="${MICRO_BATCH_SIZE_FOR_UPDATE:-1}"
MICRO_BATCH_SIZE_FOR_EXPERIENCE="${MICRO_BATCH_SIZE_FOR_EXPERIENCE:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-10}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-1}"
PROJECT_NAME="${PROJECT_NAME:-rcpc-rubric-judge}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_rubric_judge_rcpc}"
SAVE_CHECKPOINT_PATH="${SAVE_CHECKPOINT_PATH:-/mnt/bn/chenhaobo-va-data/lrj/checkpoints/rcpc/qwen3-4b-RUBRIC-JUDGE-RCPC-GRPO/}"
LOAD_CHECKPOINT_PATH="${LOAD_CHECKPOINT_PATH:-null}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"
SAVE_CKPT_LIMIT="${SAVE_CKPT_LIMIT:-20}"
SAVE_SHARD_LIMIT="${SAVE_SHARD_LIMIT:-10}"
VAL_GENERATIONS_TO_LOG_TO_WANDB="${VAL_GENERATIONS_TO_LOG_TO_WANDB:-10}"
RCPC_BUDGET="${RCPC_BUDGET:-32}"
RCPC_DERIVE_CANDIDATES_FROM_BUDGET="${RCPC_DERIVE_CANDIDATES_FROM_BUDGET:-true}"
RCPC_TRANSPORT_LAMBDA="${RCPC_TRANSPORT_LAMBDA:-1.0}"
RCPC_EFFECT_NOISE_FLOOR="${RCPC_EFFECT_NOISE_FLOOR:-0.05}"
RCPC_INTERVENTION_MAX_GROUPS_PER_BATCH="${RCPC_INTERVENTION_MAX_GROUPS_PER_BATCH:--1}"
RCPC_BATCH_COUNTERFACTUAL="${RCPC_BATCH_COUNTERFACTUAL:-true}"
RCPC_COUNTERFACTUAL_SAMPLES="${RCPC_COUNTERFACTUAL_SAMPLES:-2}"
RCPC_COUNTERFACTUAL_BATCH_SIZE="${RCPC_COUNTERFACTUAL_BATCH_SIZE:-128}"
ROPD_MAX_CONCURRENCY="${ROPD_MAX_CONCURRENCY:-16}"
ROPD_VERIFIER_MAX_OUTPUT_TOKENS="${ROPD_VERIFIER_MAX_OUTPUT_TOKENS:-2048}"
ROPD_NUM_EXAMINE="${ROPD_NUM_EXAMINE:-1}"
ROPD_REQUIRE_STRICT_COT_FORMAT="${ROPD_REQUIRE_STRICT_COT_FORMAT:-true}"
ROPD_ZERO_SCORE_ON_FORMAT_ERROR="${ROPD_ZERO_SCORE_ON_FORMAT_ERROR:-true}"
ROPD_ZERO_CRITERIA_ON_FORMAT_ERROR="${ROPD_ZERO_CRITERIA_ON_FORMAT_ERROR:-true}"
ROPD_PRINT_STUDENT_OUTPUTS="${ROPD_PRINT_STUDENT_OUTPUTS:-true}"
ROPD_PRINT_MAX_STUDENT_OUTPUTS="${ROPD_PRINT_MAX_STUDENT_OUTPUTS:-3}"
ROPD_PRINT_VERIFIER_OUTPUTS="${ROPD_PRINT_VERIFIER_OUTPUTS:-false}"
ROPD_RCPC_PRINT_INTERVENTION_SUMMARY="${ROPD_RCPC_PRINT_INTERVENTION_SUMMARY:-true}"
ROPD_RCPC_PRINT_INTERVAL="${ROPD_RCPC_PRINT_INTERVAL:-10}"
ROPD_RCPC_PRINT_MAX_GROUPS="${ROPD_RCPC_PRINT_MAX_GROUPS:-1}"
ROPD_RCPC_PRINT_MAX_BLOCKS="${ROPD_RCPC_PRINT_MAX_BLOCKS:-16}"

mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"

if [[ -z "${JUDGE_MODEL}" ]]; then
    echo "ERROR: JUDGE_MODEL is required. Set it to the verifier/judge model name." >&2
    exit 1
fi

if [[ -z "${JUDGE_API_KEY}" ]]; then
    echo "ERROR: JUDGE_API_KEY is required. Set it to the verifier/judge provider API key." >&2
    exit 1
fi

if [[ "${MODEL_PATH}" == /* && ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "ERROR: MODEL_PATH looks like a local checkpoint but config.json was not found: ${MODEL_PATH}" >&2
    echo "Set MODEL_PATH to the CoT SFT final checkpoint, or explicitly set it to a Hugging Face model id for base-model ablations." >&2
    exit 1
fi

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
        worker.reward.num_examine=1
        worker.reward.ropd_max_concurrency=1
        worker.reward.ropd_rcpc_intervention_max_groups_per_batch=1
        worker.reward.ropd_rcpc_budget=1
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
    config=./examples/rcpc-rubric-judge/qwen3-RUBRIC-JUDGE-RCPC-GRPO.yaml \
    worker.actor.model.model_path="${MODEL_PATH}" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    worker.reward.ropd_model="${JUDGE_MODEL}" \
    worker.reward.ropd_api_key_env=JUDGE_API_KEY \
    worker.reward.ropd_base_url_env=JUDGE_BASE_URL \
    worker.reward.ropd_api_style="${JUDGE_API_STYLE}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.val_before_train=false \
    data.prompt_key="${PROMPT_KEY}" \
    data.rollout_batch_size="${ROLLOUT_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    worker.actor.global_batch_size="${GLOBAL_BATCH_SIZE}" \
    worker.actor.micro_batch_size_per_device_for_update="${MICRO_BATCH_SIZE_FOR_UPDATE}" \
    worker.actor.micro_batch_size_per_device_for_experience="${MICRO_BATCH_SIZE_FOR_EXPERIENCE}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.max_val_batches="${MAX_VAL_BATCHES}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.save_checkpoint_path="${SAVE_CHECKPOINT_PATH}" \
    trainer.load_checkpoint_path="${LOAD_CHECKPOINT_PATH}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_ckpt_limit="${SAVE_CKPT_LIMIT}" \
    trainer.save_shard_limit="${SAVE_SHARD_LIMIT}" \
    trainer.val_generations_to_log_to_wandb="${VAL_GENERATIONS_TO_LOG_TO_WANDB}" \
    worker.reward.ropd_rcpc_budget="${RCPC_BUDGET}" \
    worker.reward.ropd_rcpc_derive_candidates_from_budget="${RCPC_DERIVE_CANDIDATES_FROM_BUDGET}" \
    worker.reward.ropd_rcpc_transport_lambda="${RCPC_TRANSPORT_LAMBDA}" \
    worker.reward.ropd_rcpc_effect_noise_floor="${RCPC_EFFECT_NOISE_FLOOR}" \
    worker.reward.ropd_rcpc_intervention_max_groups_per_batch="${RCPC_INTERVENTION_MAX_GROUPS_PER_BATCH}" \
    worker.reward.ropd_rcpc_batch_counterfactual="${RCPC_BATCH_COUNTERFACTUAL}" \
    worker.reward.ropd_rcpc_counterfactual_samples="${RCPC_COUNTERFACTUAL_SAMPLES}" \
    worker.reward.ropd_rcpc_counterfactual_batch_size="${RCPC_COUNTERFACTUAL_BATCH_SIZE}" \
    worker.reward.num_examine="${ROPD_NUM_EXAMINE}" \
    worker.reward.ropd_require_strict_cot_format="${ROPD_REQUIRE_STRICT_COT_FORMAT}" \
    worker.reward.ropd_zero_score_on_format_error="${ROPD_ZERO_SCORE_ON_FORMAT_ERROR}" \
    worker.reward.ropd_zero_criteria_on_format_error="${ROPD_ZERO_CRITERIA_ON_FORMAT_ERROR}" \
    worker.reward.ropd_print_student_outputs="${ROPD_PRINT_STUDENT_OUTPUTS}" \
    worker.reward.ropd_print_max_student_outputs="${ROPD_PRINT_MAX_STUDENT_OUTPUTS}" \
    worker.reward.ropd_print_verifier_outputs="${ROPD_PRINT_VERIFIER_OUTPUTS}" \
    worker.reward.ropd_max_concurrency="${ROPD_MAX_CONCURRENCY}" \
    worker.reward.ropd_verifier_max_output_tokens="${ROPD_VERIFIER_MAX_OUTPUT_TOKENS}" \
    worker.reward.ropd_rcpc_print_intervention_summary="${ROPD_RCPC_PRINT_INTERVENTION_SUMMARY}" \
    worker.reward.ropd_rcpc_print_interval="${ROPD_RCPC_PRINT_INTERVAL}" \
    worker.reward.ropd_rcpc_print_max_groups="${ROPD_RCPC_PRINT_MAX_GROUPS}" \
    worker.reward.ropd_rcpc_print_max_blocks="${ROPD_RCPC_PRINT_MAX_BLOCKS}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
