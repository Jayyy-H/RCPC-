#!/usr/bin/env bash
set -euo pipefail

set -x

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_PATH="${MODEL_PATH:-/mnt/bn/chenhaobo-va-data/lrj/checkpoints/valleylite-7b-answer-binary-sft/final}"
INPUT_FILE="${INPUT_FILE:-/mnt/bn/fengshuyang-bytenas-10t/jiangzishang/mllm/data_flywheel/result/grpo_data/test_clean.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-${PWD}/result_answer_binary.jsonl}"

PROMPT_KEY="${PROMPT_KEY:-problem}"
TARGET_KEY="${TARGET_KEY:-solution}"
YES_TOKEN_TEXT="${YES_TOKEN_TEXT:-Yes}"
NO_TOKEN_TEXT="${NO_TOKEN_TEXT:-No}"
MAX_PIXELS="${MAX_PIXELS:-100352}"
MIN_PIXELS="${MIN_PIXELS:-50176}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
PRECISION="${PRECISION:-bf16}"
DEVICE="${DEVICE:-}"
GPU_DEVICES="${GPU_DEVICES:-}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LIMIT="${LIMIT:-}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_RANK="${SHARD_RANK:-0}"
SKIP_ERRORS="${SKIP_ERRORS:-false}"
BASELINE_SCORE_FALLBACK="${BASELINE_SCORE_FALLBACK:-none}"
AUTO_MERGE_CHECKPOINT="${AUTO_MERGE_CHECKPOINT:-true}"
FALLBACK_TO_FINAL="${FALLBACK_TO_FINAL:-false}"
PARTIAL_MERGE_ON_FAILURE="${PARTIAL_MERGE_ON_FAILURE:-true}"
MAX_SHARD_RETRIES="${MAX_SHARD_RETRIES:-1}"
LOAD_TRUNCATED_IMAGES="${LOAD_TRUNCATED_IMAGES:-true}"

export TRUST_REMOTE_CODE
export LOAD_TRUNCATED_IMAGES

build_cmd() {
  local output_file="$1"
  local num_shards="$2"
  local shard_rank="$3"
  local device="$4"

  cmd=(
    python3 scripts/inference_ipr_binary_logits_jsonl.py
    --model_path "${MODEL_PATH}"
    --input_file "${INPUT_FILE}"
    --output_file "${output_file}"
    --prompt_key "${PROMPT_KEY}"
    --target_key "${TARGET_KEY}"
    --yes_token_text "${YES_TOKEN_TEXT}"
    --no_token_text "${NO_TOKEN_TEXT}"
    --max_pixels "${MAX_PIXELS}"
    --min_pixels "${MIN_PIXELS}"
    --precision "${PRECISION}"
    --attn_implementation "${ATTN_IMPLEMENTATION}"
    --batch_size "${BATCH_SIZE}"
    --num_shards "${num_shards}"
    --shard_rank "${shard_rank}"
    --baseline_score_fallback "${BASELINE_SCORE_FALLBACK}"
  )

  if [[ "${TRUST_REMOTE_CODE}" == "true" || "${TRUST_REMOTE_CODE}" == "1" ]]; then
    cmd+=(--trust_remote_code)
  fi

  if [[ -n "${device}" ]]; then
    cmd+=(--device "${device}")
  fi

  if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi

  if [[ "${SKIP_ERRORS}" == "true" || "${SKIP_ERRORS}" == "1" ]]; then
    cmd+=(--skip_errors)
  fi

  if [[ "${AUTO_MERGE_CHECKPOINT}" == "true" || "${AUTO_MERGE_CHECKPOINT}" == "1" ]]; then
    cmd+=(--auto_merge_checkpoint)
  fi

  if [[ "${FALLBACK_TO_FINAL}" == "true" || "${FALLBACK_TO_FINAL}" == "1" ]]; then
    cmd+=(--fallback_to_final)
  fi

  if [[ "${LOAD_TRUNCATED_IMAGES}" == "true" || "${LOAD_TRUNCATED_IMAGES}" == "1" ]]; then
    cmd+=(--load_truncated_images)
  fi
}

if [[ -n "${GPU_DEVICES}" ]]; then
  IFS=',' read -ra gpu_list <<< "${GPU_DEVICES}"
  num_gpus="${#gpu_list[@]}"
  mkdir -p "$(dirname "${OUTPUT_FILE}")"
  tmp_dir="$(mktemp -d "${OUTPUT_FILE}.shards.XXXXXX")"
  pids=()
  shard_outputs=()

  launch_shard() {
    local rank="$1"
    local gpu="$2"
    local attempt="$3"
    local shard_output="${shard_outputs[$rank]}"

    echo "[INFO] starting shard rank ${rank}/${num_gpus} on GPU ${gpu}, attempt ${attempt}; output=${shard_output}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      build_cmd "${shard_output}" "${num_gpus}" "${rank}" "cuda:0"
      "${cmd[@]}"
    ) &
    pids[$rank]="$!"
  }

  wait_for_ranks() {
    local failed_name="$1"
    shift
    local -n failed_ref="${failed_name}"
    local rank
    local pid
    local gpu

    failed_ref=()
    for rank in "$@"; do
      pid="${pids[$rank]}"
      gpu="${gpu_list[$rank]}"
      if ! wait "${pid}"; then
        echo "[ERROR] shard rank ${rank}/${num_gpus} on GPU ${gpu} failed; partial output=${shard_outputs[$rank]}" >&2
        failed_ref+=("${rank}")
      fi
    done
  }

  merge_shards() {
    local destination="$1"
    local require_all="$2"
    local remove_tmp_dir="$3"
    local merged_output="${destination}.merge_tmp.$$"
    local merged_lines=0
    local found_shard=0
    local rank
    local shard_output
    local part_lines

    : > "${merged_output}"
    for rank in "${!gpu_list[@]}"; do
      shard_output="${shard_outputs[$rank]}"
      if [[ ! -f "${shard_output}" ]]; then
        if [[ "${require_all}" == "true" ]]; then
          echo "[ERROR] missing shard output: ${shard_output}" >&2
          echo "[ERROR] keeping temporary shard directory for inspection: ${tmp_dir}" >&2
          rm -f "${merged_output}"
          return 1
        fi
        echo "[WARN] skip missing shard rank ${rank}: ${shard_output}" >&2
        continue
      fi

      found_shard=1
      part_lines="$(wc -l < "${shard_output}" | tr -d ' ')"
      echo "[INFO] merging shard rank ${rank}: ${part_lines} lines from ${shard_output}"
      cat "${shard_output}" >> "${merged_output}"
      merged_lines=$((merged_lines + part_lines))
    done

    if [[ "${found_shard}" -eq 0 ]]; then
      echo "[ERROR] no shard outputs found to merge under ${tmp_dir}" >&2
      rm -f "${merged_output}"
      return 1
    fi

    mv "${merged_output}" "${destination}"
    if [[ "${remove_tmp_dir}" == "true" ]]; then
      rm -rf "${tmp_dir}"
      echo "[INFO] merged ${merged_lines} lines into ${destination}; removed shard directory ${tmp_dir}"
    else
      echo "[INFO] merged ${merged_lines} lines into ${destination}; kept shard directory ${tmp_dir}"
    fi
  }

  merge_partial_outputs() {
    if [[ "${PARTIAL_MERGE_ON_FAILURE}" == "true" || "${PARTIAL_MERGE_ON_FAILURE}" == "1" ]]; then
      partial_output="${OUTPUT_FILE}.partial"
      echo "[WARN] merging currently available shard outputs into ${partial_output}" >&2
      merge_shards "${partial_output}" "false" "false" || true
    fi
  }

  terminate_children() {
    local pid
    for pid in "${pids[@]}"; do
      kill "${pid}" 2>/dev/null || true
    done
    for pid in "${pids[@]}"; do
      wait "${pid}" 2>/dev/null || true
    done
  }

  handle_interrupt() {
    echo "[WARN] interrupted; terminating shard processes and saving partial outputs" >&2
    terminate_children
    merge_partial_outputs
    exit 130
  }
  trap handle_interrupt INT TERM

  all_ranks=()
  for rank in "${!gpu_list[@]}"; do
    gpu="${gpu_list[$rank]}"
    shard_output="${tmp_dir}/part_${rank}.jsonl"
    shard_outputs[$rank]="${shard_output}"
    all_ranks+=("${rank}")
    launch_shard "${rank}" "${gpu}" 1
  done

  failed_ranks=()
  wait_for_ranks failed_ranks "${all_ranks[@]}"

  retry=1
  while [[ "${#failed_ranks[@]}" -gt 0 && "${retry}" -le "${MAX_SHARD_RETRIES}" ]]; do
    echo "[WARN] retrying failed shard ranks in parallel: ${failed_ranks[*]} (retry ${retry}/${MAX_SHARD_RETRIES})" >&2
    retry_ranks=("${failed_ranks[@]}")
    for rank in "${retry_ranks[@]}"; do
      gpu="${gpu_list[$rank]}"
      launch_shard "${rank}" "${gpu}" "$((retry + 1))"
    done
    wait_for_ranks failed_ranks "${retry_ranks[@]}"
    retry=$((retry + 1))
  done

  if [[ "${#failed_ranks[@]}" -gt 0 ]]; then
    echo "[ERROR] one or more shards failed; keeping temporary shard directory for inspection: ${tmp_dir}" >&2
    merge_partial_outputs
    exit 1
  fi

  trap - INT TERM
  merge_shards "${OUTPUT_FILE}" "true" "true"
else
  build_cmd "${OUTPUT_FILE}" "${NUM_SHARDS}" "${SHARD_RANK}" "${DEVICE}"
  "${cmd[@]}"
fi
