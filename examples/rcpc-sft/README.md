# Qwen3 CoT SFT Cold Start

This entrypoint trains a text-only Qwen3 model on generated CoT data before RCPC/ROPD RL. The SFT stage uses CoT data; the later RCPC/ROPD RL stage should use QA-only data.

Default train file:

```bash
/mnt/bn/chenhaobo-va-data/lrj/data/WebInstruct-verified/train_5k_cot.jsonl
```

Expected SFT fields are `question`, `cot`, and `answer`. The SFT script formats
each row as a user prompt containing `<question>...</question>` and an assistant
target containing `<think>...</think><answer>...</answer>`.

The later RCPC/ROPD RL stage should use QA-only data, for example:

```bash
/mnt/bn/chenhaobo-va-data/lrj/data/WebInstruct-verified/rcpc_5k.jsonl
```

That second-stage file should contain `question` and `answer`/`final_answer`/
`reference_answer`; it does not need `cot`, because RCPC uses rubric/criterion
reward rather than cross-entropy CoT alignment.

Smoke test:

```bash
SFT_SMOKE=1 REPORT_TO=none bash examples/rcpc-sft/qwen3-4B-COT-SFT.sh
```

One-epoch SFT:

```bash
mkdir -p /mnt/bn/chenhaobo-va-data/lrj/log/rcpc
nohup bash examples/rcpc-sft/qwen3-4B-COT-SFT.sh \
  > /mnt/bn/chenhaobo-va-data/lrj/log/rcpc/qwen3_sft_01.log 2>&1 &
```

After SFT finishes, start RCPC/GRPO RL from:

```bash
/mnt/bn/chenhaobo-va-data/lrj/checkpoints/rcpc/qwen3-4b-webinstruct-cot-sft/final
```

Example fixed-rubric RCPC/GRPO launch after SFT:

```bash
mkdir -p /mnt/bn/chenhaobo-va-data/lrj/log/rcpc
nohup bash -c 'ROPD_SMOKE=0 JUDGE_MODEL=your-judge-model JUDGE_API_KEY=your-api-key MODEL_PATH=/mnt/bn/chenhaobo-va-data/lrj/checkpoints/rcpc/qwen3-4b-webinstruct-cot-sft/final TRAIN_FILE=/mnt/bn/chenhaobo-va-data/lrj/data/RaR-Science-20k-o3-mini/splits/rl_rubrics_train.jsonl VAL_FILE=/mnt/bn/chenhaobo-va-data/lrj/data/RaR-Science-20k-o3-mini/splits/val_rubrics.jsonl TRAIN_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 N_GPUS_PER_NODE=8 bash examples/rcpc-rubric-judge/qwen3-RUBRIC-JUDGE-RCPC-GRPO.sh' \
  > /mnt/bn/chenhaobo-va-data/lrj/log/rcpc/qwen3_rcpc_01.log 2>&1 &
```
