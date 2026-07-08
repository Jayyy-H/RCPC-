# RCPC-test

This repository is a cleaned RCPC/Qwen3-4B training copy. It keeps only the
text-only Qwen3 cold-start SFT path and the fixed-rubric judge RCPC/GRPO path.
Legacy vertical-domain training entrypoints and prompts were removed from this copy.

## Stages

1. Cold-start SFT on CoT data:

```bash
bash examples/rcpc-sft/qwen3-4B-COT-SFT.sh
```

2. RCPC/GRPO with fixed rubrics and an external judge/verifier:

```bash
export JUDGE_MODEL="your-judge-model"
export JUDGE_API_KEY="your-api-key"
export JUDGE_BASE_URL="https://your-openai-compatible-endpoint"  # optional
export JUDGE_API_STYLE="responses"                              # responses or chat_completions

bash examples/rcpc-rubric-judge/qwen3-RUBRIC-JUDGE-RCPC-GRPO.sh
```

The default RCPC causal-intervention budget is `32`. Override it with:

```bash
RCPC_BUDGET=32 bash examples/rcpc-rubric-judge/qwen3-RUBRIC-JUDGE-RCPC-GRPO.sh
```

## Expected Data

The SFT stage expects JSONL rows with `question`, `cot`, and `answer`.

The RCPC/GRPO stage expects JSONL rows with:

- `question`
- `answer` or `reference_answer`
- `rubric`: a list of rubric criteria with `title`, `description`, and `weight`

Use `scripts/merge_rar_rubrics.py` if a QA split needs rubric fields copied
from the original RaR-Science JSONL by matching `question`.

## Judge Interface

No GPT key or internal gateway is hardcoded in this copy. The verifier/judge is
selected by environment variables and can be any OpenAI-compatible service:

- `JUDGE_MODEL`
- `JUDGE_API_KEY`
- `JUDGE_BASE_URL`
- `JUDGE_API_STYLE=responses|chat_completions`
