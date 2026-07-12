# RCPC 方法与训练观测 README

## 1. 方法关键部分与代码对应关系

### 1.1 固定 Rubric Judge Reward

当前 `RCPC-test` 使用的是 fixed-rubric judge 版本，不再调用 teacher/rubricator 动态生成 rubric。训练样本需要自带 `rubric` 字段，judge/verifier 只负责对每条 rollout response 在每个 rubric criterion 上做判断。

主要代码：

- `verl/workers/reward/custom.py`
  - `compute_score: rubric_judge_rcpc` 会实例化 `FixedRubricRCPCRewardScorer`。
- `verl/utils/reward_score/rubric_judge_rcpc.py`
  - `_parse_raw_rubric(...)`：解析数据中的 `rubric` 字段。
  - `_format_fixed_criterion(...)`：把正权重/负权重 rubric 转成 verifier 可理解的判断标准。
  - `_quality_value(...)`：把 verifier 的 boolean judgement 转成“质量方向”的 criterion value。
    - 正权重 criterion：`TRUE = 好`。
    - 负权重 pitfall criterion：`TRUE = 踩坑 = 坏`，所以质量值会反向。
  - `_compute_fixed_group_criterion_advantages(...)`：按 criterion 维度计算 group 内 advantage。
  - `_score_group(...)`：对一个 prompt group 内的多条 rollout response 调用 verifier，得到分数、criterion advantage，并触发 RCPC credit shaping。
- `prompts/rcpc_rubric_judge/verifier.txt`
  - verifier/judge 的 prompt 模板。

### 1.2 Micro Action 选取

主要代码：

- `verl/utils/reward_score/rcpc.py`
  - `split_micro_actions(...)`
  - `score_actions(...)`
  - `build_candidates(...)`

当前切分逻辑：

1. 优先只在 `<reasoning>...</reasoning>` 内切分；如果没有 `<reasoning>`，则对整个 response 切分。
2. 切分边界包括：
   - 非数字小数中的逗号 `,` / `，`
   - 换行、分号
   - 句号/问号/感叹号后的空白
   - `<reasoning>`、`<answer>` 等标签边界
   - 列表编号边界
   - reasoning marker，例如 `therefore`、`however`、`so`、`if`、`then`、`because`、`wait`、`check`、`using`、`substituting` 等
3. 对过短 action 做合并，但保护 `so/then/wait/therefore/...` 这类可能承载关键转折的短 action。
4. 对过长 action 再按句子边界拆分。
5. 如果 action token 数超过 `ropd_rcpc_max_action_tokens`，会再切成更短 token span。

Action 打分逻辑：

- token uncertainty 来自 `-old_log_probs`，即模型对当前生成 token 的不确定性。
- 每个 action 的 raw uncertainty 使用长度自适应 Top-R mean：
  - `R = ceil(sqrt(action_token_count))`
  - 取该 action 内最高的 R 个 token uncertainty 求平均。
- 对同一条 response 内所有 action 做 robust normalization：
  - `uncertainty_robust_z = (action_score - median) / denom`
  - `denom` 由 MAD、std、`ropd_rcpc_min_robust_denom` 共同保护，避免分母过小。

### 1.3 候选 Action 与 Block 合并

主要代码：

- `verl/utils/reward_score/rcpc.py`
  - `aggregate_blocks(...)`
  - `build_candidates(...)`
- `verl/utils/reward_score/ropd.py`
  - `_build_rcpc_candidates_for_response(...)`

候选 action：

- `build_candidates(...)` 会按 `uncertainty_robust_z` 从高到低选 `top_actions`。
- 当 `ropd_rcpc_derive_candidates_from_budget=true` 时：
  - `top_actions = 2 * B_group`
  - `top_blocks = B_group`
  - `B_group` 来自 `ropd_rcpc_budget` / `RCPC_BUDGET`

Block 合并规则：

- 只有同时满足以下条件的 action 才能作为 anchor：
  - 它在 top action candidate set 中。
  - `uncertainty_robust_z >= ropd_rcpc_min_anchor_z`
  - 它是局部峰值：`z >= left_z` 且 `z > right_z`
- anchor 的左右邻居只有在也属于 top action candidate set 且没有被其他 block 使用时，才会被并入 block。
- 每个 block 会记录：
  - `anchor_action_id`
  - `action_ids`
  - `token_start/token_end`
  - `char_start/char_end`
  - `anchor_robust_z`
  - `block_mean_robust_z`
  - `block_max_robust_z`
  - `text`

### 1.4 基于 Budget 选择要干预的 Block

主要代码：

- `verl/utils/reward_score/ropd.py`
  - 初始化预算：`__init__` 中读取 `ropd_rcpc_budget`
  - `_build_rcpc_intervention_items(...)`
  - `_run_rcpc_interventions(...)`
  - `_populate_counterfactual_texts(...)`
  - `_score_rcpc_intervention_items(...)`

当前默认：

- `RCPC_BUDGET=32`
- `ropd_rcpc_derive_candidates_from_budget=true`

因此预算含义是每个 prompt group 的核心预算 `B_group`：

- 每个 group 最多保留 `B_group` 个 candidate block。
- 每个 group 最多执行 `B_group` 个 causal intervention。
- action candidate 数由同一个 budget 派生为 `2 * B_group`。

干预 block 的优先级：

```text
selection_priority = response_advantage_scale * block_salience
```

- `response_advantage_scale`：该 response 在各 rubric criterion 上的 advantage 强度，按 criterion 权重聚合。
- `block_salience`：block 的不确定性峰值，通常是 `block_max_robust_z`。
- priority <= 0 的 block 不会进入干预。

干预方式：

- 当前主配置是 `ropd_rcpc_intervention_mode: prefix_regen`。
- 对某个 block 干预时，不是简单删除文本后重新评分，而是：
  1. 截取到该 block 之前的 response prefix。
  2. 让 policy model 从这个 prefix 继续生成后续 reasoning + answer。
  3. 调用 verifier 重新打分。
  4. 用原始分数与反事实分数差估计该 block 的局部因果效应。

反事实采样次数：

- `ropd_rcpc_counterfactual_samples`
- shell 中对应 `RCPC_COUNTERFACTUAL_SAMPLES`
- 当前默认是 `2`。

### 1.5 Criterion Advantage 如何进入 GRPO

主要代码：

- `verl/utils/reward_score/ropd.py`
  - `__call__(...)` 中写入 `data.batch["criterion_advantages"]`
  - `_build_rcpc_token_advantages(...)`
- `verl/utils/reward_score/rcpc.py`
  - `build_token_advantages(...)`
  - `_calibrate_intervention_effects(...)`
- `verl/trainer/core_algos.py`
  - `compute_grpo_outcome_advantage(...)`
- `verl/trainer/ray_trainer.py`
  - `compute_advantage(...)`

逻辑：

1. Verifier 先得到每条 response 的 fixed-rubric score。
2. 按 criterion 维度做 group 内标准化 advantage。
3. 如果 RCPC token advantage 开启，则把 criterion advantage 进一步分配到 candidate block/token span。
4. `core_algos.compute_grpo_outcome_advantage(...)` 检测到 `criterion_advantages` 后，会直接用它作为 token-level advantage，而不是只用最终 scalar reward 做传统 GRPO outcome advantage。

## 2. 训练中重点观察的日志与指标

除了常规的 `reward/final/*`、`val/test_score/mean`、`actor/entropy_loss`、`actor/pg_loss`、`response_length/*`，RCPC 训练尤其要看下面这些。

### 2.1 Reward / Judge 健康度

- `reward/group_ok_ratio`
  - 每个 step 中成功完成 judge/reward 计算的 prompt group 比例。
  - 如果低，说明 verifier 返回格式、数据字段或 reward 解析可能有问题。

- `reward/format_valid_ratio`
  - rollout response 满足严格 `<reasoning>...</reasoning><answer>...</answer>` 格式的比例。
  - 如果长期接近 0，说明 prompt/SFT checkpoint/forced prefix 仍有格式问题。
  - 当前配置中格式错误不会直接把所有 rubric reward 清零，但格式健康度仍然很重要。

- `reward/verifier_raw/mean|max|min`
  - verifier 根据 rubric 直接给出的原始得分分布。
  - 用来判断 judge 是否过严/过松。

- `reward/final_nonzero_ratio`
  - final score 非零 response 比例。
  - 如果 reward mean 很低但 nonzero ratio 不低，说明有区分信号；如果两者都低，需要看格式、judge prompt 或数据质量。

### 2.2 Criterion Advantage 信号

- `criterion_advantage/mean`
  - criterion advantage 的平均值。标准化后通常接近 0。

- `criterion_advantage/std`
  - group 内是否有可学习的区分信号。
  - 太低：同一 group 内 responses 在 rubric 维度几乎没有差异，GRPO/RCPC 学不到太多。
  - 太高：可能是 judge 过于离散、rubric 权重过猛，或 response 质量差异很大。

### 2.3 RCPC Credit Assignment 指标

- `rcpc/enabled`
  - 是否开启 RCPC。

- `rcpc/intervention_enabled`
  - 是否开启因果干预。

- `rcpc/intervention_groups`
  - 当前 batch 是否实际产生过干预。
  - 如果一直是 0，需要检查：
    - `ropd_rcpc_intervention_enabled`
    - `run_rcpc_intervention`
    - `criterion_advantage/std`
    - candidate block 是否为空
    - budget 或 priority 是否把 block 过滤掉

- `rcpc/nonzero_blocks`
  - 每条 response 中被分配到非零 credit 的 candidate block 数。
  - 太低：候选过少、min anchor 太高、action 切太碎/太粗、criterion advantage 没信号。
  - 太高：候选过泛，可能 credit 太分散。

- `rcpc/token_coverage`
  - 有多少 response token 被 candidate block 或 fallback unit 覆盖。
  - 如果接近 0，说明 action/block 没覆盖有效 reasoning。
  - 如果接近 1，说明 credit 可能退化为接近全 response 分配，RCPC 的局部性变弱。

- `rcpc/causal_units`
  - 使用了反事实因果效应的 block/unit 数。
  - 如果 intervention 开启但该值很低，说明实际完成的有效干预少。

- `rcpc/calibrated_effect_abs_mean`
  - 校准后的局部因果效应绝对值均值。
  - 长期为 0：反事实干预没有带来 reward/rubric 判断变化，可能 block 不关键、judge 不敏感，或反事实生成太相似。

- `rcpc/shrinkage_mean`
  - 因果效应校准时的 shrinkage 强度。
  - 越大表示估计更保守；如果过大，RCPC 可能接近退回 criterion advantage。

- `rcpc/effect_signal_variance_mean`
  - 同一 criterion 下不同 block 的因果效应信号方差。
  - 太低表示 block 间没有明显差异。

- `rcpc/effect_noise_mean`
  - 反事实采样带来的噪声估计。
  - 如果噪声很高，可考虑提高 `RCPC_COUNTERFACTUAL_SAMPLES`，但会变慢。

- `rcpc/conservation_error`
  - token/block credit 分配前后的总量守恒误差。
  - 应该接近 0；如果偏大，说明 credit transport 逻辑可能有问题。

- `rcpc/transport_lambda`
  - 当前使用的 transport lambda。
  - 不是结果指标，而是记录当前 credit redistribution 强度。

### 2.4 Reward/Judge/RCPC 耗时指标

这些指标会以 `timing_s/.../sum|mean|max` 形式进入日志/W&B。

- `timing_s/reward/verifier_initial_group`
  - 对原始 rollout responses 做 verifier 打分的耗时。

- `timing_s/rcpc/credit_prepare_group`
  - 构建 candidate action/block、准备 RCPC credit 的耗时。

- `timing_s/rcpc/intervention_score_wall`
  - 干预后的反事实 response 再 verifier 打分的 wall time。

- `timing_s/reward/group_total`
  - 单个 prompt group reward 计算总耗时。

- `timing_s/reward/score_groups_wall`
  - 一个 batch 内所有 group 的 reward scoring wall time。

- `timing_s/reward/deferred_rcpc_wall`
  - 如果使用 deferred RCPC，这里反映延迟执行 RCPC 的耗时。

- `timing_s/reward/build_criterion_tensor`
  - 将 criterion/token advantage 写回训练 tensor 的耗时。

- `timing_s/reward/total`
  - reward manager 总耗时。

- `timing_count/verifier_initial_requests`
  - verifier 初始打分请求数量。

- `timing_count/verifier_initial_answers`
  - verifier 初始打分覆盖的 answer 数。

如果一个 step 特别慢，优先比较：

1. `timing_s/gen`
2. `timing_s/reward/score_groups_wall/sum`
3. `timing_s/rcpc/intervention_score_wall/sum`
4. `timing_s/update_actor`

## 3. 如何从训练日志观察 micro action 和 block 是否合理

### 3.1 默认摘要日志

当前默认会打印：

```text
[ropd rcpc intervention summary]
{
  "step": ...,
  "budget": {...},
  "scores": {...},
  "criterion_advantages": {...},
  "rcpc_metrics": {...},
  "trajectories": [
    {
      "batch_index": ...,
      "trajectory_preview": "...",
      "blocks": [
        {
          "block_index": ...,
          "response_index": ...,
          "action_ids": [...],
          "token_start": ...,
          "token_end": ...,
          "selection_priority": ...,
          "anchor_robust_z": ...,
          "block_mean_robust_z": ...,
          "text": "...",
          "intervened_scores": [...],
          "criterion_effects": {...},
          "score_effect": ...
        }
      ]
    }
  ]
}
```

打印频率由下面参数控制：

- `ROPD_RCPC_PRINT_INTERVENTION_SUMMARY=true`
- `ROPD_RCPC_PRINT_INTERVAL=10`
- `ROPD_RCPC_PRINT_MAX_GROUPS=1`
- `ROPD_RCPC_PRINT_MAX_BLOCKS=16`

配置位置：

- shell: `examples/rcpc-rubric-judge/qwen3-RUBRIC-JUDGE-RCPC-GRPO.sh`
- yaml: `examples/rcpc-rubric-judge/qwen3-RUBRIC-JUDGE-RCPC-GRPO.yaml`

### 3.2 判断 action/block 合理性的 checklist

看 `blocks[*].text`：

- 好的现象：
  - block 是局部 reasoning unit，不是整段 response。
  - block 往往覆盖关键公式、关键判断、关键转换、关键纠错、关键 conclusion bridge。
  - `action_ids` 通常是 1 到 3 个相邻 action，而不是无意义的大段拼接。
  - `token_start/token_end` 覆盖范围适中。

- 可疑现象：
  - block 只是一两个无意义词，例如 “so”、“then”。
  - block 太长，像完整段落。
  - block 总是选最后 answer 标签附近，而不是 reasoning 中的关键步骤。
  - block 文本和 `trajectory_preview` 对不上，说明 token/char offset 可能有 bug。
  - `anchor_robust_z` 很低但仍被选中，说明 `ropd_rcpc_min_anchor_z` 可能太低。

看 `intervened_scores` / `score_effect` / `criterion_effects`：

- 好的现象：
  - 移除或截断某个关键 block 后，某些 rubric criterion 判断发生变化。
  - `score_effect` 对明显关键 block 更大。
  - 不同 block 的 `criterion_effects` 有区分。

- 可疑现象：
  - 所有 block 的 `score_effect` 长期为 0。
  - 所有 block 的 `criterion_effects` 都相同。
  - 反事实生成后的分数比原始还高很多，可能说明原始 block 是错误推理或候选确实抓到了错误点；这不一定是坏事，需要结合文本看。

### 3.3 打开完整候选输出

如果想看完整 `actions/top_actions/candidate_blocks`，可以临时打开：

```bash
worker.reward.ropd_print_rcpc_outputs=true
```

或在 yaml 中设置：

```yaml
worker:
  reward:
    ropd_print_rcpc_outputs: true
```

注意：这个输出会很大，建议只在 smoke/debug 时打开。

## 4. 常用可调超参数

### 4.1 启动脚本中最常调的参数

文件：

- `examples/rcpc-rubric-judge/qwen3-RUBRIC-JUDGE-RCPC-GRPO.sh`

#### `RCPC_BUDGET`

默认：

```bash
RCPC_BUDGET="${RCPC_BUDGET:-32}"
```

含义：

- 每个 prompt group 的 RCPC intervention budget。
- 当 `RCPC_DERIVE_CANDIDATES_FROM_BUDGET=true`：
  - `effective_top_actions = 2 * RCPC_BUDGET`
  - `effective_top_blocks = RCPC_BUDGET`
  - `blocks_per_group = RCPC_BUDGET`

调参建议：

- 训练太慢：先降到 16 或 24。
- `rcpc/intervention_groups` 正常但 `rcpc/calibrated_effect_abs_mean` 长期为 0：单纯增大 budget 未必有用，优先看候选 block 是否合理。
- `rcpc/nonzero_blocks` 太少：可适当增大 budget 或降低 `ropd_rcpc_min_anchor_z`。

#### `RCPC_INTERVENTION_MAX_GROUPS_PER_BATCH`

默认：

```bash
RCPC_INTERVENTION_MAX_GROUPS_PER_BATCH="${RCPC_INTERVENTION_MAX_GROUPS_PER_BATCH:--1}"
```

含义：

- `-1`：当前 batch 内所有 prompt group 都允许做 RCPC 干预。
- `0`：不做干预。
- 正整数：每个 batch 最多只对这么多个 group 做干预。

调参建议：

- 用于控制总成本。
- 如果训练太慢，但不想降低每个 group 的 budget，可以先限制 group 数，例如 `4` 或 `8`。

#### `RCPC_COUNTERFACTUAL_SAMPLES`

默认：

```bash
RCPC_COUNTERFACTUAL_SAMPLES="${RCPC_COUNTERFACTUAL_SAMPLES:-2}"
```

含义：

- 每个被干预 block 做多少次 prefix-regeneration 反事实采样。

调参建议：

- 噪声大：增大到 3 或 4，但耗时会近似线性增加。
- 训练太慢：降到 1。

#### `RCPC_COUNTERFACTUAL_BATCH_SIZE`

默认：

```bash
RCPC_COUNTERFACTUAL_BATCH_SIZE="${RCPC_COUNTERFACTUAL_BATCH_SIZE:-128}"
```

含义：

- prefix-regeneration 反事实请求的批处理 chunk size。

调参建议：

- 太小：vLLM 调用 overhead 大。
- 太大：可能导致显存/调度压力增大。

#### `ROPD_MAX_CONCURRENCY`

默认：

```bash
ROPD_MAX_CONCURRENCY="${ROPD_MAX_CONCURRENCY:-16}"
```

含义：

- verifier/judge 并发请求数。

调参建议：

- verifier 等待时间长：可增大，但要看服务限流。
- 出现 API timeout/限流：降低。

### 4.2 YAML 中的 RCPC 方法参数

文件：

- `examples/rcpc-rubric-judge/qwen3-RUBRIC-JUDGE-RCPC-GRPO.yaml`

#### `ropd_rcpc_derive_candidates_from_budget`

默认：

```yaml
ropd_rcpc_derive_candidates_from_budget: true
```

含义：

- `true`：统一用 `ropd_rcpc_budget` 派生 top actions/top blocks/intervention blocks。
- `false`：允许手动使用 `ropd_rcpc_top_actions`、`ropd_rcpc_top_blocks`、`ropd_rcpc_intervention_max_blocks_per_group` 等 ablation 参数。

建议：

- 正式主实验保持 `true`，避免各预算不一致。
- 做消融实验时再设成 `false`。

#### `ropd_rcpc_min_action_chars`

默认：

```yaml
ropd_rcpc_min_action_chars: 12
```

含义：

- 太短的 action 会尝试与前一个 action 合并，除非是 protected reasoning marker。

调参：

- action 过碎：增大。
- 关键短转折经常被吞掉：减小，但注意噪声。

#### `ropd_rcpc_max_action_chars`

默认：

```yaml
ropd_rcpc_max_action_chars: 260
```

含义：

- 超过该字符长度的 action 会进一步拆分。

调参：

- block 经常过长：降低。
- action 被切得太碎：提高。

#### `ropd_rcpc_max_action_tokens`

默认：

```yaml
ropd_rcpc_max_action_tokens: 24
```

含义：

- 单个 action 的最大 token 数。

调参：

- action 跨越太多 reasoning move：降低。
- action 被截断到不完整：提高。

#### `ropd_rcpc_min_anchor_z`

默认：

```yaml
ropd_rcpc_min_anchor_z: 0.5
```

含义：

- block anchor 必须达到的 robust uncertainty z-score 下限。

调参：

- 候选 block 太少：降低到 0.3。
- 候选噪声太多：提高到 0.8 或 1.0。

#### `ropd_rcpc_transport_lambda`

默认：

```yaml
ropd_rcpc_transport_lambda: 1.0
```

含义：

- 控制 causal effect 对 token/block credit redistribution 的影响强度。

调参：

- 想更依赖因果干预结果：提高。
- 因果估计噪声大，想更保守：降低。

#### `ropd_rcpc_effect_noise_floor`

默认：

```yaml
ropd_rcpc_effect_noise_floor: 0.05
```

含义：

- 因果效应估计的最低噪声下限，用于 shrinkage/calibration。

调参：

- 反事实估计太激进：提高。
- 因果效应总是被压得太小：降低。

#### `ropd_rcpc_fallback_to_criterion_advantage`

默认：

```yaml
ropd_rcpc_fallback_to_criterion_advantage: true
```

含义：

- 当没有可用 candidate block 或 intervention effects 时，是否退回到整段 response 的 criterion advantage。

建议：

- 主训练保持 `true`，否则没有候选时可能导致 advantage 全零。
- 做严格局部 credit ablation 时可设为 `false`。

### 4.3 Verifier/Judge 参数

#### `ROPD_MAX_CONCURRENCY`

控制 verifier 并发。见上。

#### `ROPD_VERIFIER_MAX_OUTPUT_TOKENS`

默认：

```bash
ROPD_VERIFIER_MAX_OUTPUT_TOKENS="${ROPD_VERIFIER_MAX_OUTPUT_TOKENS:-2048}"
```

含义：

- verifier JSON 输出最大 token 数。

调参：

- rubric 很多、answers 很多，verifier 输出被截断：增大。
- verifier 响应慢且输出很短：可降低。

### 4.4 格式相关参数

#### `ROPD_REQUIRE_STRICT_COT_FORMAT`

默认：

```bash
ROPD_REQUIRE_STRICT_COT_FORMAT=true
```

含义：

- 是否统计并要求 `<reasoning>...</reasoning><answer>...</answer>` 格式。

#### `ROPD_ZERO_SCORE_ON_FORMAT_ERROR`

默认：

```bash
ROPD_ZERO_SCORE_ON_FORMAT_ERROR=false
```

含义：

- 格式错误是否把 final reward 置零。

#### `ROPD_ZERO_CRITERIA_ON_FORMAT_ERROR`

默认：

```bash
ROPD_ZERO_CRITERIA_ON_FORMAT_ERROR=false
```

含义：

- 格式错误是否把 criterion judgement/advantage 置零。

建议：

- 当前主训练保持 false，因为早期格式不稳定时如果硬清零，RCPC 没有学习信号。
- 如果 SFT checkpoint 格式已经很稳，可考虑打开，用来强化协议遵循。

## 5. 快速排查表

| 现象 | 优先检查 |
|---|---|
| `reward/group_ok_ratio` 低 | verifier JSON 是否合规；`JUDGE_*` 是否正确；数据是否有 `rubric` |
| `reward/format_valid_ratio` 长期低 | SFT checkpoint、forced prefix、response 格式 prompt |
| `criterion_advantage/std` 接近 0 | group 内 rollout 差异太小；judge 太宽；rubric 不区分 |
| `rcpc/intervention_groups` 为 0 | intervention 是否开启；budget；priority 是否全为 0；是否有 group 被允许干预 |
| `rcpc/calibrated_effect_abs_mean` 为 0 | block 不关键；反事实生成太相似；judge 不敏感；`counterfactual_samples` 太少 |
| `rcpc/token_coverage` 太低 | action/block 候选过少；提高 budget 或降低 `min_anchor_z` |
| `rcpc/token_coverage` 太高 | 候选过泛；降低 budget 或提高 `min_anchor_z` |
| step 很慢 | 看 `timing_s/rcpc/intervention_score_wall`、`timing_s/reward/verifier_initial_group`、`timing_s/gen` |

## 6. 推荐的调参顺序

1. 先看 `reward/group_ok_ratio` 和 `reward/format_valid_ratio`，确认 reward pipeline 和格式健康。
2. 再看 `criterion_advantage/std`，确认 group 内确实有 rubric 维度区分信号。
3. 再看 `[ropd rcpc intervention summary]`，人工检查 block 文本是否是合理 reasoning unit。
4. 再看 `rcpc/calibrated_effect_abs_mean`、`rcpc/causal_units`、`score_effect`，确认反事实干预真的改变 verifier judgement。
5. 最后再调大/调小 `RCPC_BUDGET`、`RCPC_COUNTERFACTUAL_SAMPLES`、`ropd_rcpc_min_anchor_z` 等成本/质量参数。

