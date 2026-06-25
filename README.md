# RCPC: Rubric-Conditioned Path Causal Credit


当前仓库包含两部分：

1. 方法归纳：完整描述 RCPC 的问题定义、rubric reward、候选 action 锚定、局部区间合并、因果干预、hidden-state mediation gate，以及如何接入 GRPO/ROPD。
2. 实验脚本：用 Qwen3/Qwen 系列模型在任意 JSONL/CSV 数据集上生成 reasoning trace，提取 token entropy，锚定 Top-K candidate actions，并做 peak-centered local block aggregation。

## 1. Motivation

传统 RLHF/RLVR/GRPO 常把 reward 放在整条 response 上，甚至只依赖 final answer 是否正确。这会导致几个问题：

- Reward 太稀疏：模型不知道到底是哪一步 reasoning 贡献了最终正确或错误。
- Credit assignment 粗糙：同一条 response 中，正确 evidence use、错误 assumption、无关 filler 会共享同一个 advantage。
- Rubric 信号被浪费：ROPD/Rubricator/Verifier 能知道每条 response 在不同 criterion 上的表现，但标准训练通常又把这些分数压回一个 scalar。
- Token/action 级优化缺位：reasoning 的真实错误往往发生在局部决策点，例如错误选择 evidence、跳过 boundary condition、混淆条件、错误排除选项。

RCPC 的核心目标是：在不额外训练第三个 reward/causal model 的前提下，把 rubric-level supervision 转换成 action/block-level causal credit，并用于更细粒度的 policy optimization。

## 2. Problem Setup

对每个 prompt \(x_i\)，policy model 采样一组 responses：

\[
y_{i,1}, y_{i,2}, ..., y_{i,G}
\]

ROPD-style pipeline 产生：

- Teacher response \(y_i^T\)：高质量参考 reasoning。
- Rubrics \(C_i = \{c_{i,1}, ..., c_{i,K}\}\)：由 Rubricator 根据 prompt、teacher response、student responses 动态生成。
- Verifier scores \(s_{i,g,k}\)：第 \(g\) 条 student response 在第 \(k\) 条 criterion 上的得分。
- Final scalar score \(S_{i,g}\)：通常是 criterion 加权求和并归一化后的 response-level reward。

标准 GRPO 通常只使用 \(S_{i,g}\) 做组内 advantage。RCPC 希望保留并利用更丰富的结构：

\[
\{s_{i,g,k}\}_{k=1}^{K}
\]

即每条 response 在每个 rubric criterion 上的表现。

## 3. Overall Pipeline

RCPC 可以拆成六个阶段。

### Stage A: Rubric-Conditioned Response Scoring

先保持 ROPD 的基本形式：

1. Teacher 生成高质量 reasoning trace。
2. Rubricator 根据当前 prompt、teacher response、student group responses 生成任务相关 rubrics。
3. Verifier 对每条 student response 按每条 criterion 打分。
4. 得到 criterion-level score matrix。

与普通 ROPD 不同，RCPC 不急于把所有 criterion 分数压成单一 reward，而是保留：

\[
s_{i,g,k}
\]

并计算 criterion-wise advantage：

\[
A_{i,g,k} =
\frac{s_{i,g,k} - \mathrm{mean}_{g'}(s_{i,g',k})}
{\mathrm{std}_{g'}(s_{i,g',k})+\epsilon}
\]

这一步能区分“这条 response 是在哪个 rubric 维度上更好或更差”。

### Stage B: Candidate Reasoning Action Anchoring

我们把 response 切分为 micro-sentence action：

\[
y_{i,g} = (a_{i,g,1}, a_{i,g,2}, ..., a_{i,g,M})
\]

当前实现采用轻量规则：

- 逗号、中文逗号作为主要边界。
- 换行、分号、句号、列表项作为辅助边界。
- reasoning connectives 作为辅助边界，例如 therefore、however、so、if、then、because、but、thus、hence、next、check、finally。
- 设置最小字符长度、最大字符长度、最大 token 长度，避免过碎或过长。

对每个 generated token \(t\)，从模型 generation logits 计算 entropy：

\[
H_t = -\sum_v p(v|x,y_{<t})\log p(v|x,y_{<t})
\]

对 action \(a_m\) 内 token 集合 \(T_m\)，用长度自适应 top-r mean 聚合：

\[
E_m =
\frac{1}{\lceil \sqrt{|T_m|} \rceil}
\sum_{t \in \mathrm{TopR}(T_m,\lceil \sqrt{|T_m|} \rceil)} H_t
\]

这比 max 更抗噪声，也比 mean 更不容易被长 span 稀释。

然后在同一条 response 内做 robust normalization：

\[
\tilde{E}_m =
\frac{E_m - \mathrm{median}(E)}
{\mathrm{MAD}(E)+\epsilon}
\]

当前脚本会在 MAD 太小时 fallback 到 standard deviation，避免数值爆炸。

最终选择 Top-K action 作为 candidate reasoning decisions：

\[
\mathcal{A}^{cand}_{i,g} = \mathrm{TopK}_m(\tilde{E}_m)
\]

直觉：高 entropy action 往往对应模型局部不确定、路径分叉、证据选择、条件判断、结论跳转等位置。这些位置更可能是值得做 causal credit 的候选点。

### Stage C: Peak-Centered Local Block Aggregation

单个 micro-action 有时语义不完整，例如一个条件判断需要左右相邻句共同构成一个 reasoning unit。因此 RCPC 不只看单点 action，也构造局部 block：

\[
b_{i,g,m:n} = (a_{i,g,m}, ..., a_{i,g,n})
\]

当前 probe 采用严格的 peak-centered 规则：

1. Anchor action 必须在 Stage B 的 Top-K candidate set 中。
2. Anchor 必须是局部 entropy peak：

\[
\tilde{E}_m \geq \tilde{E}_{m-1}, \quad
\tilde{E}_m > \tilde{E}_{m+1}
\]

3. 只允许合并 anchor 的一阶左右邻居。
4. 左右邻居也必须在 Stage B 的 Top-K candidate set 中，才允许进入 block。
5. 不把非候选 action 强行合并进 block。

这样得到的 block 是“以 entropy peak 为中心的局部 reasoning neighborhood”，用于吸收相邻 reasoning actions 的局部交互效应。

### Stage D: Local Causal Intervention

候选 action/block 只是 suspicious set，还不是 causal attribution。下一步要估计某个 action/block 对 rubric criterion \(c_k\) 的局部因果效应。

可定义 intervention operator：

- Mask intervention：删除或遮蔽 action/block。
- Rewrite intervention：用 teacher-style 或 neutral placeholder 替换该 span。
- Counterfactual correction：把疑似错误 span 改写成更符合 rubric 的版本。
- Counterfactual corruption：把疑似正确 span 改写成错误或不完整版本。

对 action/block \(z\)，估计其对 criterion \(k\) 的 causal effect：

\[
\Delta_{z,k}
= V_k(y) - V_k(\mathrm{do}(y \setminus z))
\]

或

\[
\Delta_{z,k}
= V_k(\mathrm{do}(y \leftarrow z^{corrected})) - V_k(y)
\]

其中 \(V_k\) 是 verifier 对 criterion \(k\) 的评分函数。

重点是：干预发生在同一条 response 的局部 span 上，因此比跨样本相关性更接近 causal credit。

### Stage E: Hidden-State Mediation Gate

LLM/verifier 的 span attribution 可能不完全精准。RCPC 可以用 policy model 自身的 hidden states 做 mediation gate，过滤不可靠 attribution。

候选特征包括：

- Span hidden state mean / max pooling。
- Span 与 final answer token hidden state 的相似度。
- Span 与 rubric text embedding 的相似度。
- Span 前后 hidden state shift。
- Intervention 前后 next-token distribution shift。
- Span entropy、logprob、gradient norm、attention-to-evidence 等模型内部信号。

一个简单的 gate 可以写成：

\[
G_{z,k} = \sigma(f(h_z, h_{answer}, e_k, \tilde{E}_z, \Delta_{z,k}))
\]

但为了避免引入第三个模型，早期版本不训练 \(f\)，而是使用规则或无参组合：

- causal effect 足够大；
- span entropy 足够高；
- span 与 criterion embedding/teacher evidence 有较高相似度；
- intervention 后 verifier score 发生稳定变化。

最终 credit：

\[
Credit_{z,k} = G_{z,k} \cdot \Delta_{z,k}
\]

### Stage F: RCPC Advantage for Policy Optimization

把 criterion-wise advantage 与 action/block causal credit 结合：

\[
A^{RCPC}_{i,g,z}
= \sum_k A_{i,g,k} \cdot w_{z,k}
\]

其中 \(w_{z,k}\) 来自 causal effect 或 gated causal credit。

训练时可以有两种接入方式：

1. Soft action weighting：仍使用 response-level GRPO loss，但对 action/block 对应 token 的 loss 乘以局部权重。
2. Token-level credit shaping：把 action/block credit 分配到其 token span 上，得到 token-level advantage。

概念上，原 GRPO：

\[
\mathcal{L}_{GRPO}
= - \sum_t A_{i,g}\log \pi_\theta(y_t|x,y_{<t})
\]

RCPC 变为：

\[
\mathcal{L}_{RCPC}
= - \sum_t A_{i,g,t}^{RCPC}\log \pi_\theta(y_t|x,y_{<t})
\]

其中 \(A_{i,g,t}^{RCPC}\) 由包含 token \(t\) 的 action/block credit 聚合得到。

## 4. Why Causal, Not Only Correlational

只看 entropy 或 verifier attribution 仍然是相关信号。RCPC 的 causal 部分来自局部干预：

- 对同一条 response 做 span-level do-operation；
- 控制 prompt、其余 response context、rubric criterion 不变；
- 观察 criterion score 的反事实变化；
- 只把能改变 rubric score 的 span 视为 causal candidate。

这使得 RCPC 不只是“高 entropy token 加权”，而是“高不确定候选 + rubric-conditioned local intervention + hidden-state mediation gate”。

## 5. Current Probe Scripts

当前仓库提供的脚本用于验证前两步：

1. Candidate reasoning action anchoring。
2. Peak-centered local block aggregation。

它还不会真正执行 causal intervention 或训练 policy model，但会输出后续做 causal attribution 所需的 token/action/block 结构。

### Input Data Format

推荐 JSONL，每行一条样本：

```json
{"id": "sample-001", "source": "hotpotqa", "question": "Question text here", "answer": "gold answer here"}
```

字段名可以通过命令行参数修改：

- `--id-field`
- `--source-field`
- `--question-field`
- `--answer-field`

CSV 也支持，但推荐 JSONL。

### Run Anchor and Block Probe

复制并修改下面命令中的空路径：

```bash
python scripts/run_anchor_block_probe.py \
  --model-path "" \
  --data-path "" \
  --out-dir runs/anchor_block_probe \
  --num-samples 50 \
  --max-new-tokens 512 \
  --top-actions 12 \
  --top-blocks 6 \
  --device auto
```

也可以使用 shell 模板：

```bash
bash scripts/run_anchor_block_probe.sh
```

运行前需要编辑脚本里的：

```bash
MODEL_PATH=""
DATA_PATH=""
```

### Outputs

输出目录包含：

- `results.jsonl`：每条样本的 prompt、response、tokens、actions、top_actions、candidate_blocks。
- `top_actions.csv`：所有 Top-K candidate actions 的表格。
- `candidate_blocks.csv`：所有 peak-centered local blocks 的表格。
- `review.md`：便于人工审查的 Markdown 汇总。
- `stats.json`：简单统计信息。

## 6. Recommended Evaluation Directions

不建议只用数学题验证 RCPC。数学数据容易把高 entropy 放在公式、符号、试错和 hesitation 上，反而不利于验证 action anchoring。

更推荐：

- Multi-hop evidence QA：HotpotQA、2WikiMultiHopQA、MuSiQue、StrategyQA。
- Logical reasoning：FOLIO、ReClor、LogiQA、BBH logical deduction。
- Commonsense/causal reasoning：CommonsenseQA、SocialIQA、PIQA、COPA。
- Document/evidence reasoning：DROP、ContractNLI。

这些任务中的 candidate actions 更可能对应：

- evidence selection；
- bridge entity；
- condition checking；
- option elimination；
- causal assumption；
- boundary/exemption handling；
- conclusion consistency。

## 7. Roadmap

Short term:

- 用不同 Qwen3 参数规模验证 candidate action 和 block 是否稳定。
- 在非数学 reasoning 数据上人工检查 Top-K action 的合理性。
- 对比 entropy anchoring、low logprob anchoring、gradient-based anchoring、LLM attribution anchoring。

Middle term:

- 实现 local intervention evaluator。
- 让 verifier 输出 criterion-level score changes。
- 构造 span-level causal credit。
- 比较 response-level GRPO、criterion-wise GRPO、RCPC token/action-weighted GRPO。

Long term:

- 引入 hidden-state mediation gate。
- 研究 causal credit 的稳定性、可迁移性和泛化能力。
- 在 multi-hop QA、logic、commonsense、agent/web reasoning 上系统评测。

