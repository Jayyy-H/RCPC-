# SFTRL联合训练（preview）
目前仅支持valleyb7v3的SFT+RL训练，qwen等开源模型未来会更新支持。一些启动config配置等待后续完善

## 快速启动
启动脚本在./examples/ipr-grpo/ipr-7B-GRPO.sh


## 数据集准备
每条问题额外准备一条专家轨迹，通过data.target_key指定。对于一个问题，在update_policy时，会在n条on-policy样本上计算GRPO损失，在off-policy上计算SFT损失。

## 相关论文可参考（仅展示部分）
1. Learning to Reason under Off-Policy Guidance
2. On-Policy RL Meets Off-Policy Experts: Harmonizing Supervised Fine-Tuning and Reinforcement Learning via Dynamic Weighting
3. Towards a Unified View of Large Language Model Post-Training