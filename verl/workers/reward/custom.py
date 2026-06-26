# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import math_compute_score, product_compute_score, ipr_compute_score
from verl.utils.reward_score.ropd import RopdIPRRewardScorer


class CustomRewardManager:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: str,
        reward_config=None,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.ropd_scorer = None
        if compute_score == "math":
            self.compute_score = math_compute_score
        elif compute_score == "product":
            self.compute_score = product_compute_score
        elif compute_score == "ipr":
            self.compute_score = ipr_compute_score
        elif compute_score in {"ropd", "ropd_ipr"}:
            self.compute_score = None
            self.ropd_scorer = RopdIPRRewardScorer(
                tokenizer=tokenizer,
                reward_config=reward_config,
                num_examine=num_examine,
            )
        else:
            raise NotImplementedError()

    def __call__(self, data: DataProto) -> torch.Tensor:
        if self.ropd_scorer is not None:
            return self.ropd_scorer(data)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print = 0

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            valid_prompt_ids[valid_prompt_ids < 0] = 0

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["answer"]

            score = self.compute_score(response_str, ground_truth)
            reward_tensor[i, valid_response_length - 1] = score

            if already_print < self.num_examine:
                already_print += 1
                print("[generation]", sequences_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        return reward_tensor
