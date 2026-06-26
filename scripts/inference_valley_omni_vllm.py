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

# pip3 install vllm==0.8.3 qwen_vl_utils
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import json
import re
import argparse
from tqdm import tqdm
from multiprocessing import Pool

def split_by_image_tags(input_str):
    image_tags = re.finditer(r'<image>+', input_str)
    result = []
    start = 0
    for match in image_tags:
        tag_start, tag_end = match.span()
        result.append(input_str[start:tag_start])
        start = tag_end
    result.append(input_str[start:])
    result = [i for i in result if i != '']
    return result

def count_consecutive_images(input_string):
    matches = re.findall(r'(?:<image>)+', input_string)
    counts = [match.count('<image>') for match in matches]
    return counts

def slice_list(lst, chunk_size=9):
    result = []
    for i in range(0, len(lst), chunk_size):
        result.append(lst[i:i + chunk_size])
    return result

def process_line(curdata):
    content_list = []
    problem = curdata["problem"]
    product_imgs_num = count_consecutive_images(problem)[0]
    all_images = [i.replace('ecom-ccr-dev', 'ecom-ccr-dev-878d5f0f') if '878d5f0f' not in i else i for i in curdata['image']]
    product_imgs = all_images[:product_imgs_num]
    product_sku_imgs = all_images[product_imgs_num:]
    prompt_parts = split_by_image_tags(problem)

    content_list.append({"type": "text", "text": prompt_parts[0]})
    if len(product_imgs) > 0:
        for image_path in product_imgs:
            image = {
                "type": "image",
                "image": image_path,
                "min_pixels": 28 * 28 * 4,
                "max_pixels": 512 * 512
            }
            content_list.append(image)
    content_list.append({"type": "text", "text": prompt_parts[1]})
    if len(product_sku_imgs) > 0:
        for image in product_sku_imgs:
            image = {
                "type": "image",
                "image": image_path,
                "min_pixels": 28 * 28 * 4,
                "max_pixels": 512 * 512
            }
            content_list.append(image)
        content_list.append({"type": "text", "text": prompt_parts[2]})

    messages = [
        {"role": "system", "content": r"Please reason step by step, and put your final answer within \boxed{}."},
        {"role": "user", "content": content_list}]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }
    return llm_inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="The path for your merged model (huggingface subfolder)")
    parser.add_argument("--input_file", default=False, type=str, help="The path of the input file")
    parser.add_argument("--save_path", default=None, type=str, help="The path of the result file")
    args = parser.parse_args()

    llm = LLM(
        model=args.model_path,
        limit_mm_per_prompt={"image": 16, "video": 15},
        tensor_parallel_size=8,
        gpu_memory_utilization=0.7,
        max_model_len=20000,
        # max_num_seqs=2,  # batch size
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=5000,
        stop_token_ids=[],
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    with open(args.input_file, 'r', encoding='utf-8') as f:
        needgenerate_data = []
        alldata = [json.loads(i) for i in f.readlines()]
        sliceddata = slice_list(alldata, 1000)
        for curdatas in sliceddata:

            pool_size = 30
            with Pool(pool_size) as pool:
                needgenerate_data = list(tqdm(pool.imap(process_line, curdatas), total=len(curdatas)))

            # for i in tqdm(curdatas):
            #     needgenerate_data.append(process_line(i))

            outputs = llm.generate(needgenerate_data, sampling_params)  # 模型推理

            # 整理实验结果，保存在 'output' 字段中
            result = []
            for idx, out in enumerate(outputs):
                line = curdatas[idx].copy()
                line['output'] = out.outputs[0].text
                result.append(line)

            with open(args.save_path, 'a+', encoding='utf-8') as fout:
                for idx, line in enumerate(result):
                    fout.write(json.dumps(line, ensure_ascii=False) + '\n')

# Ussage:
# python3 ./scripts/inference_valley_omni_vllm.py \
#     --model_path /mnt/bn/chobits-wx-137-fuse/zhangcan/checkpoints/valley_omni/grpo/qwen25vl_grpo_product_RT_34k/global_step_70/actor/huggingface/ \
#     --input_file /mnt/bn/luoruipu-disk/zhangcan/valleyo/GRPO/data/multimodal_product_rt-test_397_10img.jsonl \
#     --save_path /mnt/bn/luoruipu-disk/zhangcan/EasyR1/results/qwen25vl_step70.jsonl