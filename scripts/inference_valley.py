import json
import math
import multiprocessing
import os
import re
from argparse import ArgumentParser
from io import BytesIO
from multiprocessing import Pool, Queue
from PIL import Image
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    set_seed,
    AutoConfig,
    AutoModel,
    AutoProcessor,
    GenerationConfig,
    PreTrainedTokenizer
)

precision_dict = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

# https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/evaluate_vqa.py
def split_list(lst, n):
    length = len(lst)
    avg = length // n  # 每份的大小
    result = []  # 存储分割后的子列表
    for i in range(n - 1):
        result.append(lst[i*avg:(i+1)*avg])
    result.append(lst[(n-1)*avg:])
    return result

def save_json(json_list,save_path):
    with open(save_path, 'w') as file:
        json.dump(json_list, file,indent=4)

def judge(response, gt):
    match = re.search(r'boxed\{([\s\S]*?)\}', response)
    if match:
        extracted_response = match.group(1)
        if gt.strip().lower() in extracted_response.strip().lower():
            return True
    else:
        if gt.strip().lower() in response.strip().lower():
            return True
    return False

def inference(rank, world_size, args):
    set_seed(42)
    this_rank_gpu_index = rank
    device = torch.device("cuda:" + str(this_rank_gpu_index) if torch.cuda.is_available() else "cpu")
    
    # Prepare model and processor
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, 
        trust_remote_code=True
    )
    model.eval()
    model = model.to(precision_dict[args.precision]).to(device)
    processor = AutoProcessor.from_pretrained(
        args.processor_path,
        anyres=config.anyres, 
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        grid_pinpoints=config.grid_pinpoints,
        only_crop_single_image=config.only_crop_single_image,
        use_special_start_end_token=config.use_special_start_end_token,
        trust_remote_code=True
    )

    # Read and split data
    data = []
    with open(args.data_path, "r") as f:
        for item in f:
            data.append(json.loads(item))
    if args.DDP:
        rf = open(args.output_path + ".worker_" + str(rank), "w")
        if len(data) % world_size != 0:
            data += [data[-1]] * (world_size - len(data) % world_size)
    else:
        rf = open(args.output_path, "w")

    start = len(data) // world_size * rank
    end = min(start + len(data) // world_size, len(data))

    # Inference
    for i in tqdm(range(start, end)):
        row_dict = data[i]
        if "image" in row_dict:
            row_dict["images"] = row_dict["image"]
            del row_dict["image"]
        
        if "images" not in row_dict:
            row_dict["images"] = None

        processed_data = processor({
            "conversations": [
                {"role": "user", "content": row_dict["problem"]},
            ],
            "images": row_dict["images"] 
        })

        input_ids = processed_data["input_ids"].to(device)
        image_sizes=processed_data["image_sizes"] if "image_sizes" in processed_data else None
        pixel_values=processed_data["pixel_values"].to(device) if "pixel_values" in processed_data else None
        image_grid_thw=processed_data["image_grid_thw"].to(device) if "image_grid_thw" in processed_data else None
        if config.anyres:
            images = [[item.to(precision_dict[args.precision]).to(device)for item in img] for img in processed_data["images"]]
        else:
            images = [img.to(precision_dict[args.precision]).to(device) for img in processed_data["images"]]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                images=images,
                image_sizes=image_sizes,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                do_sample=False,
                repetition_penalty=1.0,
                max_new_tokens=args.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True
            )

        input_token_len = input_ids.shape[1]
        generation_text = processor.batch_decode(output_ids.sequences[:, input_token_len:])[0]
        generation_text = generation_text.replace("<|im_end|>", "")

        row_dict["response"] = generation_text
        row_dict["match"] = judge(row_dict["response"], row_dict["answer"])
        rf.write(json.dumps(row_dict).strip() + "\n")  # 将结果写入当前worker文件

def gather_result(args, world_size):
    merged_data = []
    
    # Read fragmented files in rank order
    for rank in range(world_size):
        worker_path = f"{args.output_path}.worker_{rank}"
        if os.path.exists(worker_path):
            with open(worker_path, "r") as f:
                for line in f:
                    merged_data.append(json.loads(line.strip()))
            # Remove the worker file
            os.remove(worker_path)

    # Duplicate removal
    id_set = set()
    merged_data = [item for item in merged_data if item["id"] not in id_set and not id_set.add(item["id"])]

    # Count the number of matches
    match_cnt = 0
    for item in merged_data:
        if item["match"]:
            match_cnt += 1
    
    merged_data = {
        "accuracy": match_cnt / len(merged_data),
        "results": merged_data
    }
    
    # Write to output file
    with open(args.output_path, "w") as f:
        json.dump(merged_data, f, indent=4)
    


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str)
    parser.add_argument("--output_path", "-o", type=str)
    parser.add_argument("--data_path", "-a", type=str, default="/mnt/bn/yangmin-priv/czh/data/Product_R1/RT/test/data_PBR_easyr1_0302.jsonl")
    parser.add_argument("--world_size", "-w", type=int, default=8)
    parser.add_argument("--DDP", type=bool, default=True)
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--processor_path", type=str, default="/mnt/bn/yangmin-priv/czh/models/Valley_B7_v3_navit_cot_AutoModel_clean/")
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument("--min_pixels", type=int, default=78400)
    args = parser.parse_args()
    if args.DDP:
        mp.spawn(inference, args=(args.world_size, args), nprocs=args.world_size)
        gather_result(args, args.world_size)
        
    else:
        inference(0, args.world_size, args)
   

    
   