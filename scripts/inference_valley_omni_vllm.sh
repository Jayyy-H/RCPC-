python3 ./scripts/inference_valley_omni_vllm.py \
    --model_path /mnt/bn/chobits-wx-137-fuse/zhangcan/checkpoints/valley_omni/grpo/qwen25vl_grpo_product_RT_34k/global_step_70/actor/huggingface/ \
    --input_file /mnt/bn/luoruipu-disk/zhangcan/valleyo/GRPO/data/multimodal_product_rt-test_397_10img.jsonl \
    --save_path /mnt/bn/luoruipu-disk/zhangcan/EasyR1/results/qwen25vl_step70.jsonl