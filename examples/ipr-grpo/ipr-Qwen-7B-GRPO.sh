set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
MODEL_PATH=/mnt/bn/yangmin-priv/czh/models/Qwen2.5-VL-7B-Instruct/ # /mnt/bn/fengshuyang-bytenas-10t/jiangzishang/mllm/EasyGuard/ckp/valley_b7_v4_warmup/checkpoint-163 #/mnt/bn/yangmin-priv/czh/models/Valley_B7_v3_navit_cot_AutoModel_clean/ # replace it with your local file path

python3 -m verl.trainer.main \
    config=/mnt/bn/fengshuyang-bytenas-10t/jiangzishang/mllm/EasyR1/examples/ipr-grpo/ipr-Qwen-7B-GRPO.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.val_before_train=true \
    data.val_batch_size=140 \
    trainer.experiment_name=valley_b7v3_cot-GRPO \

# RUN_CODE_PATH="/mnt/bn/chenhaobo-va-data/ipr_mllm/code/IPR_MLLM"
# export CUDA_VISIBLE_DEVICES="2,3"
# nohup python3 $RUN_CODE_PATH/scripts/fuck_gpu.py --cuda $CUDA_VISIBLE_DEVICES