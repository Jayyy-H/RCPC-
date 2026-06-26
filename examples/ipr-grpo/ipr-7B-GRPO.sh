set -x
# ray stop
# ray start --head

export VLLM_ATTENTION_BACKEND=XFORMERS
MODEL_PATH=/mnt/bn/yangmin-priv/czh/models/Valley_B7_v3_navit_cot_AutoModel_clean/ # /mnt/bn/fengshuyang-bytenas-10t/jiangzishang/mllm/EasyGuard/ckp/valley_b7_v3_warmup/ # /mnt/bn/yangmin-priv/czh/models/Valley_B7_v3_navit_cot_AutoModel_clean/ 

export CUDA_VISIBLE_DEVICES="4,5,6,7"
python3 -m verl.trainer.main \
    config=./examples/ipr-grpo/ipr-7B-GRPO.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.val_before_train=false \
    data.val_batch_size=128 \
    trainer.experiment_name=sftrl_test_sftcoef_decay \

# RUN_CODE_PATH="/mnt/bn/chenhaobo-va-data/ipr_mllm/code/IPR_MLLM"
# export CUDA_VISIBLE_DEVICES="2,3"
# nohup python3 $RUN_CODE_PATH/scripts/fuck_gpu.py --cuda $CUDA_VISIBLE_DEVICES