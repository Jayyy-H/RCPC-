bash /mnt/bn/fengshuyang-bytenas-10t/jiangzishang/mllm/EasyR1/examples/ipr-grpo/ipr-7B-GRPO.sh  > ./logs/sftrl.log 2>&1 
RUN_CODE_PATH="/mnt/bn/chenhaobo-va-data/ipr_mllm/code/IPR_MLLM"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
nohup python3 $RUN_CODE_PATH/scripts/fuck_gpu.py --cuda $CUDA_VISIBLE_DEVICES