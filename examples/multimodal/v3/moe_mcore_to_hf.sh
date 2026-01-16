MCORE_PATH=$1
HF_BASE_PATH=$2
CKPT_STEP=$3
#MODEL_TYPE=${4:-12b}

mkdir -p $HF_BASE_PATH
HF_PATH=$HF_BASE_PATH/mcore_to_hf

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Step 1: Convert mcore to hf
python tools/checkpoint/convert.py \
    --model-type hybrid \
    --loader llava \
    --saver hf_moe_llava \
    --load-dir $MCORE_PATH \
    --save-dir $HF_PATH \
    --megatron-path . \
    --max-queue-size 1 \
    --ckpt-step $CKPT_STEP

# Create a txt file in $HF_BASE_PATH saying
# original mcore path: $MCORE_PATH at iteration $CKPT_STEP
touch $HF_BASE_PATH/mcore_to_hf_info.txt
echo "original mcore path: $MCORE_PATH at iteration $CKPT_STEP" >> $HF_BASE_PATH/mcore_to_hf_info.txt

cp /lustre/fsw/portfolios/llmservice/users/kchumachenko/workspace/output/moe_hf_config/* $HF_PATH
