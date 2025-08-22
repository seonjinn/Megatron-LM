MCORE_PATH=$1
HF_BASE_PATH=$2
CKPT_STEP=$3
MODEL_TYPE=${4:-12b}

mkdir -p $HF_BASE_PATH
HF_PATH=$HF_BASE_PATH/mcore_to_hf

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Step 1: Convert mcore to hf
python tools/checkpoint/convert.py \
    --model-type hybrid \
    --loader llava \
    --saver hf_llava \
    --load-dir $MCORE_PATH \
    --save-dir $HF_PATH \
    --megatron-path . \
    --max-queue-size 1 \
    --ckpt-step $CKPT_STEP

# Create a txt file in $HF_BASE_PATH saying
# original mcore path: $MCORE_PATH at iteration $CKPT_STEP
touch $HF_BASE_PATH/mcore_to_hf_info.txt
echo "original mcore path: $MCORE_PATH at iteration $CKPT_STEP" >> $HF_BASE_PATH/mcore_to_hf_info.txt

# copy the configuration based on the model type, error out if not 9b or 12b
if [ "$MODEL_TYPE" == "9b" ]; then
    cp /lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/n5p5_9b_model_config/* $HF_PATH
elif [ "$MODEL_TYPE" == "12b" ]; then
    cp /lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/n5p5_12b_hf_config/* $HF_PATH
else
    echo "Error: MODEL_TYPE must be either '9b' or '12b', but got '$MODEL_TYPE'"
    exit 1
fi
