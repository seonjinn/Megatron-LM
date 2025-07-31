MCORE_PATH=$1
HF_BASE_PATH=$2
CKPT_STEP=$3

mkdir -p $HF_BASE_PATH
HF_PATH=$HF_BASE_PATH/mcore_to_hf
FIXED_HF_PATH=$HF_BASE_PATH/mcore_to_hf_fixed

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

# Step 2: Modify cls token
python examples/multimodal/tools/modify_cls_token.py $HF_PATH $FIXED_HF_PATH

# Create a txt file in $HF_BASE_PATH saying
# original mcore path: $MCORE_PATH at iteration $CKPT_STEP
touch $HF_BASE_PATH/mcore_to_hf_info.txt
echo "original mcore path: $MCORE_PATH at iteration $CKPT_STEP" >> $HF_BASE_PATH/mcore_to_hf_info.txt

# Step 3: Copy model configs
cp /lustre/fsw/portfolios/llmservice/users/charlwang/nvwork/250709_vlm/vision_model_config/* $FIXED_HF_PATH