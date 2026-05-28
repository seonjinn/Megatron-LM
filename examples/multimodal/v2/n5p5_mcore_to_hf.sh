# Stop/return error if failed (necessary to detect if TP1 conversion failed)
#   - Exit immediately on error (-e)
#   - Treat unset variables as errors (-u)
#   - Return first non-zero exit code (-o pipefail)
set -euo pipefail

MODEL_NAME=$1
MCORE_PATH=$2
HF_BASE_PATH=$3
CKPT_STEP=$4
MODEL_TYPE=${5:-12b}

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

# Step 2: Create a txt file in $HF_BASE_PATH showing original mcore path and ckpt step
touch $HF_BASE_PATH/mcore_to_hf_info.txt
echo "original mcore path: $MCORE_PATH at iteration $CKPT_STEP" >> $HF_BASE_PATH/mcore_to_hf_info.txt

# Step 3: Copy the "default" hf config based on the model type, error out if not 9b or 12b.
# IMPORTANT: Do NOT copy model.safetensors.index.json -- the converter just generated
#   one with the correct weight map (including sound model keys, etc.). Overwriting it
#   with the template's stale copy would make those weights invisible to torch / HF loaders.
if [ "$MODEL_TYPE" == "9b" ]; then
    HF_CONFIG_SRC=/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/n5p5_9b_model_config
elif [ "$MODEL_TYPE" == "12b" ]; then
    HF_CONFIG_SRC=/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/n5p5_12b_hf_config
else
    echo "Error: MODEL_TYPE must be either '9b' or '12b', but got '$MODEL_TYPE'"
    exit 1
fi
rsync -a --exclude='model.safetensors.index.json' "$HF_CONFIG_SRC/" "$HF_PATH/"

# Step 4: Overwrite a few model-specific params using create_yaml_inference_config.py --update_hf_config
python examples/multimodal/tools/create_yaml_inference_config.py --model_name $MODEL_NAME --update_hf_config $HF_PATH

# Step 5: Verify both config.yaml and config.json exist
if [ ! -f "$HF_PATH/config.yaml" ]; then
    echo "Error: config.yaml does not exist in $HF_PATH"
    exit 1
fi
if [ ! -f "$HF_PATH/config.json" ]; then
    echo "Error: config.json does not exist in $HF_PATH"
    exit 1
fi