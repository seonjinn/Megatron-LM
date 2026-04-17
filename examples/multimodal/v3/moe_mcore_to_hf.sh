# Stop/return error if failed (necessary to detect if TP1 conversion failed)
#   - Exit immediately on error (-e)
#   - Treat unset variables as errors (-u)
#   - Return first non-zero exit code (-o pipefail)
set -euo pipefail

MODEL_NAME=$1
MCORE_PATH=$2
HF_BASE_PATH=$3
CKPT_STEP=$4
HF_CONFIG_SRC=$5
TP=${6:-1}
ETP=${7:-1}

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

# Step 2: Create a txt file in $HF_BASE_PATH showing original mcore path and ckpt step
touch $HF_BASE_PATH/mcore_to_hf_info.txt
echo "original mcore path: $MCORE_PATH at iteration $CKPT_STEP" >> $HF_BASE_PATH/mcore_to_hf_info.txt

# Step 3: Copy the "default" hf config from the template directory, then copy tokenizer.
# tokenizer.json lives on lustre (not in git) — see the README at that path for context.
TOKENIZER_SRC=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/tokenizers/nemotron-v3-nano/tokenizer.json
if [ ! -f "$TOKENIZER_SRC" ]; then
    echo "Error: tokenizer not found at $TOKENIZER_SRC" >&2
    exit 1
fi

rsync -aL --exclude='model.safetensors.index.json' --exclude='tokenizer.json' "$HF_CONFIG_SRC/" "$HF_PATH/"
cp "$TOKENIZER_SRC" "$HF_PATH/tokenizer.json"

if [[ ! -f "$HF_PATH/tokenizer.json" ]]; then
    echo "Error: Failed to copy tokenizer.json to $HF_PATH" >&2
    exit 1
fi

# Step 4: Overwrite a few model-specific params using create_yaml_inference_config.py --update_hf_config
python examples/multimodal/tools/create_yaml_inference_config.py \
--model_name $MODEL_NAME \
--update_hf_config $HF_PATH \
--tensor-model-parallel-size $TP \
--expert-tensor-parallel-size $ETP

# Step 5: Verify both config.yaml and config.json exist
# NOTE: MCORE_PATH is `<user_lustre>/workspace/output/<model_name>/checkpoints`
#   and create_yaml_inference_config.py stores config.yaml in parent `<model_name>/` dir
if [ ! -f "$MCORE_PATH/../config.yaml" ]; then
    echo "Error: config.yaml does not exist in $MCORE_PATH/../"
    exit 1
fi
if [ ! -f "$HF_PATH/config.json" ]; then
    echo "Error: config.json does not exist in $HF_PATH"
    exit 1
fi

echo "Conversion completed successfully!"
