# Stop/return error if failed (necessary to detect if TP1 conversion failed)
#   - Exit immediately on error (-e)
#   - Treat unset variables as errors (-u)
#   - Return first non-zero exit code (-o pipefail)
set -euo pipefail

MODEL_NAME=$1
MCORE_PATH=$2
HF_BASE_PATH=$3
CKPT_STEP=$4
HF_CONFIG_SRC=${5:-examples/multimodal/v3/nano_mcore_to_hf}
TP=${6:-1}
ETP=${7:-1}

CKPT_STEP_DEC=$((10#$CKPT_STEP))
ITER_NAME=$(printf "iter_%07d" "$CKPT_STEP_DEC")
ITER_DIR="${MCORE_PATH%/}/${ITER_NAME}"
CKPT_FILE="${ITER_DIR}/mp_rank_00/model_optim_rng.pt"
if [[ ! -f "$CKPT_FILE" ]]; then
    CKPT_FILE="${ITER_DIR}/mp_rank_00_000/model_optim_rng.pt"
fi
if [[ ! -f "$CKPT_FILE" ]]; then
    echo "Error: checkpoint shard not found under $ITER_DIR" >&2
    exit 1
fi

mkdir -p "$HF_BASE_PATH"
HF_PATH="${HF_BASE_PATH%/}/mcore_to_hf"
CONFIG_YAML_PATH="${MCORE_PATH%/}/../config.yaml"

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Step 1: Convert mcore to hf
python tools/checkpoint/convert.py \
    --model-type hybrid \
    --loader llava \
    --saver hf_moe_llava \
    --load-dir "$MCORE_PATH" \
    --save-dir "$HF_PATH" \
    --megatron-path . \
    --max-queue-size 1 \
    --ckpt-step "$CKPT_STEP_DEC"

# Step 2: Create a txt file in $HF_BASE_PATH showing original mcore path and ckpt step
printf "original mcore path: %s at iteration %s\n" "$MCORE_PATH" "$CKPT_STEP_DEC" \
    > "$HF_BASE_PATH/mcore_to_hf_info.txt"

# Step 3: Copy the HF model-code/config template, then copy the tokenizer used
# by the checkpoint. TOKENIZER_SRC can be supplied explicitly; otherwise derive
# it from the model config. Keep the old shared tokenizer as a compatibility
# fallback for checkpoints whose config predates tokenizer-model.
TOKENIZER_SRC=${TOKENIZER_SRC:-}
if [[ -z "$TOKENIZER_SRC" && -f "$CONFIG_YAML_PATH" ]]; then
    TOKENIZER_SRC=$(sed -n 's/^tokenizer-model:[[:space:]]*//p' "$CONFIG_YAML_PATH" | head -n 1)
    TOKENIZER_SRC=${TOKENIZER_SRC#\"}
    TOKENIZER_SRC=${TOKENIZER_SRC%\"}
    TOKENIZER_SRC=${TOKENIZER_SRC#\'}
    TOKENIZER_SRC=${TOKENIZER_SRC%\'}
fi
TOKENIZER_SRC=${TOKENIZER_SRC:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/tokenizers/nemotron-v3-nano/tokenizer.json}
if [[ -d "$TOKENIZER_SRC" ]]; then
    TOKENIZER_SRC="${TOKENIZER_SRC%/}/tokenizer.json"
fi
if [[ ! -f "$TOKENIZER_SRC" ]]; then
    echo "Error: tokenizer not found at $TOKENIZER_SRC" >&2
    echo "Set TOKENIZER_SRC=/path/to/tokenizer.json or regenerate config.yaml from this checkpoint." >&2
    exit 1
fi

rsync -aL --exclude='model.safetensors.index.json' --exclude='tokenizer.json' "$HF_CONFIG_SRC/" "$HF_PATH/"
cp "$TOKENIZER_SRC" "$HF_PATH/tokenizer.json"

if [[ ! -f "$HF_PATH/tokenizer.json" ]]; then
    echo "Error: Failed to copy tokenizer.json to $HF_PATH" >&2
    exit 1
fi

# Step 4: Generate config.yaml from the exact iteration being converted and
# update model-specific fields in the HF config. Avoid --model_name here: its
# historical default path points at a different cluster and selects latest.
python examples/multimodal/tools/create_yaml_inference_config.py \
    --ckpt_path "$CKPT_FILE" \
    --output_config "$CONFIG_YAML_PATH" \
    --update_hf_config "$HF_PATH"

# Step 5: Verify both config.yaml and config.json exist
if [ ! -f "$CONFIG_YAML_PATH" ]; then
    echo "Error: config.yaml does not exist at $CONFIG_YAML_PATH"
    exit 1
fi
if [ ! -f "$HF_PATH/config.json" ]; then
    echo "Error: config.json does not exist in $HF_PATH"
    exit 1
fi

echo "Conversion completed successfully!"
