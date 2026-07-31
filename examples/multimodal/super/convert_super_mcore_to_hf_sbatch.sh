#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DEFAULT_REPO=$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)

DEFAULT_INPUT_PATH="${SUPER_MCORE_INPUT_PATH:-}"
DEFAULT_SAVE_DIR="${SUPER_HF_SAVE_DIR:-}"
DEFAULT_CONTAINER="${SUPER_CONTAINER_IMAGE:-}"
DEFAULT_TOKENIZER_SRC="${SUPER_HF_TOKENIZER_SRC:-}"
DEFAULT_RADIO_HF_SRC="${SUPER_HF_RADIO_SRC:-}"

usage() {
    cat <<EOF
Usage:
  $0 [options]

Submits a Slurm job that converts a Megatron/MCore Super MoE LLaVA checkpoint to
HF safetensors with tools/checkpoint/convert.py.

Options:
  --input PATH          Checkpoint root or iter_XXXXXXX dir.
                        Required unless SUPER_MCORE_INPUT_PATH is set.
  --save-dir PATH       HF output directory.
                        Required unless SUPER_HF_SAVE_DIR is set.
  --ckpt-step N         Checkpoint iteration. Inferred from --input iter dir or
                        latest_checkpointed_iteration.txt when omitted.
  --repo PATH           Megatron-LM repo path. Default: ${DEFAULT_REPO}
  --container PATH      Enroot/SquashFS image.
                        Required unless SUPER_CONTAINER_IMAGE is set.
  --hf-config-src PATH  HF config/template directory.
                        Default: <repo>/examples/multimodal/super/super_mcore_to_hf
  --tokenizer-src PATH  Optional tokenizer asset directory to copy into HF output.
                        Default: SUPER_HF_TOKENIZER_SRC if set.
  --radio-hf-src PATH   Optional local nvidia/C-RADIOv2-H HF-code checkout to copy
                        into HF output for offline loading. Default: SUPER_HF_RADIO_SRC if set.
  --account NAME        Slurm account. Default: llmservice_fm_vision
  --partition NAME      Slurm partition. Default: batch
  --time HH:MM:SS       Slurm time limit. Default: 04:00:00
  --job-name NAME       Slurm job name. Default: hf_convert_super_1377_sft
  --gpus N              GPUs per node. Default: 8
  --max-queue-size N    Converter queue size. Default: 1
  --dry-run             Print the resolved settings without submitting.
  -h, --help            Show this help.

Example:
  $0 \\
    --input /path/to/checkpoints/iter_0058764 \\
    --save-dir /path/to/hf_output
EOF
}

INPUT_PATH="${DEFAULT_INPUT_PATH}"
SAVE_DIR="${DEFAULT_SAVE_DIR}"
CKPT_STEP=""
REPO="${DEFAULT_REPO}"
CONTAINER_IMAGE="${DEFAULT_CONTAINER}"
TOKENIZER_SRC="${DEFAULT_TOKENIZER_SRC}"
RADIO_HF_SRC="${DEFAULT_RADIO_HF_SRC}"
HF_CONFIG_SRC=""
ACCOUNT="llmservice_fm_vision"
PARTITION="batch"
TIME_LIMIT="04:00:00"
JOB_NAME="hf_convert_super_1377_sft"
GPUS_PER_NODE="8"
MAX_QUEUE_SIZE="1"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            INPUT_PATH="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --ckpt-step)
            CKPT_STEP="$2"
            shift 2
            ;;
        --repo)
            REPO="$2"
            shift 2
            ;;
        --container)
            CONTAINER_IMAGE="$2"
            shift 2
            ;;
        --hf-config-src)
            HF_CONFIG_SRC="$2"
            shift 2
            ;;
        --tokenizer-src)
            TOKENIZER_SRC="$2"
            shift 2
            ;;
        --radio-hf-src)
            RADIO_HF_SRC="$2"
            shift 2
            ;;
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --gpus)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --max-queue-size)
            MAX_QUEUE_SIZE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$INPUT_PATH" ]]; then
    echo "Missing --input (or SUPER_MCORE_INPUT_PATH)." >&2
    usage >&2
    exit 2
fi
if [[ -z "$SAVE_DIR" ]]; then
    echo "Missing --save-dir (or SUPER_HF_SAVE_DIR)." >&2
    usage >&2
    exit 2
fi
if [[ -z "$CONTAINER_IMAGE" ]]; then
    echo "Missing --container (or SUPER_CONTAINER_IMAGE)." >&2
    usage >&2
    exit 2
fi

REPO=$(readlink -f "$REPO")
INPUT_PATH=$(readlink -f "$INPUT_PATH")
SAVE_DIR=${SAVE_DIR%/}
if [[ -n "$TOKENIZER_SRC" ]]; then
    TOKENIZER_SRC=$(readlink -f "$TOKENIZER_SRC")
fi
if [[ -n "$RADIO_HF_SRC" ]]; then
    RADIO_HF_SRC=$(readlink -f "$RADIO_HF_SRC")
fi
HF_CONFIG_SRC=${HF_CONFIG_SRC:-"${REPO}/examples/multimodal/super/super_mcore_to_hf"}

MCORE_PATH="$INPUT_PATH"
INPUT_BASENAME=$(basename "$INPUT_PATH")
if [[ "$INPUT_BASENAME" =~ ^iter_([0-9]+)$ ]]; then
    MCORE_PATH=$(dirname "$INPUT_PATH")
    if [[ -z "$CKPT_STEP" ]]; then
        CKPT_STEP=$((10#${BASH_REMATCH[1]}))
    fi
fi

if [[ -z "$CKPT_STEP" ]]; then
    LATEST_FILE="${MCORE_PATH}/latest_checkpointed_iteration.txt"
    if [[ ! -f "$LATEST_FILE" ]]; then
        echo "Could not infer --ckpt-step; missing ${LATEST_FILE}" >&2
        exit 1
    fi
    CKPT_STEP=$(tr -d '[:space:]' < "$LATEST_FILE")
fi

CKPT_STEP_DEC=$((10#$CKPT_STEP))
ITER_NAME=$(printf "iter_%07d" "$CKPT_STEP_DEC")
LOG_DIR="${SAVE_DIR}/logs"

if [[ ! -d "$REPO" ]]; then
    echo "Repo path does not exist: $REPO" >&2
    exit 1
fi
if [[ ! -f "$REPO/tools/checkpoint/convert.py" ]]; then
    echo "convert.py not found under repo: $REPO" >&2
    exit 1
fi
if [[ ! -d "$MCORE_PATH/$ITER_NAME" ]]; then
    echo "Checkpoint iteration directory not found: $MCORE_PATH/$ITER_NAME" >&2
    exit 1
fi
if [[ ! -f "$CONTAINER_IMAGE" ]]; then
    echo "Container image not found: $CONTAINER_IMAGE" >&2
    exit 1
fi
if [[ ! -d "$HF_CONFIG_SRC" ]]; then
    echo "HF config/template directory not found: $HF_CONFIG_SRC" >&2
    exit 1
fi
if [[ -n "$TOKENIZER_SRC" && ! -d "$TOKENIZER_SRC" ]]; then
    echo "Tokenizer source directory not found: $TOKENIZER_SRC" >&2
    exit 1
fi
if [[ -n "$RADIO_HF_SRC" && ! -d "$RADIO_HF_SRC" ]]; then
    echo "RADIO HF source directory not found: $RADIO_HF_SRC" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

cat <<EOF
Resolved conversion settings:
  repo:           $REPO
  checkpoint dir: $MCORE_PATH
  iteration:      $CKPT_STEP_DEC ($ITER_NAME)
  save dir:       $SAVE_DIR
  HF template:    $HF_CONFIG_SRC
  tokenizer src:  ${TOKENIZER_SRC:-<none>}
  RADIO HF src:   ${RADIO_HF_SRC:-<none>}
  container:      $CONTAINER_IMAGE
  account:        $ACCOUNT
  partition:      $PARTITION
  time limit:     $TIME_LIMIT
  logs:           $LOG_DIR/convert_%j.log
EOF

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

sbatch \
    -A "$ACCOUNT" \
    -p "$PARTITION" \
    -N 1 \
    --gres="gpu:${GPUS_PER_NODE}" \
    --ntasks-per-node=1 \
    --mem=0 \
    --exclusive \
    -t "$TIME_LIMIT" \
    --job-name="$JOB_NAME" \
    --output="${LOG_DIR}/convert_%j.log" \
    --export=REPO="$REPO",MCORE_PATH="$MCORE_PATH",CKPT_STEP="$CKPT_STEP_DEC",HF_PATH="$SAVE_DIR",HF_CONFIG_SRC="$HF_CONFIG_SRC",TOKENIZER_SRC="$TOKENIZER_SRC",RADIO_HF_SRC="$RADIO_HF_SRC",CONTAINER_IMAGE="$CONTAINER_IMAGE",MAX_QUEUE_SIZE="$MAX_QUEUE_SIZE" <<'SBATCH'
#!/bin/bash
set -euo pipefail

srun \
    --container-image "$CONTAINER_IMAGE" \
    --container-mounts /scratch,/lustre \
    --no-container-entrypoint \
    bash -lc '
set -euo pipefail

cd "$REPO"

ITER_NAME=$(printf "iter_%07d" "$((10#$CKPT_STEP))")
CKPT_FILE="${MCORE_PATH}/${ITER_NAME}/mp_rank_00_000/model_optim_rng.pt"
if [[ ! -f "$CKPT_FILE" ]]; then
    CKPT_FILE="${MCORE_PATH}/${ITER_NAME}/mp_rank_00/model_optim_rng.pt"
fi
if [[ ! -f "$CKPT_FILE" ]]; then
    echo "Could not find model_optim_rng.pt for ${MCORE_PATH}/${ITER_NAME}" >&2
    exit 1
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="${REPO}:${REPO}/tools/checkpoint:${PYTHONPATH:-}"
CONVERT_USER="${SLURM_JOB_USER:-${USER:-$(id -un)}}"
export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/llmservice/users/${CONVERT_USER}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "$HF_PATH" "$HF_HOME" "$HF_HUB_CACHE"

printf "Starting conversion at %s\n" "$(date -Is)"
printf "Repo: %s\n" "$REPO"
printf "MCore path: %s\n" "$MCORE_PATH"
printf "Iteration: %s (%s)\n" "$CKPT_STEP" "$ITER_NAME"
printf "HF path: %s\n" "$HF_PATH"

python -u tools/checkpoint/convert.py \
    --model-type hybrid \
    --loader llava \
    --saver hf_moe_llava \
    --load-dir "$MCORE_PATH" \
    --save-dir "$HF_PATH" \
    --megatron-path "$REPO" \
    --max-queue-size "$MAX_QUEUE_SIZE" \
    --ckpt-step "$CKPT_STEP"

printf "Copying HF template/config files at %s\n" "$(date -Is)"
# The template contains relative symlinks into examples/multimodal/v3.  Dereference them so
# the converted HF checkpoint can be loaded as a standalone directory.
rsync -aL --exclude="model.safetensors.index.json" "$HF_CONFIG_SRC/" "$HF_PATH/"

printf "Updating HF config from checkpoint args at %s\n" "$(date -Is)"
python -u examples/multimodal/tools/create_yaml_inference_config.py \
    --ckpt_path "$CKPT_FILE" \
    --output_config "$HF_PATH/config.yaml" \
    --update_hf_config "$HF_PATH"

if [[ -n "${TOKENIZER_SRC:-}" ]]; then
    printf "Copying tokenizer assets from %s at %s\n" "$TOKENIZER_SRC" "$(date -Is)"
    rsync -aL --ignore-missing-args \
        "$TOKENIZER_SRC/tokenizer.json" \
        "$TOKENIZER_SRC/tokenizer_config.json" \
        "$TOKENIZER_SRC/special_tokens_map.json" \
        "$TOKENIZER_SRC/chat_template.jinja" \
        "$HF_PATH/"
fi

if [[ -f "$HF_CONFIG_SRC/chat_template.jinja" ]]; then
    printf "Installing super VLM chat template from %s at %s\n" "$HF_CONFIG_SRC/chat_template.jinja" "$(date -Is)"
    cp -L "$HF_CONFIG_SRC/chat_template.jinja" "$HF_PATH/chat_template.jinja"
fi

if [[ -n "${RADIO_HF_SRC:-}" ]]; then
    printf "Copying RADIO HF code from %s at %s\n" "$RADIO_HF_SRC" "$(date -Is)"
    rsync -aL --include="*.py" --exclude="*" "$RADIO_HF_SRC/" "$HF_PATH/"
fi

printf "Normalizing HF config/package metadata at %s\n" "$(date -Is)"
python -u - "$HF_PATH" <<'"'"'PY'"'"'
import json
import sys
from pathlib import Path

hf_path = Path(sys.argv[1])
config_path = hf_path / "config.json"
index_path = hf_path / "model.safetensors.index.json"

with config_path.open() as f:
    config = json.load(f)
with index_path.open() as f:
    weight_map = json.load(f).get("weight_map", {})

if (hf_path / "hf_model.py").exists():
    vision_config = config.setdefault("vision_config", {})
    vision_config["auto_map"] = {
        "AutoConfig": "hf_model.RADIOConfig",
        "AutoModel": "hf_model.RADIOModel",
    }

config["max_sequence_length"] = 262144
config["img_context_token_id"] = 18
config["sound_context_token_id"] = 27

llm_config = config.setdefault("llm_config", {})
llm_config["max_position_embeddings"] = 262144
config["routed_scaling_factor"] = llm_config.get(
    "routed_scaling_factor", config.get("routed_scaling_factor", 5.0)
)

sound_config = config.get("sound_config")
if isinstance(sound_config, dict):
    sound_config["projection_bias"] = True

if "vision_model.radio_model.model.patch_generator.video_embedder.weight" in weight_map:
    config.setdefault("video_temporal_patch_size", 2)

with config_path.open("w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")

tokenizer_config_path = hf_path / "tokenizer_config.json"
chat_template_path = hf_path / "chat_template.jinja"
if tokenizer_config_path.exists() and chat_template_path.exists():
    with tokenizer_config_path.open() as f:
        tokenizer_config = json.load(f)
    tokenizer_config["chat_template"] = chat_template_path.read_text()
    tokenizer_config["pad_token"] = "<|im_end|>"
    tokenizer_config["padding_side"] = "left"
    with tokenizer_config_path.open("w") as f:
        json.dump(tokenizer_config, f, indent=2)
        f.write("\n")
PY

printf "original mcore path: %s at iteration %s\n" "$MCORE_PATH" "$CKPT_STEP" > "$HF_PATH/mcore_to_hf_info.txt"
printf "Conversion completed at %s\n" "$(date -Is)"
ls -lah "$HF_PATH"
'
SBATCH
