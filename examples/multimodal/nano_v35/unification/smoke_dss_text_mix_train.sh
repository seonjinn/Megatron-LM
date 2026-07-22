#!/bin/bash
#SBATCH -A llmservice_fm_vision
#SBATCH -p batch
#SBATCH --qos=short
#SBATCH -t 00:15:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --job-name=nano35-dss-mix-train-smoke

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ! -f "${SCRIPT_DIR}/experiment_config.sh" ]]; then
    SCRIPT_DIR=${NANO_V35_EXPERIMENT_DIR:-"${PWD}/examples/multimodal/nano_v35/unification"}
fi
source "${SCRIPT_DIR}/experiment_config.sh"

export MODEL_NAME=${MODEL_NAME:-"${EXPERIMENT_ID}_dss_text_mix_train_smoke"}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${BASE_VLM_CKPT_DIR}"}
export DATA_TRAIN=${DATA_TRAIN:-"${SCRIPT_DIR}/smoke_dss_text_mix.yaml"}

export ENABLE_WANDB=0
export SKIP_SAVE=1
export NUM_GPU=4
export NW=${NW:-2}
# TP=2 and EP=32 on 64 ranks gives DP=32, so the global batch must be a
# multiple of 32 even for a single optimizer-step smoke test.
export BZ=32
export MBZ=1
export LI=1
export USE_PACKING=1
export PBS=${PBS:-32}
export PACKING_SEQ_LEN=16384
export DECODER_SEQ_LEN=16384
export EARLY_EXIT_ITERS=1
export ALLOW_MISSING_VISION_PROJECTION=1
export CUSTOM_ARGS="${CUSTOM_ARGS:-} --no-load-optim --no-load-rng"

exec "${SCRIPT_DIR}/sft_vlm.sh"
