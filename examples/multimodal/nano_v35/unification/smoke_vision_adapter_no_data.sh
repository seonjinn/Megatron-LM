#!/bin/bash

# Model/checkpoint initialization smoke test only. No optimizer, data, training,
# or checkpoint writes are performed.
#SBATCH -A llmservice_fm_vision
#SBATCH -p batch
#SBATCH --qos=short
#SBATCH -t 00:05:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --job-name=nano35-vision-init-smoke

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ! -f "${SCRIPT_DIR}/experiment_config.sh" ]]; then
    SCRIPT_DIR=${NANO_V35_EXPERIMENT_DIR:-"${PWD}/examples/multimodal/nano_v35/unification"}
fi
source "${SCRIPT_DIR}/experiment_config.sh"

export MODEL_NAME=${MODEL_NAME:-"${EXPERIMENT_ID}_vision_model_init_smoke"}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${BASE_VLM_CKPT_DIR}"}
export TRAIN_ENTRYPOINT="${SCRIPT_DIR}/model_init_no_data.py"

export ENABLE_WANDB=0
export SKIP_SAVE=1
export NUM_GPU=4
export NW=0
export CUSTOM_ARGS="${CUSTOM_ARGS:-} --skip-train --no-load-optim --no-load-rng"

exec "${SCRIPT_DIR}/pretrain_vision_adapter.sh"
