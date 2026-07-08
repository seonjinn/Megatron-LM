#!/bin/bash

#SBATCH -A nemotron_omni_vision
#SBATCH -p batch_block1
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton
#SBATCH --nodes=64
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --gpus-per-node=8
#SBATCH --job-name=nano35_base3p5_vlm_sft

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ! -f "${SCRIPT_DIR}/experiment_config.sh" ]]; then
    SCRIPT_DIR=${NANO_V35_EXPERIMENT_DIR:-"${PWD}/examples/multimodal/nano_v35/oci-nrt/experiments/sft_combined_omni_16k_vlm_only"}
fi
source "${SCRIPT_DIR}/experiment_config.sh"

export MODEL_NAME=${MODEL_NAME:-"${SFT_MODEL_NAME}"}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${VISION_PRETRAIN_CKPT_DIR}"}
export DATA_TRAIN=${DATA_TRAIN:-"${SFT_DATA_RECIPE}"}
export NANO_V35_SCRIPT_DIR="${OCI_NRT_DIR}"
export LANGUAGE_RECOMPUTE_MODULES=${LANGUAGE_RECOMPUTE_MODULES:-"core_attn mlp layernorm moe_act moe"}
export VISION_RECOMPUTE_NUM_LAYERS=${VISION_RECOMPUTE_NUM_LAYERS:-32}
export WANDB_NAME=${WANDB_NAME:-"${SFT_MODEL_NAME}"}
export WANDB_RUN_ID=${WANDB_RUN_ID:-"n35e14v2"}
export WANDB_RESUME=${WANDB_RESUME:-"allow"}

exec "${OCI_NRT_DIR}/sft_nano_v35_conv3d_radiov4_1377_svg_newcontainer.sh"
