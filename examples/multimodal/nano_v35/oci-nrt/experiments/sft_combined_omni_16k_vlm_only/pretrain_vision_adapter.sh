#!/bin/bash

#SBATCH -A nemotron_omni_vision
#SBATCH -p batch_block1
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton
#SBATCH --nodes=32
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --gpus-per-node=8
#SBATCH --job-name=nano35_base3p5_vision_pretrain

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ! -f "${SCRIPT_DIR}/experiment_config.sh" ]]; then
    SCRIPT_DIR=${NANO_V35_EXPERIMENT_DIR:-"${PWD}/examples/multimodal/nano_v35/oci-nrt/experiments/sft_combined_omni_16k_vlm_only"}
fi
source "${SCRIPT_DIR}/experiment_config.sh"

export MODEL_NAME=${MODEL_NAME:-"${VISION_PRETRAIN_MODEL_NAME}"}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${BASE_VLM_CKPT_DIR}"}
export NANO_V35_SCRIPT_DIR="${OCI_NRT_DIR}"

exec "${OCI_NRT_DIR}/pretrain_nano_v35_radiov4_1377_svg_newcontainer.sh"
