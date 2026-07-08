#!/bin/bash

#SBATCH -A nemotron_omni_vision
#SBATCH -p batch_singlenode,batch_short,backfill
#SBATCH -t 00:30:00
#SBATCH --mem=251740
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --overcommit
#SBATCH --job-name=nano35_base3p5_vlm

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ! -f "${SCRIPT_DIR}/experiment_config.sh" ]]; then
    SCRIPT_DIR=${NANO_V35_EXPERIMENT_DIR:-"${PWD}/examples/multimodal/nano_v35/oci-nrt/experiments/sft_combined_omni_16k_vlm_only"}
fi
source "${SCRIPT_DIR}/experiment_config.sh"

export LM_MCORE_DIR="${BASE_MCORE_CKPT_DIR}/torch"
export OUTPUT_CKPT_DIR="${BASE_VLM_CKPT_DIR}"
export NANO_V35_SCRIPT_DIR="${OCI_NRT_DIR}"

exec "${OCI_NRT_DIR}/combine_nano_v35_with_radio.sh"
