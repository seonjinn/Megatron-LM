#!/bin/bash

#SBATCH -A nemotron_omni_vision
#SBATCH -p batch_singlenode,batch_short,backfill
#SBATCH -t 00:30:00
#SBATCH --mem=251740
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --overcommit
#SBATCH --job-name=nano_v35_combine_radio_v4

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ! -f "${SCRIPT_DIR}/cluster_config.sh" ]]; then
    SCRIPT_DIR=${NANO_V35_SCRIPT_DIR:-"${PWD}/examples/multimodal/nano_v35/oci-nrt"}
fi
source "${SCRIPT_DIR}/cluster_config.sh"

SOURCE=$(pwd)

LM_ITERATION=${LM_ITERATION:-${ITERATION:-1}}
VISION_ITERATION=${VISION_ITERATION:-1}
OUTPUT_ITERATION=${OUTPUT_ITERATION:-1}
DRY_RUN=${DRY_RUN:-0}

LOGS_DIR="${OUTPUT_CKPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

run_cmd="cd ${SOURCE}; python -u examples/multimodal/nano_v35/combine_nano_v35_with_radio.py \
    --lm-dir ${LM_MCORE_DIR} \
    --vision-dir ${VISION_CKPT_DIR} \
    --output-dir ${OUTPUT_CKPT_DIR} \
    --lm-iteration ${LM_ITERATION} \
    --vision-iteration ${VISION_ITERATION} \
    --output-iteration ${OUTPUT_ITERATION}"

echo "LM_MCORE_DIR=${LM_MCORE_DIR}"
echo "VISION_CKPT_DIR=${VISION_CKPT_DIR}"
echo "OUTPUT_CKPT_DIR=${OUTPUT_CKPT_DIR}"
echo "${run_cmd}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    exit 0
fi

if [[ ! -d "${LM_MCORE_DIR}/iter_$(printf '%07d' "${LM_ITERATION}")" ]]; then
    echo "LM checkpoint iteration is missing under ${LM_MCORE_DIR}"
    exit 1
fi
if [[ ! -d "${VISION_CKPT_DIR}/iter_$(printf '%07d' "${VISION_ITERATION}")" ]]; then
    echo "Vision checkpoint iteration is missing under ${VISION_CKPT_DIR}"
    exit 1
fi
if [[ ! -r "${CONTAINER_IMAGE}" ]]; then
    echo "Container image is missing or unreadable: ${CONTAINER_IMAGE}"
    exit 1
fi

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
srun -l --verbose \
    --container-image "${CONTAINER_IMAGE}" \
    --container-mounts "${CONTAINER_MOUNTS}" \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    sh -c "${run_cmd}"
