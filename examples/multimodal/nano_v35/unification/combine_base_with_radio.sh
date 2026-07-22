#!/bin/bash

#SBATCH -A nemotron_n4_post
#SBATCH -p batch
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
    SCRIPT_DIR=${NANO_V35_EXPERIMENT_DIR:-"${PWD}/examples/multimodal/nano_v35/unification"}
fi
source "${SCRIPT_DIR}/experiment_config.sh"

LM_MCORE_DIR=${LM_MCORE_DIR:-"${BASE_MCORE_CKPT_DIR}/torch"}
OUTPUT_CKPT_DIR=${OUTPUT_CKPT_DIR:-"${WORKSPACE}/checkpoints/${EXPERIMENT_ID}/base_vlm_radio_v4_tp2_ep32"}

LM_ITERATION=${LM_ITERATION:-${ITERATION:-1}}
VISION_ITERATION=${VISION_ITERATION:-1}
OUTPUT_ITERATION=${OUTPUT_ITERATION:-1}
DRY_RUN=${DRY_RUN:-0}

LOGS_DIR="${OUTPUT_CKPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

run_cmd="cd ${CODE_DIR}; python -u examples/multimodal/nano_v35/combine_nano_v35_with_radio.py \
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
    echo "LM checkpoint iteration is missing under ${LM_MCORE_DIR}" >&2
    exit 1
fi
if [[ ! -d "${VISION_CKPT_DIR}/iter_$(printf '%07d' "${VISION_ITERATION}")" ]]; then
    echo "Vision checkpoint iteration is missing under ${VISION_CKPT_DIR}" >&2
    exit 1
fi
if [[ ! -r "${CONTAINER_IMAGE}" ]]; then
    echo "Container image is missing or unreadable: ${CONTAINER_IMAGE}" >&2
    exit 1
fi

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
srun -l --verbose \
    --container-image "${CONTAINER_IMAGE}" \
    --container-mounts "${CONTAINER_MOUNTS}" \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    sh -c "${run_cmd}"
