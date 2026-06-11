#!/bin/bash

#SBATCH -A nemotron_omni_vision
#SBATCH -p batch
#SBATCH -t 00:30:00
#SBATCH --mem=251740
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --overcommit
#SBATCH --job-name=nano_v35_combine_radio_v4

set -euo pipefail

USER_NAME=${SLURM_JOB_USER:-${USER:-$(whoami)}}
SOURCE=$(pwd)

LM_MCORE_DIR=${LM_MCORE_DIR:-"/lustre/fsw/portfolios/llmservice/users/${USER_NAME}/workspace/checkpoints/nano_v35_llm_mcore_tp2_ep32_mtpfix/torch"}
VISION_CKPT_DIR=${VISION_CKPT_DIR:-${RADIO_CKPT_DIR:-"/lustre/fsw/portfolios/llmservice/users/tpoon/checkpoints/c-radio-v4-h-rc2-tp2"}}
OUTPUT_CKPT_DIR=${OUTPUT_CKPT_DIR:-"/lustre/fsw/portfolios/llmservice/users/${USER_NAME}/workspace/checkpoints/nano_v35_vlm/nano_v35_moe_tp2_ep32_radio_v4_mtpfix"}
LM_ITERATION=${LM_ITERATION:-${ITERATION:-1}}
VISION_ITERATION=${VISION_ITERATION:-1}
OUTPUT_ITERATION=${OUTPUT_ITERATION:-1}
DRY_RUN=${DRY_RUN:-0}

CONTAINER_IMAGE=${CONTAINER_IMAGE:-"/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/containers/pytorch25.11-moe-avlm-editable-energon-super.sqsh"}
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

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
srun -l --verbose \
    --container-image "${CONTAINER_IMAGE}" \
    --container-mounts "/lustre,/scratch" \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    sh -c "${run_cmd}"
