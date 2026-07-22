#!/bin/bash

#SBATCH -A nemotron_n4_post
#SBATCH -p batch
#SBATCH --qos=interactive
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --job-name=nano35-build-train-arm64

set -euo pipefail

SCRIPT_DIR=${NANO_V35_CONTAINER_SCRIPT_DIR:-${SLURM_SUBMIT_DIR}/examples/multimodal/nano_v35/unification/containers}
OUTPUT_DIR=${OUTPUT_DIR:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/${USER}/containers}
BASE_IMAGE=${BASE_IMAGE:-nvcr.io#nvidia/pytorch:25.11-py3}
TRAIN_IMAGE=${TRAIN_IMAGE:-${OUTPUT_DIR}/pytorch25.11-moe-avlm-editable-energon-super-triton35.sqsh}

mkdir -p "${OUTPUT_DIR}"
if [[ -e "${TRAIN_IMAGE}" ]]; then
    echo "Refusing to overwrite existing image: ${TRAIN_IMAGE}" >&2
    exit 1
fi

echo "Base image: ${BASE_IMAGE}"
echo "Training image: ${TRAIN_IMAGE}"

srun --kill-on-bad-exit=1 \
    --container-image="${BASE_IMAGE}" \
    --container-mounts=/lustre \
    --container-mount-home \
    --container-remap-root \
    --container-writable \
    --container-save="${TRAIN_IMAGE}" \
    --no-container-entrypoint \
    bash "${SCRIPT_DIR}/install_training_arm64.sh"

test -s "${TRAIN_IMAGE}"
ls -lh "${TRAIN_IMAGE}"
