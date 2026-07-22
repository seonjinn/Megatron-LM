#!/bin/bash

#SBATCH -A nemotron_n4_post
#SBATCH -p batch
#SBATCH --qos=interactive
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --job-name=nano35-build-eval-arm64

set -euo pipefail

SCRIPT_DIR=${NANO_V35_CONTAINER_SCRIPT_DIR:-${SLURM_SUBMIT_DIR}/examples/multimodal/nano_v35/unification/containers}
OUTPUT_DIR=${OUTPUT_DIR:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/${USER}/containers}
TRAIN_IMAGE=${TRAIN_IMAGE:-${OUTPUT_DIR}/pytorch25.11-moe-avlm-editable-energon-super-triton35.sqsh}
EVAL_IMAGE=${EVAL_IMAGE:-${OUTPUT_DIR}/pytorch25.11-moe-avlm-editable-energon-super-eval.sqsh}

if [[ ! -s "${TRAIN_IMAGE}" ]]; then
    echo "Training image is missing: ${TRAIN_IMAGE}" >&2
    exit 1
fi
if [[ -e "${EVAL_IMAGE}" ]]; then
    echo "Refusing to overwrite existing image: ${EVAL_IMAGE}" >&2
    exit 1
fi

echo "Training image: ${TRAIN_IMAGE}"
echo "Evaluation image: ${EVAL_IMAGE}"

srun --kill-on-bad-exit=1 \
    --container-image="${TRAIN_IMAGE}" \
    --container-mounts=/lustre \
    --container-mount-home \
    --container-remap-root \
    --container-writable \
    --container-save="${EVAL_IMAGE}" \
    --no-container-entrypoint \
    bash "${SCRIPT_DIR}/install_eval_arm64.sh"

test -s "${EVAL_IMAGE}"
ls -lh "${EVAL_IMAGE}"
