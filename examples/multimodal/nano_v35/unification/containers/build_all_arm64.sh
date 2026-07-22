#!/bin/bash

# Submit training build -> training validation -> eval build -> eval validation.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

train_job=$(sbatch --parsable "${SCRIPT_DIR}/build_training_arm64.sbatch")
train_validate_job=$(sbatch --parsable \
    --dependency="afterok:${train_job}" \
    --export=ALL,IMAGE_KIND=training \
    "${SCRIPT_DIR}/validate_arm64.sbatch")
eval_job=$(sbatch --parsable \
    --dependency="afterok:${train_validate_job}" \
    "${SCRIPT_DIR}/build_eval_arm64.sbatch")
eval_validate_job=$(sbatch --parsable \
    --dependency="afterok:${eval_job}" \
    --export=ALL,IMAGE_KIND=eval \
    "${SCRIPT_DIR}/validate_arm64.sbatch")

echo "training build job: ${train_job}"
echo "training validation job: ${train_validate_job}"
echo "evaluation build job: ${eval_job}"
echo "evaluation validation job: ${eval_validate_job}"
