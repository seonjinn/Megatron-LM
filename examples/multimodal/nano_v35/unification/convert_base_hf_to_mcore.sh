#!/bin/bash

#SBATCH -A nemotron_n4_post
#SBATCH -p batch
#SBATCH -t 00:30:00
#SBATCH --mem=503480
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --overcommit
#SBATCH --job-name=nano35_base3p5_convert

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ! -f "${SCRIPT_DIR}/experiment_config.sh" ]]; then
    SCRIPT_DIR=${NANO_V35_EXPERIMENT_DIR:-"${PWD}/examples/multimodal/nano_v35/unification"}
fi
source "${SCRIPT_DIR}/experiment_config.sh"

HF_CKPT_DIR=${HF_CKPT_DIR:-"${BASE_HF_CKPT_DIR}"}
MCORE_CKPT_DIR=${MCORE_CKPT_DIR:-"${BASE_MCORE_CKPT_DIR}"}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCHINDUCTOR_WORKER_START=${TORCHINDUCTOR_WORKER_START:-fork}

TARGET_TP=${TARGET_TP:-${TP:-2}}
TARGET_EP=${TARGET_EP:-${EP:-32}}
TARGET_ETP=${TARGET_ETP:-1}
MAX_QUEUE_SIZE=${MAX_QUEUE_SIZE:-1}
DRY_RUN=${DRY_RUN:-0}

MCORE_TORCH_CKPT_DIR="${MCORE_CKPT_DIR}/torch"
LOGS_DIR="${MCORE_CKPT_DIR}/logs"
mkdir -p "${MCORE_TORCH_CKPT_DIR}" "${LOGS_DIR}"

OPTIONS=" \
    --model-type hybrid \
    --loader hf_moe \
    --saver core \
    --load-dir ${HF_CKPT_DIR} \
    --save-dir ${MCORE_TORCH_CKPT_DIR} \
    --megatron-path ${CODE_DIR} \
    --max-queue-size ${MAX_QUEUE_SIZE} \
    --target-tensor-parallel-size ${TARGET_TP} \
    --target-expert-parallel-size ${TARGET_EP} \
    --target-expert-tensor-parallel-size ${TARGET_ETP} \
"

run_cmd="cd ${CODE_DIR}; python -u tools/checkpoint/convert.py ${OPTIONS}"

echo "HF_CKPT_DIR=${HF_CKPT_DIR}"
echo "MCORE_TORCH_CKPT_DIR=${MCORE_TORCH_CKPT_DIR}"
echo "TARGET_TP=${TARGET_TP}"
echo "TARGET_EP=${TARGET_EP}"
echo "TARGET_ETP=${TARGET_ETP}"
echo "${run_cmd}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    exit 0
fi

if [[ ! -d "${HF_CKPT_DIR}" ]]; then
    echo "HF checkpoint directory does not exist: ${HF_CKPT_DIR}" >&2
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
    sh -c "export TRITON_CACHE_DIR=/tmp/triton_cache_\${SLURM_NODEID}; ${run_cmd}"
