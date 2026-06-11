#!/bin/bash

#SBATCH -A nemotron_omni_vision
#SBATCH -p batch
#SBATCH -t 00:30:00
#SBATCH --mem=503480
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --overcommit
#SBATCH --job-name=nano_v35_convert_hf_to_mcore_tp2_ep32

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_WORKER_START=fork

USER_NAME=${SLURM_JOB_USER:-${USER:-$(whoami)}}
SOURCE=$(pwd)

HF_CKPT_DIR=${HF_CKPT_DIR:-"/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/checkpoints/nano_v35_checkpoint_hf"}
MCORE_CKPT_DIR=${MCORE_CKPT_DIR:-"/lustre/fsw/portfolios/llmservice/users/${USER_NAME}/workspace/checkpoints/nano_v35_llm_mcore_tp2_ep32_mtpfix"}
TARGET_TP=${TARGET_TP:-${TP:-2}}
TARGET_EP=${TARGET_EP:-${EP:-32}}
TARGET_ETP=${TARGET_ETP:-1}
MAX_QUEUE_SIZE=${MAX_QUEUE_SIZE:-1}
DRY_RUN=${DRY_RUN:-0}

CONTAINER_IMAGE=${CONTAINER_IMAGE:-"/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/containers/pytorch25.11-moe-avlm-editable-energon-super.sqsh"}

MCORE_TORCH_CKPT_DIR="${MCORE_CKPT_DIR}/torch"
LOGS_DIR="${MCORE_CKPT_DIR}/logs"
mkdir -p "${MCORE_TORCH_CKPT_DIR}" "${LOGS_DIR}"

OPTIONS=" \
    --model-type hybrid \
    --loader hf_moe \
    --saver core \
    --load-dir ${HF_CKPT_DIR} \
    --save-dir ${MCORE_TORCH_CKPT_DIR} \
    --megatron-path ${SOURCE} \
    --max-queue-size ${MAX_QUEUE_SIZE} \
    --target-tensor-parallel-size ${TARGET_TP} \
    --target-expert-parallel-size ${TARGET_EP} \
    --target-expert-tensor-parallel-size ${TARGET_ETP} \
"

run_cmd="cd ${SOURCE}; python -u tools/checkpoint/convert.py ${OPTIONS}"

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
    echo "HF checkpoint directory does not exist: ${HF_CKPT_DIR}"
    exit 1
fi

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
srun -l --verbose \
    --container-image "${CONTAINER_IMAGE}" \
    --container-mounts "/lustre,/scratch" \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    sh -c "export TRITON_CACHE_DIR=/tmp/triton_cache_\${SLURM_NODEID}; ${run_cmd}"
