#!/bin/bash

# Slurm script to convert hybrid HF models to mcore format
# 
# Usage:
#   sbatch tools/checkpoint/convert_hf_to_mcore.sh <input_dir> <output_dir> [target_tp]
#
# Arguments:
#   input_dir  - Path to input HF checkpoint directory
#   output_dir - Path to output mcore checkpoint directory  
#   target_tp  - Target tensor parallel size (optional, default: 1)
#
# Examples:
#   sbatch tools/checkpoint/convert_hf_to_mcore.sh /path/to/input /path/to/output
#   sbatch tools/checkpoint/convert_hf_to_mcore.sh /path/to/input /path/to/output 4
#
# The script will convert a hybrid HF model to mcore format using the conversion command:
# python tools/checkpoint/convert.py --model-type hybrid --loader hf_hybrid --saver core \
#   --load-dir <input_dir> --save-dir <output_dir> --megatron-path . \
#   --max-queue-size 1 --target-tensor-parallel-size <target_tp>

#SBATCH -A llmservice_fm_vision
#SBATCH -p batch,batch_large,batch_short
#SBATCH -t 00:30:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --gpus-per-node=8
#SBATCH --job-name=convert_hf_to_mcore

export CUDA_DEVICE_MAX_CONNECTIONS=1

if [[ -n "${SLURM_JOB_USER:-}" ]]; then
    USER="${SLURM_JOB_USER}"
else
    USER="$(whoami)"
fi

# Parse command line arguments
if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch $0 <hf_ckpt_dir> <mcore_ckpt_dir> [target_tp]"
    echo "  hf_ckpt_dir  - Path to input HF checkpoint directory"
    echo "  mcore_ckpt_dir - Path to output mcore checkpoint directory"
    echo "  target_tp  - Target tensor parallel size (optional, default: 1)"
    exit 1
fi

HF_CKPT_DIR="$1"
MCORE_CKPT_DIR="$2"
TARGET_TP="${3:-1}"  # Default to 1 if not provided

MAX_QUEUE_SIZE=1     # Fixed value
MCORE_TORCH_CKPT_DIR="${MCORE_CKPT_DIR}/torch"
MCORE_DIST_CKPT_DIR="${MCORE_CKPT_DIR}/dist"

SOURCE=`pwd`
LOGS_DIR="${MCORE_CKPT_DIR}/logs"

# Create logs directory if it doesn't exist
mkdir -p ${LOGS_DIR}

# Conversion command
OPTIONS=" \
    --model-type hybrid \
    --loader hf_hybrid \
    --saver core \
    --load-dir ${HF_CKPT_DIR} \
    --save-dir ${MCORE_TORCH_CKPT_DIR} \
    --megatron-path . \
    --max-queue-size ${MAX_QUEUE_SIZE} \
    --target-tensor-parallel-size ${TARGET_TP} \
"

# Validate input directory exists
if [ ! -d "${HF_CKPT_DIR}" ]; then
    echo "Error: Input directory does not exist: ${HF_CKPT_DIR}"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "${MCORE_TORCH_CKPT_DIR}"
mkdir -p "${MCORE_DIST_CKPT_DIR}"

echo "Converting HF hybrid model to mcore format..."
echo "Input directory: ${HF_CKPT_DIR}"
echo "Output mcore torch directory: ${MCORE_TORCH_CKPT_DIR}"
echo "Output mcore dist directory: ${MCORE_DIST_CKPT_DIR}"
echo "Target tensor parallel size: ${TARGET_TP}"
echo "Max queue size: ${MAX_QUEUE_SIZE}"

run_cmd="cd ${SOURCE}; python -u tools/checkpoint/convert.py ${OPTIONS}"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

srun -l --verbose \
--container-image /lustre/fsw/portfolios/llmservice/users/matthieul/docker/adlr+megatron-lm+pytorch+nemotron5p5-apr2025-nvrx-patchedte+datasets+convert_hf_to_mcore.sqsh \
--container-mounts "/lustre" \
--output=${LOGS_DIR}/%x_%j_$DATETIME.log \
--no-container-mount-home \
sh -c "${run_cmd}; bash tools/checkpoint/convert_legacy_to_dist.sh ${MCORE_TORCH_CKPT_DIR} ${MCORE_DIST_CKPT_DIR} ${TARGET_TP}"

echo "Conversion completed. Check logs at: ${LOGS_DIR}/" 
