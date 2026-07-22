#!/bin/bash
#SBATCH -A llmservice_fm_vision
#SBATCH -p cpu
#SBATCH -t 00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --job-name=nano35-dss-mix-data-smoke

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ! -f "${SCRIPT_DIR}/experiment_config.sh" ]]; then
    SCRIPT_DIR=${NANO_V35_EXPERIMENT_DIR:-"${PWD}/examples/multimodal/nano_v35/unification"}
fi
source "${SCRIPT_DIR}/experiment_config.sh"

export MODEL_NAME=${MODEL_NAME:-"${EXPERIMENT_ID}_dss_text_mix_data_smoke"}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${BASE_VLM_CKPT_DIR}"}
export DATA_TRAIN=${DATA_TRAIN:-"${SCRIPT_DIR}/smoke_dss_text_mix.yaml"}
export TRAIN_ENTRYPOINT="${CODE_DIR}/examples/multimodal/iter_data.py"

export ENABLE_WANDB=0
export SKIP_SAVE=1
export NUM_GPU=1
export NW=0
export BZ=1
export MBZ=1
export TP=1
export EP=1
export USE_PACKING=0
export USE_CHECKPOINT_ARGS=0
export USE_SEQUENCE_PARALLEL=0
export USE_MOE_GROUPED_GEMM=0
export ITER_DATA_MAX_ITERS=${ITER_DATA_MAX_ITERS:-64}
export ALLOW_MISSING_VISION_PROJECTION=1
export CUSTOM_ARGS="${CUSTOM_ARGS:-} --no-load-optim --no-load-rng"

# iter_data initializes its one-process Gloo group before Megatron translates
# Slurm rank variables, so provide the trivial env:// rendezvous explicitly.
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29517}

exec "${SCRIPT_DIR}/sft_vlm.sh"
