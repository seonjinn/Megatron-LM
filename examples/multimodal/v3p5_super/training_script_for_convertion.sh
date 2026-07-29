#!/bin/bash

#SBATCH -p batch
#SBATCH -q short
#SBATCH --account=nemotron_n4_post
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=32
#SBATCH --segment=2
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --job-name=super35_distcp_to_torch

set -euo pipefail

# Convert the Super 3.5 repeated-MTP checkpoint from torch_dist (distcp) to
# Megatron's legacy torch format.
#
# The default input is:
#   /lustre/fsw/portfolios/llmservice/users/wdai/checkpoints/
#     super-repeated-mtp-reinit-embeddings
#
# Submit with:
#   sbatch examples/multimodal/v3p5_super/training_script_for_convertion.sh
#
# The converted checkpoint is written below CKPT_CONVERT_SAVE in a "torch"
# subdirectory. Override the defaults through exported environment variables,
# for example:
#   CKPT_CONVERT_SAVE=/path/to/output sbatch training_script_for_convertion.sh

# Scrub inherited PMIx/MPI wiring so a submission made from inside an allocation
# does not leak that allocation's process-manager configuration into Pyxis.
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
unset SLURM_MPI_TYPE SLURM_PMIX_MAPPING_SERV SLURM_PMIXP_ABORT_AGENT_PORT
while IFS= read -r variable_name; do
    unset "${variable_name}"
done < <(compgen -e | grep -E '^(PMIX_|OMPI_MCA_|I_MPI_)' || true)
export SLURM_MPI_TYPE=none

export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export TORCHINDUCTOR_WORKER_START=fork
export QUANTIZATION_TYPE_DEBUG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16

export NCCL_NVLS_ENABLE=0
export NCCL_PROTO=simple
export NCCL_SHM_DISABLE=1
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,BOOTSTRAP}

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton-cache}

SOURCE_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -n "${MEGATRON_LM_DIR:-}" ]]; then
    MEGATRON_LM_DIR=$(cd -- "${MEGATRON_LM_DIR}" && pwd)
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pretrain_mamba.py" ]]; then
    MEGATRON_LM_DIR=$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)
else
    MEGATRON_LM_DIR=$(cd -- "${SOURCE_SCRIPT_DIR}/../../.." && pwd)
fi
IMAGE=${IMAGE:-"/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/adithyare/containers/pt_ultra_mamba_ssmv230_23jan28.sqsh"}

USER_NAME=${SLURM_JOB_USER:-${USER:-$(whoami)}}
NAME=${SLURM_JOB_NAME:-super35_distcp_to_torch}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/lustre/fs1/portfolios/nemotron/projects/nemotron_omni_vision/users/${USER_NAME}/workspace/output"}
RUN_DIR=${RUN_DIR:-"${OUTPUT_ROOT}/${NAME}"}

CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/lustre/fsw/portfolios/llmservice/users/wdai/checkpoints/super-repeated-mtp-reinit-embeddings"}
FINETUNE_DIR=${FINETUNE_DIR:-"${RUN_DIR}/finetune_checkpoint"}
CKPT_CONVERT_SAVE=${CKPT_CONVERT_SAVE:-"${RUN_DIR}/converted_checkpoint"}

SKIP_SAVE=${SKIP_SAVE:-1}
NO_SAVE_OPTIM=${NO_SAVE_OPTIM:-1}
NO_SAVE_RNG=${NO_SAVE_RNG:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}

LOGS_DIR=${LOGS_DIR:-"${RUN_DIR}/logs"}
DATACACHE_DIR=${DATACACHE_DIR:-"${RUN_DIR}/data_cache"}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"${RUN_DIR}/tensorboard"}

TOKENIZER_MODEL_PATH=${TOKENIZER_MODEL_PATH:-"/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/adithyare/nemotron_super/tokenizer"}
BLEND_PATH=${BLEND_PATH:-"/lustre/fsw/portfolios/llmservice/users/venkats/training_actual_0603/launch_scripts/blends/sft_mix_v18_512k.json"}

if [[ ! -f "${MEGATRON_LM_DIR}/pretrain_mamba.py" ]]; then
    echo "ERROR: pretrain_mamba.py is missing from MEGATRON_LM_DIR=${MEGATRON_LM_DIR}" >&2
    exit 1
fi
if [[ ! -f "${IMAGE}" ]]; then
    echo "ERROR: container image is missing: ${IMAGE}" >&2
    exit 1
fi
if [[ ! -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "ERROR: input checkpoint tracker is missing: ${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" >&2
    exit 1
fi
if [[ "${CKPT_CONVERT_SAVE}" == "${CHECKPOINT_DIR}" ]]; then
    echo "ERROR: CKPT_CONVERT_SAVE must not be the input CHECKPOINT_DIR." >&2
    exit 1
fi

mkdir -p "${LOGS_DIR}" "${DATACACHE_DIR}" "${TENSORBOARD_DIR}"

if [[ -n "${CKPT_CONVERT_SAVE}" ]]; then
    mkdir -p "${CKPT_CONVERT_SAVE}"
    CHECKPOINT_ARGS=" \
        --load ${CHECKPOINT_DIR} \
        --ckpt-format torch_dist \
        --auto-detect-ckpt-format \
        --ckpt-convert-format torch \
        --ckpt-convert-save ${CKPT_CONVERT_SAVE} \
        --no-use-tokenizer-model-from-checkpoint-args \
        --no-load-optim \
        --no-load-rng \
        --no-save-optim \
        --no-save-rng \
    "
else
    CHECKPOINT_ARGS=" \
        --pretrained-checkpoint ${CHECKPOINT_DIR} \
        --load ${FINETUNE_DIR} \
        --ckpt-format torch \
    "

    if [[ "${SKIP_SAVE}" -eq 0 ]]; then
        CHECKPOINT_ARGS+=" \
            --save ${FINETUNE_DIR} \
            --dataloader-save ${FINETUNE_DIR}/dataloader \
            --save-interval ${SAVE_INTERVAL} \
        "
        if [[ "${NO_SAVE_OPTIM}" -eq 1 ]]; then
            CHECKPOINT_ARGS+=" --no-save-optim"
        fi
        if [[ "${NO_SAVE_RNG}" -eq 1 ]]; then
            CHECKPOINT_ARGS+=" --no-save-rng"
        fi
    fi
fi

if [[ -n "${CKPT_CONVERT_SAVE}" ]]; then
    CHECKPOINT_IO_ARGS="--ckpt-fully-parallel-load"
else
    CHECKPOINT_IO_ARGS=" \
        --ckpt-fully-parallel-save \
        --ckpt-fully-parallel-load \
        --ckpt-assume-constant-structure \
        --use-persistent-ckpt-worker \
    "
fi

SEQ_LEN=${SEQ_LEN:-524288}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-192000}
LR_WARMUP_SAMPLES=${LR_WARMUP_SAMPLES:-100}
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES - LR_WARMUP_SAMPLES))
LOG_INTERVAL=${LOG_INTERVAL:-10}
GBS=${GBS:-32}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-2e-6}

# The input checkpoint's saved args record world_size=128, TP=8, CP=8,
# EP=8, ETP=1, and PP=1. The Slurm request and parallelism below reproduce
# that topology while the distributed checkpoint is loaded.
OPTIONS=" \
    --use-checkpoint-args \
    --calculate-per-token-loss \
    --sft \
    --sft-tokenizer-prompt-format identity \
    --distributed-timeout-minutes 120 \
    --num-dataset-builder-threads 32 \
    --tokenizer-type SFTTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL_PATH} \
    \
    --recompute-granularity selective \
    --recompute-modules moe \
    --mtp-use-repeated-layer \
    \
    --context-parallel-size 8 \
    --tensor-model-parallel-size 8 \
    --expert-model-parallel-size 8 \
    --expert-tensor-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --mtp-hybrid-override-pattern \"*E\" \
    \
    ${CHECKPOINT_ARGS} \
    ${CHECKPOINT_IO_ARGS} \
    --save-retain-interval ${SAVE_INTERVAL} \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style constant \
    --train-samples ${TRAIN_SAMPLES} \
    --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
    --lr-decay-samples ${LR_DECAY_SAMPLES} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --log-interval ${LOG_INTERVAL} \
    --micro-batch-size 1 \
    --global-batch-size ${GBS} \
    --overlap-grad-reduce \
    --overlap-param-gather \
    \
    --mtp-num-layers 2 \
    --mtp-loss-scaling-factor 0.3 \
    \
    --cuda-graph-scope mamba attn moe_router \
    --te-rng-tracker \
    --high-priority-stream-groups ep \
    --manual-gc-interval 10 \
    --ddp-num-buckets 10 \
    --manual-gc \
    \
    --moe-latent-size 1024 \
    --moe-permute-fusion \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl native \
    --use-fused-weighted-squared-relu \
    \
    --moe-token-dispatcher-type alltoall \
    --moe-router-score-function sigmoid \
    --moe-grouped-gemm \
    --num-experts 512 \
    --moe-router-topk 22 \
    --moe-aux-loss-coeff 1e-4 \
    --moe-router-topk-scaling-factor 5.0 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-shared-expert-intermediate-size 5376 \
    \
    --attention-backend flash \
    --num-workers 1 \
    --disable-gloo-process-groups \
    --squared-relu \
    --no-mmap-bin-files \
    --exit-duration-in-mins 190 \
    --no-create-attention-mask-in-dataloader \
    --sequence-parallel \
    --use-distributed-optimizer \
    --override-opt_param-scheduler \
    \
    --mamba-num-heads 128 \
    --is-hybrid-model \
    --untie-embeddings-and-output-weights \
    --init-method-std 0.014 \
    --position-embedding-type none \
    --num-layers 88 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 2 \
    --ffn-hidden-size 2688 \
    --kv-channels 128 \
    \
    --per-split-data-args-path ${BLEND_PATH} \
    --data-cache-path ${DATACACHE_DIR} \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --disable-bias-linear \
    --normalization RMSNorm \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --log-progress \
    --log-energy \
    --log-memory-interval 200 \
    --logging-level 20 \
    --log-straggler \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 \
    --check-weight-hash-across-dp-replicas-interval 20000 \
    --ddp-pad-buckets-for-high-nccl-busbw \
    --timing-log-option minmax \
    --eval-interval 1000 \
    --eval-iters 14 \
    --bf16 \
    --use-mcore-models \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --dist-ckpt-strictness log_unexpected \
    --tensorboard-dir ${TENSORBOARD_DIR}"

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
LOG_FILE="${LOGS_DIR}/${NAME}_${SLURM_JOB_ID:-no_slurm}_${DATETIME}.log"
RUN_CMD="python -u ${MEGATRON_LM_DIR}/pretrain_mamba.py ${OPTIONS}"

{
    echo "MEGATRON_LM_DIR=${MEGATRON_LM_DIR}"
    echo "IMAGE=${IMAGE}"
    echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
    echo "CKPT_CONVERT_SAVE=${CKPT_CONVERT_SAVE}"
    echo "Expected converted checkpoint: ${CKPT_CONVERT_SAVE}/torch"
    git -C "${MEGATRON_LM_DIR}" log --oneline -1
} | tee -a "${LOG_FILE}"

if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
    printf '%s\n' "${RUN_CMD}"
    exit 0
fi

srun -l \
    --mpi=none \
    --no-container-mount-home \
    --container-image="${IMAGE}" \
    --container-mounts="/lustre:/lustre" \
    --container-env=NVTE_FWD_LAYERNORM_SM_MARGIN,NVTE_BWD_LAYERNORM_SM_MARGIN,TORCHINDUCTOR_WORKER_START,QUANTIZATION_TYPE_DEBUG,PYTORCH_CUDA_ALLOC_CONF,OMP_NUM_THREADS,TRITON_CACHE_DIR,NCCL_NVLS_ENABLE,NCCL_PROTO,NCCL_SHM_DISABLE,NCCL_P2P_NET_CHUNKSIZE,NCCL_IB_SL,NCCL_IB_TIMEOUT,NCCL_DEBUG,NCCL_DEBUG_SUBSYS \
    --output="${LOG_FILE}" \
    sh -c "${RUN_CMD}"

echo "Conversion completed: ${CKPT_CONVERT_SAVE}/torch"
