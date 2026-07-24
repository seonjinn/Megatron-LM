#!/bin/bash
#
# Nano-3.5 SFT model/runtime setup from:
#   nano-3.5-sft-alex-512k-hermes-rerun-jun2-guyueh-mlm-vlm2-rebase.sh
#
# Dataset side is swapped to the Energon multimodal JSONL path:
#   - MetadatasetV2 YAML over WDAI raw messages JSONLs
#   - subflavors.cook: openai_messages_jsonl
#   - online packing through MultiModalTaskEncoder

#SBATCH -p batch
#SBATCH -q normal
#SBATCH --account=nemotron_n4_post
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=64
#SBATCH --time=4:00:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --comment='{"APS": {"auto_resume_mode": "singleton_dependency"}}'
#SBATCH --dependency=singleton
#SBATCH --job-name=nano-3.5-sft-wdai-energon-online-packing

set -euo pipefail

################################################################
### TransformerEngine
################################################################
export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
export NVTE_CPU_OFFLOAD_V1=${NVTE_CPU_OFFLOAD_V1:-1}
export TORCHINDUCTOR_WORKER_START=${TORCHINDUCTOR_WORKER_START:-fork}

################################################################
### UCX
################################################################
export UCX_MEM_MMAP_HOOK_MODE=${UCX_MEM_MMAP_HOOK_MODE:-none}
export UCX_MEM_CUDA_HOOK_MODE=${UCX_MEM_CUDA_HOOK_MODE:-none}
export UCX_MEM_MALLOC_HOOKS=${UCX_MEM_MALLOC_HOOKS:-none}
export UCX_ERROR_SIGNALS=${UCX_ERROR_SIGNALS:-none}

################################################################
### General
################################################################
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export SHELL=/bin/bash

USER=${SLURM_JOB_USER:-${USER}}
NAME=${MODEL_NAME:-${SLURM_JOB_NAME:-nano-3.5-sft-wdai-energon-online-packing}}

OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/fsw/portfolios/llmservice/users/guyueh/sft-runs}
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-/lustre/fsw/portfolios/llmservice/users/guyueh/mac_mirror/megatron-lm-vlm2-rebase-online-packing}
IMAGE=${IMAGE:-/lustre/fsw/portfolios/llmservice/users/guyueh/container_images/megatron_lm_26.04_a6d61fb_energon_732.sqsh}
BINDPCIE_SCRIPT=${BINDPCIE_SCRIPT:-/lustre/fsw/portfolios/llmservice/users/soumyes/sft-runs/code/bindpcie.sh}

WANDB_PROJECT=${WANDB_PROJECT:-nano-3.5-sft-guyueh}
WANDB_ARGS=""
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${HOME}/.bashrc" ]; then
    source <(grep -E '^[[:space:]]*export[[:space:]]+WANDB_API_KEY=' "${HOME}/.bashrc" | tail -n 1)
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY is not set. Disabling W&B logging for this run."
else
    export WANDB_API_KEY
    WANDB_ARGS="--wandb-project ${WANDB_PROJECT} --wandb-exp-name ${NAME}"
fi

RUN_DIR=${RUN_DIR:-${OUTPUT_ROOT}}
LOGS_DIR=${LOGS_DIR:-${RUN_DIR}/logs/${NAME}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints/${NAME}}
DATACACHE_DIR=${DATACACHE_DIR:-${RUN_DIR}/data_cache/${NAME}}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-${RUN_DIR}/tensorboard/${NAME}}
WANDB_SAVE_DIR=${WANDB_SAVE_DIR:-${RUN_DIR}/wandb/${NAME}}
HF_CACHE_DIR=${HF_CACHE_DIR:-${DATACACHE_DIR}/hf_cache}
export HF_HOME=${HF_HOME:-${HF_CACHE_DIR}/home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_CACHE_DIR}/datasets}

mkdir -p "${LOGS_DIR}" "${CHECKPOINT_DIR}" "${DATACACHE_DIR}" "${TENSORBOARD_DIR}" "${WANDB_SAVE_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}"
if [ -n "${WANDB_ARGS}" ]; then
    WANDB_ARGS="${WANDB_ARGS} --wandb-save-dir ${WANDB_SAVE_DIR}"
fi

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton-cache}

BASE_MODEL_PATH=${BASE_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/guyueh/checkpoints/nano3p5-base-lc-reinit-emb-llava-wrapper-mtp2-tp8-ep8-with-mtp}

DATA_TRAIN=${DATA_TRAIN:-/lustre/fsw/portfolios/llmservice/users/guyueh/sft-data/wdai-nano-3-5-epoch-0-050126-jsonl-links/openai_messages.all.yaml}
TOKENIZER_MODEL_PATH=${TOKENIZER_MODEL_PATH:-/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/adithyare/nemotron_super/tokenizer}
TOKENIZER_PROMPT_FORMAT=${TOKENIZER_PROMPT_FORMAT:-nemotron6-moe}
DATALOADER_SEED=${DATALOADER_SEED:-0}

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
    ENV_LOG_FILENAME=${NAME}_${SLURM_JOB_ID}_${DATETIME}.env.log
else
    SCRIPT_PATH=$(realpath "$0")
    ENV_LOG_FILENAME=${NAME}_${DATETIME}.env.log
fi
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")

{
    echo "<< START PATHS >>"
    echo "IMAGE=${IMAGE}"
    echo "BINDPCIE_SCRIPT=${BINDPCIE_SCRIPT}"
    echo "MEGATRON_LM_DIR=${MEGATRON_LM_DIR}"
    echo "BASE_MODEL_PATH=${BASE_MODEL_PATH}"
    echo "DATA_TRAIN=${DATA_TRAIN}"
    echo "TOKENIZER_MODEL_PATH=${TOKENIZER_MODEL_PATH}"
    echo "TOKENIZER_PROMPT_FORMAT=${TOKENIZER_PROMPT_FORMAT}"
    echo "DATALOADER_SEED=${DATALOADER_SEED}"
    echo "MASTER_PORT=${MASTER_PORT:-}"
    echo "ENABLE_ONLINE_PACKING=${ENABLE_ONLINE_PACKING:-1}"
    echo "PACKING_BUFFER_SIZE=${PACKING_BUFFER_SIZE:-10000}"
    echo "PACKING_SEQ_LEN=${PACKING_SEQ_LEN:-${DECODER_SEQ_LEN:-524288}}"
    echo "PACKING_ALGORITHM=${PACKING_ALGORITHM:-balanced_greedy_knapsack}"
    echo "PACKING_ALGORITHM_PARAMETERS=${PACKING_ALGORITHM_PARAMETERS:-}"
    echo "RELAX_THINKING_TRACE_CHECK=${RELAX_THINKING_TRACE_CHECK:-0}"
    echo "RESET_POSITION_IDS_FROM_PACKED_METADATA=${RESET_POSITION_IDS_FROM_PACKED_METADATA:-1}"
    echo "DATALOADER_SEQ_LEN=${DATALOADER_SEQ_LEN:-}"
    echo "USE_MTP=${USE_MTP:-1}"
    echo "NUM_WORKERS=${NUM_WORKERS:-1}"
    echo "MAX_SAMPLES_PER_SEQUENCE=${MAX_SAMPLES_PER_SEQUENCE:-100}"
    echo "SHUFFLE_BUFFER_SIZE=${SHUFFLE_BUFFER_SIZE:-100}"
    echo "SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-1}"
    echo "LOG_PACKED_SEQUENCE_STATS=${LOG_PACKED_SEQUENCE_STATS:-1}"
    echo "ALLOW_LLM_ONLY_CHECKPOINT=${ALLOW_LLM_ONLY_CHECKPOINT:-0}"
    echo "TRAIN_ITERS=${TRAIN_ITERS:-}"
    echo "LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-}"
    echo "LR_DECAY_ITERS=${LR_DECAY_ITERS:-}"
    echo "EXTRA_ARGS=${EXTRA_ARGS:-}"
    echo "TP=${TP:-8}"
    echo "CP=${CP:-8}"
    echo "EP=${EP:-8}"
    echo "ETP=${ETP:-1}"
    echo "PP=${PP:-1}"
    echo "RUN_DIR=${RUN_DIR}"
    echo "LOGS_DIR=${LOGS_DIR}"
    echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
    echo "DATACACHE_DIR=${DATACACHE_DIR}"
    echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}"
    echo "WANDB_SAVE_DIR=${WANDB_SAVE_DIR}"
    echo "HF_HOME=${HF_HOME}"
    echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
    echo "SCRIPT_DIR=${SCRIPT_DIR}"
    echo "<< END PATHS >>"
    echo
    echo "<< START GIT >>"
    git -C "${MEGATRON_LM_DIR}" log --oneline -1 || true
    git -C "${MEGATRON_LM_DIR}" status --porcelain --branch || true
    git -C "${MEGATRON_LM_DIR}" diff || true
    echo "<< END GIT >>"
    echo
    echo "<< START ENV >>"
    env
    echo "<< END ENV >>"
} 2>&1 | tee -a "${LOGS_DIR}/${ENV_LOG_FILENAME}"

################################################################
### Hyperparameters from Nano text SFT
################################################################
PACKING_SEQ_LEN=${PACKING_SEQ_LEN:-${DECODER_SEQ_LEN:-524288}}
DECODER_SEQ_LEN=${DECODER_SEQ_LEN:-${PACKING_SEQ_LEN}}
VISION_SEQ_LEN=${VISION_SEQ_LEN:-256}
DATALOADER_SEQ_LEN=${DATALOADER_SEQ_LEN:-}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-9645437}
TRAIN_ITERS=${TRAIN_ITERS:-}
LR_WARMUP_SAMPLES=${LR_WARMUP_SAMPLES:-150000}
LR_DECAY_SAMPLES=${LR_DECAY_SAMPLES:-9495437}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-0}
LR_DECAY_ITERS=${LR_DECAY_ITERS:-}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-250}
SAVE_RETAIN_INTERVAL=${SAVE_RETAIN_INTERVAL:-500}
GBS=${GBS:-64}
LR=${LR:-2e-5}
MIN_LR=${MIN_LR:-5e-6}
EXIT_DURATION_MINS=${EXIT_DURATION_MINS:-230}

PACKING_BUFFER_SIZE=${PACKING_BUFFER_SIZE:-10000}
PACKING_ALGORITHM=${PACKING_ALGORITHM:-balanced_greedy_knapsack}
PACKING_ALGORITHM_PARAMETERS=${PACKING_ALGORITHM_PARAMETERS:-}
RELAX_THINKING_TRACE_CHECK=${RELAX_THINKING_TRACE_CHECK:-0}
RESET_POSITION_IDS_FROM_PACKED_METADATA=${RESET_POSITION_IDS_FROM_PACKED_METADATA:-1}
ENABLE_ONLINE_PACKING=${ENABLE_ONLINE_PACKING:-1}
USE_MTP=${USE_MTP:-1}
NUM_WORKERS=${NUM_WORKERS:-1}
MAX_SAMPLES_PER_SEQUENCE=${MAX_SAMPLES_PER_SEQUENCE:-100}
SHUFFLE_BUFFER_SIZE=${SHUFFLE_BUFFER_SIZE:-100}
SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-1}
GLOBAL_TOKEN_LOSS_NORMALIZATION=${GLOBAL_TOKEN_LOSS_NORMALIZATION:-0}
LOG_PACKED_SEQUENCE_STATS=${LOG_PACKED_SEQUENCE_STATS:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}
ALLOW_LLM_ONLY_CHECKPOINT=${ALLOW_LLM_ONLY_CHECKPOINT:-0}
TP=${TP:-8}
CP=${CP:-8}
EP=${EP:-8}
ETP=${ETP:-1}
PP=${PP:-1}

if [[ -n "${SPECIAL_TOKENS_OVERRIDE:-}" ]]; then
    SPECIAL_TOKENS=${SPECIAL_TOKENS_OVERRIDE}
else
    # Text-only SFT should not register VLM marker strings such as <img> and </ref>
    # as special tokens; the raw text data can contain them literally.
    SPECIAL_TOKENS="--special-tokens"
fi

MTP_OPTIONS=""
if [[ "${USE_MTP}" -eq 1 ]]; then
    MTP_OPTIONS=" \
    --mtp-use-repeated-layer \
    --mtp-hybrid-override-pattern \"*E\" \
    --mtp-num-layers 2 \
    --mtp-loss-scaling-factor 0.1"
fi

LLM_ONLY_CHECKPOINT_OPTIONS=""
if [[ "${ALLOW_LLM_ONLY_CHECKPOINT}" -eq 1 ]]; then
    LLM_ONLY_CHECKPOINT_OPTIONS="--allow-llm-only-checkpoint"
fi

PRETRAINED_CHECKPOINT_OPTIONS=""
if [[ -n "${BASE_MODEL_PATH}" && "${BASE_MODEL_PATH}" != "none" && "${BASE_MODEL_PATH}" != "NONE" ]]; then
    PRETRAINED_CHECKPOINT_OPTIONS="--pretrained-checkpoint ${BASE_MODEL_PATH}"
fi

CHECKPOINT_OPTIONS=""
if [[ "${SAVE_CHECKPOINTS}" -eq 1 ]]; then
    CHECKPOINT_OPTIONS=" \
    --async-save \
    --save ${CHECKPOINT_DIR} \
    --load ${CHECKPOINT_DIR} \
    --dataloader-save ${CHECKPOINT_DIR}/dataloader \
    --ckpt-format torch_dist \
    --ckpt-fully-parallel-save \
    --ckpt-fully-parallel-load \
    --ckpt-assume-constant-structure \
    --dist-ckpt-save-pre-mcore-014 \
    --use-persistent-ckpt-worker"
fi

THINKING_TRACE_OPTIONS=""
if [[ "${RELAX_THINKING_TRACE_CHECK}" -eq 1 ]]; then
    THINKING_TRACE_OPTIONS="--relax-thinking-trace-check"
fi

PACKED_POSITION_ID_OPTIONS=""
if [[ "${RESET_POSITION_IDS_FROM_PACKED_METADATA}" -eq 1 ]]; then
    PACKED_POSITION_ID_OPTIONS="--reset-position-ids-from-packed-metadata"
fi

DATALOADER_SEQ_LEN_OPTIONS=""
if [[ -n "${DATALOADER_SEQ_LEN}" ]]; then
    DATALOADER_SEQ_LEN_OPTIONS="--dataloader-seq-length ${DATALOADER_SEQ_LEN}"
fi

PACKING_OPTIONS=""
if [[ "${ENABLE_ONLINE_PACKING}" -eq 1 ]]; then
    PACKING_OPTIONS=" \
    --packing-buffer-size ${PACKING_BUFFER_SIZE} \
    --packing-seq-length ${PACKING_SEQ_LEN} \
    --packing-knapsack-algorithm ${PACKING_ALGORITHM}"
    if [[ -n "${PACKING_ALGORITHM_PARAMETERS}" ]]; then
        PACKING_OPTIONS+=" --packing-algorithm-parameters '${PACKING_ALGORITHM_PARAMETERS}'"
    fi
fi

TOKEN_LOSS_NORMALIZATION_OPTIONS=""
case "${GLOBAL_TOKEN_LOSS_NORMALIZATION}" in
    1|true|TRUE|yes|YES)
        TOKEN_LOSS_NORMALIZATION_OPTIONS="--calculate-per-token-loss"
        ;;
    0|false|FALSE|no|NO)
        TOKEN_LOSS_NORMALIZATION_OPTIONS="--no-calculate-per-token-loss"
        ;;
    *)
        echo "GLOBAL_TOKEN_LOSS_NORMALIZATION must be 0/1, true/false, or yes/no; got ${GLOBAL_TOKEN_LOSS_NORMALIZATION}" >&2
        exit 1
        ;;
esac

PACKED_SEQUENCE_STATS_OPTIONS=""
if [[ "${LOG_PACKED_SEQUENCE_STATS}" -eq 1 ]]; then
    PACKED_SEQUENCE_STATS_OPTIONS="--log-packed-sequence-stats"
fi

TRAIN_DURATION_OPTIONS="--train-samples ${TRAIN_SAMPLES}"
if [[ -n "${TRAIN_ITERS}" ]]; then
    TRAIN_DURATION_OPTIONS="--train-iters ${TRAIN_ITERS}"
fi

LR_SCHEDULE_OPTIONS="--lr-warmup-samples ${LR_WARMUP_SAMPLES} --lr-decay-samples ${LR_DECAY_SAMPLES}"
if [[ -n "${TRAIN_ITERS}" ]]; then
    if [[ -z "${LR_DECAY_ITERS}" ]]; then
        LR_DECAY_ITERS=${TRAIN_ITERS}
    fi
    LR_SCHEDULE_OPTIONS="--lr-warmup-iters ${LR_WARMUP_ITERS} --lr-decay-iters ${LR_DECAY_ITERS}"
fi

OPTIONS=" \
    --sft \
    --transformer-impl transformer_engine \
    --use-te \
    --distributed-timeout-minutes 240 \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL_PATH} \
    --tokenizer-prompt-format ${TOKENIZER_PROMPT_FORMAT} \
    ${THINKING_TRACE_OPTIONS} \
    ${PACKED_POSITION_ID_OPTIONS} \
    --data-path ${DATA_TRAIN} \
    --dataloader-type external \
    --use-new-dataloader-path \
    --dataloader-seed ${DATALOADER_SEED} \
    ${PACKING_OPTIONS} \
    --max-samples-per-sequence ${MAX_SAMPLES_PER_SEQUENCE} \
    --shuffle-buffer-size ${SHUFFLE_BUFFER_SIZE} \
    ${TOKEN_LOSS_NORMALIZATION_OPTIONS} \
    ${PACKED_SEQUENCE_STATS_OPTIONS} \
    --prompt-path ${MEGATRON_LM_DIR}/examples/multimodal/manual_prompts.json \
    --patch-dim 16 \
    --img-h 512 \
    --img-w 512 \
    --language-model-type nemotron6-moe \
    --vision-model-type radio \
    ${SPECIAL_TOKENS} \
    --disable-vision-class-token \
    --image-tag-type internvl \
    --eod-mask-loss \
    --freeze-ViT \
    --freeze-vision-projection \
    --allow-missing-vision-projection-checkpoint \
    ${LLM_ONLY_CHECKPOINT_OPTIONS} \
    --allow-large-videos \
    --pixel-shuffle \
    --use-tiling \
    --max-num-tiles 12 \
    --use-thumbnail \
    --recompute-granularity selective \
    --recompute-modules moe \
    --fine-grained-activation-offloading \
    --offload-modules moe_act \
    --context-parallel-size ${CP} \
    --tensor-model-parallel-size ${TP} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size ${ETP} \
    --pipeline-model-parallel-size ${PP} \
    --hybrid-override-pattern MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME \
    ${MTP_OPTIONS} \
    ${PRETRAINED_CHECKPOINT_OPTIONS} \
    --save-interval ${SAVE_INTERVAL} \
    --save-retain-interval ${SAVE_RETAIN_INTERVAL} \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style cosine \
    ${TRAIN_DURATION_OPTIONS} \
    ${LR_SCHEDULE_OPTIONS} \
    --seq-length ${VISION_SEQ_LEN} \
    --decoder-seq-length ${DECODER_SEQ_LEN} \
    ${DATALOADER_SEQ_LEN_OPTIONS} \
    --max-position-embeddings ${DECODER_SEQ_LEN} \
    --log-interval ${LOG_INTERVAL} \
    --micro-batch-size 1 \
    --global-batch-size ${GBS} \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --high-priority-stream-groups ep \
    --manual-gc-interval 10 \
    --ddp-num-buckets 8 \
    --manual-gc \
    --moe-permute-fusion \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl native \
    --use-fused-weighted-squared-relu \
    --moe-token-dispatcher-type alltoall \
    --moe-shared-expert-overlap \
    --moe-router-score-function sigmoid \
    --moe-grouped-gemm \
    --num-experts 128 \
    --moe-router-topk 6 \
    --moe-aux-loss-coeff 1e-4 \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-shared-expert-intermediate-size 3712 \
    --attention-backend flash \
    --num-workers ${NUM_WORKERS} \
    --disable-gloo-process-groups \
    ${CHECKPOINT_OPTIONS} \
    --squared-relu \
    --exit-duration-in-mins ${EXIT_DURATION_MINS} \
    --rerun-mode validate_results \
    --no-create-attention-mask-in-dataloader \
    --sequence-parallel \
    --use-distributed-optimizer \
    --override-opt-param-scheduler \
    --mamba-num-heads 64 \
    --mamba-head-dim 64 \
    --is-hybrid-model \
    --untie-embeddings-and-output-weights \
    --init-method-std 0.0173 \
    --position-embedding-type none \
    --num-layers 52 \
    --hidden-size 2688 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 2 \
    --ffn-hidden-size 1856 \
    --kv-channels 128 \
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
    --log-timers-to-tensorboard \
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
    --eval-interval 999999999 \
    --eval-iters 0 \
    --bf16 \
    --use-mcore-models \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    ${WANDB_ARGS} \
    --dist-ckpt-strictness log_unexpected \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    ${EXTRA_ARGS}"

RUN_CMD="python -u ${MEGATRON_LM_DIR}/examples/multimodal/train.py ${OPTIONS}"
LAUNCH_CMD="${BINDPCIE_SCRIPT} --cpu=node --mem=node -- ${RUN_CMD}"

echo "RUN_CMD=${RUN_CMD}" 2>&1 | tee -a "${LOGS_DIR}/${ENV_LOG_FILENAME}"

if [[ ${DRY_RUN:-0} -eq 1 ]]; then
    exit 0
fi

srun -l \
     --mpi=none \
     --no-container-mount-home \
     --container-image="${IMAGE}" \
     --container-mounts="/lustre:/lustre" \
     --container-env=UCX_MEM_MMAP_HOOK_MODE,UCX_MEM_CUDA_HOOK_MODE,UCX_MEM_MALLOC_HOOKS,UCX_ERROR_SIGNALS,NVTE_CPU_OFFLOAD_V1,NVTE_FWD_LAYERNORM_SM_MARGIN,NVTE_BWD_LAYERNORM_SM_MARGIN,TORCHINDUCTOR_WORKER_START,PYTORCH_CUDA_ALLOC_CONF,OMP_NUM_THREADS,CUDA_DEVICE_MAX_CONNECTIONS,MASTER_PORT,SLURM_LOCALID,WANDB_API_KEY,HF_HOME,HF_DATASETS_CACHE \
     --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
     sh -c "${LAUNCH_CMD}"
