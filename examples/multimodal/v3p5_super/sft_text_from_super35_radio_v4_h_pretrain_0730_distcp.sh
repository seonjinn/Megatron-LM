#!/bin/bash

#SBATCH --account=nemotron_omni_vision
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --segment=16
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --dependency=singleton
#SBATCH --job-name=super35_radio_v4_h_text_sft_0731_distcp

set -euo pipefail

# Text SFT for the Super 3.5 model after RADIO v4-h vision-adapter
# pretraining. Load the converted torch_dist checkpoint with TP8/CP16/EP64.
#
# Language-model architecture and parallelism follow the Super 3.5 text run.
# Data, multimodal processing, packing, and optimization settings follow:
#
#   examples/multimodal/nano_v35/unification/sft_vlm_from_honest_dolphin.sh
#
# This stage intentionally does not freeze the language or vision model.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_CODE_DIR=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
if [[ -n "${SUPER_V35_CODE_DIR:-}" ]]; then
    CODE_DIR=$(cd -- "${SUPER_V35_CODE_DIR}" && pwd)
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/multimodal/train.py" ]]; then
    CODE_DIR=$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)
else
    CODE_DIR="${SCRIPT_CODE_DIR}"
fi

TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT:-"${CODE_DIR}/examples/multimodal/train.py"}
if [[ ! -f "${TRAIN_ENTRYPOINT}" ]]; then
    echo "ERROR: Could not resolve the Megatron-LM checkout; ${TRAIN_ENTRYPOINT} is missing." >&2
    exit 1
fi

USER_NAME=${SLURM_JOB_USER:-${USER:-$(whoami)}}
WORKSPACE=${WORKSPACE:-"/lustre/fs1/portfolios/nemotron/projects/nemotron_omni_vision/users/${USER_NAME}"}
OUTPUT_BASE=${OUTPUT_BASE:-"${WORKSPACE}/workspace/output"}

CONTAINER_IMAGE=${CONTAINER_IMAGE:-"/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/containers/pytorch25.11-moe-avlm-editable-energon-super-triton35.sqsh"}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-"/lustre"}

export NVDATASET_GROUPID=${NVDATASET_GROUPID:-omni_vision}
export NVDATASET_CACHE_DIR="/home/svc-dss/cache/nemotron"
case ",${CONTAINER_MOUNTS}," in
    *,"${NVDATASET_CACHE_DIR}:${NVDATASET_CACHE_DIR}",*) ;;
    *) CONTAINER_MOUNTS+=",${NVDATASET_CACHE_DIR}:${NVDATASET_CACHE_DIR}" ;;
esac

VISION_PRETRAIN_MODEL_NAME=${VISION_PRETRAIN_MODEL_NAME:-"super35_radio_v4_h_tp8_ep8_vision_adapter_pretrain_1377_dss_0730"}
MODEL_NAME=${MODEL_NAME:-"super35_radio_v4_h_text_sft_0731_distcp"}
VISION_PRETRAIN_CKPT_DIR=${VISION_PRETRAIN_CKPT_DIR:-"${OUTPUT_BASE}/${VISION_PRETRAIN_MODEL_NAME}/super35_llava_torch_to_distcp_v2/checkpoints/torch_dist"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${VISION_PRETRAIN_CKPT_DIR}"}
DATA_TRAIN=${DATA_TRAIN:-"${CODE_DIR}/examples/multimodal/v3_baseline_dss/sft_combined_512k_vlm_v18mix60.yaml"}

# This tokenizer has the same 131072 token-to-ID mapping as the Super tokenizer,
# except that reserved IDs 18-26 are renamed to the nine multimodal markup tokens.
TOKENIZER_MODEL=${TOKENIZER_MODEL:-"/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/repos_nanov35/resources/tokenizer/nano_v35_sft_v10_closethink_unmask_orig6k_vlm"}
TOKENIZER_PROMPT_FORMAT=${TOKENIZER_PROMPT_FORMAT:-"nemotron6-moe"}

OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"
FINETUNE_DIR="${OUTPUT}/checkpoints"
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"
WANDB_DIR="${OUTPUT}/wandb"

DRY_RUN=${DRY_RUN:-0}

for required_path in \
    "${TRAIN_ENTRYPOINT}" \
    "${TOKENIZER_MODEL}" \
    "${DATA_TRAIN}" \
    "${CONTAINER_IMAGE}"; do
    if [[ ! -r "${required_path}" ]]; then
        echo "ERROR: Required training path is not readable: ${required_path}" >&2
        exit 1
    fi
done

if [[ ! -r "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "WARNING: Pretraining checkpoint is not ready yet: ${CHECKPOINT_DIR}" >&2
    else
        echo "ERROR: Pretraining checkpoint tracker is not readable: ${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" >&2
        echo "Wait for ${VISION_PRETRAIN_MODEL_NAME} to save a checkpoint, or set CHECKPOINT_DIR explicitly." >&2
        exit 1
    fi
fi

mkdir -p "${FINETUNE_DIR}" "${LOGS_DIR}" "${TENSORBOARD_DIR}" "${WANDB_DIR}"

export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/examples/multimodal:${PYTHONPATH:-}"

# Required for pre-Blackwell tensor parallelism greater than one.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export UB_TIMEOUT=${UB_TIMEOUT:-720}
export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
export TORCHINDUCTOR_WORKER_START=${TORCHINDUCTOR_WORKER_START:-fork}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

# Transport settings used by the Super 3.5 text run and TP8/EP8 conversion.
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_PROTO=${NCCL_PROTO:-simple}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
export NCCL_IB_SL=${NCCL_IB_SL:-1}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-19}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

if command -v srun >/dev/null 2>&1; then
    BATCH=1
else
    BATCH=0
fi

ENABLE_WANDB=${ENABLE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-"megatron-vlm-v3"}
WANDB_ENTITY=${WANDB_ENTITY:-"adlr"}
WANDB_NAME=${WANDB_NAME:-"${MODEL_NAME}"}

# torch_dist reshards the TP8/EP8 conversion into the requested TP8/CP16/EP64 layout.
TP=${TP:-8}
EP=${EP:-64}
CP=${CP:-16}
NUM_GPU=${NUM_GPU:-4}

# Multimodal SFT runtime settings inherited from the Nano v3.5 stage.
MBZ=${MBZ:-1}
NW=${NW:-4}
AD=${AD:-0.0}
HD=${HD:-0.0}
LI=${LI:-10}
TIMING_LOG_LEVEL=${TIMING_LOG_LEVEL:-1}
ENABLE_TENSORBOARD_TIMERS=${ENABLE_TENSORBOARD_TIMERS:-1}
LOG_PACKED_SEQUENCE_STATS=${LOG_PACKED_SEQUENCE_STATS:-1}
EARLY_EXIT_ITERS=${EARLY_EXIT_ITERS:-0}
SKIP_SAVE=${SKIP_SAVE:-0}
NO_SAVE_OPTIM=${NO_SAVE_OPTIM:-0}
NO_SAVE_RNG=${NO_SAVE_RNG:-0}
STRICT_DATALOADER_STATE_LOAD=${STRICT_DATALOADER_STATE_LOAD:-1}
USE_MTP=${USE_MTP:-1}
USE_DYNAMIC_RES=${USE_DYNAMIC_RES:-1}
USE_IMAGE_BREAK=${USE_IMAGE_BREAK:-0}
USE_CONV_MERGE=${USE_CONV_MERGE:-0}
USE_FP8=${USE_FP8:-0}
USE_CPE_EVAL_MODE=${USE_CPE_EVAL_MODE:-1}
USE_PACKING=${USE_PACKING:-1}
USE_BUCKETING=${USE_BUCKETING:-0}
USE_CHECKPOINT_ARGS=${USE_CHECKPOINT_ARGS:-1}
USE_SEQUENCE_PARALLEL=${USE_SEQUENCE_PARALLEL:-1}
USE_MOE_GROUPED_GEMM=${USE_MOE_GROUPED_GEMM:-1}
VISION_RECOMPUTE_NUM_LAYERS=${VISION_RECOMPUTE_NUM_LAYERS:-30}

MAIN_HYBRID_PATTERN=${MAIN_HYBRID_PATTERN:-"MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"}
MTP_HYBRID_PATTERN=${MTP_HYBRID_PATTERN:-"*E"}
if [[ "${USE_MTP}" -eq 1 ]]; then
    HYBRID_LAYER_PATTERN=${HYBRID_LAYER_PATTERN:-"${MAIN_HYBRID_PATTERN}/${MTP_HYBRID_PATTERN}/${MTP_HYBRID_PATTERN}"}
else
    HYBRID_LAYER_PATTERN=${HYBRID_LAYER_PATTERN:-"${MAIN_HYBRID_PATTERN}"}
fi
MTP_LOSS_SCALING_FACTOR=${MTP_LOSS_SCALING_FACTOR:-0.3}

SEQ_LEN=${SEQ_LEN:-256}
DECODER_SEQ_LEN=${DECODER_SEQ_LEN:-524288}
PACKING_SEQ_LEN=${PACKING_SEQ_LEN:-${DECODER_SEQ_LEN}}
PBS=${PBS:-10000}
BZ=${BZ:-32}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-2e-6}
# Megatron advances the sample-based scheduler by the original-record count
# reported through samples_seen, not by the number of packed sequences. The
# running 491520-token job averaged 85.0 records per pack through iteration 220.
# Scaling that observation to 524288 tokens gives about 90.7; round to 91 so
# 100 full-length packed samples correspond to 9100 scheduler samples.
LR_WARMUP_PACKED_SAMPLES=${LR_WARMUP_PACKED_SAMPLES:-100}
ESTIMATED_RECORDS_PER_PACK=${ESTIMATED_RECORDS_PER_PACK:-91}
LR_WARMUP_SAMPLES=${LR_WARMUP_SAMPLES:-$((LR_WARMUP_PACKED_SAMPLES * ESTIMATED_RECORDS_PER_PACK))}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
SAVE_RETAIN_INTERVAL=${SAVE_RETAIN_INTERVAL:-${SAVE_INTERVAL}}
CKPT_FORMAT=${CKPT_FORMAT:-torch_dist}
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-1e-4}
# Match the text recipe's alpha=1.0 objective: normalize globally over supervised tokens.
USE_LOSS_SCALING=${USE_LOSS_SCALING:-0}

if [[ "${BATCH}" -eq 0 ]]; then
    SPECIAL_TOKENS=" --special-tokens <image> <img> </img> <quad> </quad> <ref> </ref> <box> </box>"
else
    SPECIAL_TOKENS=" --special-tokens \<image\> \<img\> \</img\> \<quad\> \</quad\> \<ref\> \</ref\> \<box\> \</box\>"
fi

EXTRA_ARGS=""

if [[ "${ENABLE_WANDB}" -eq 1 ]]; then
    EXTRA_ARGS+=" --wandb-project ${WANDB_PROJECT} --wandb-entity ${WANDB_ENTITY} --wandb-exp-name ${WANDB_NAME} --wandb-save-dir ${WANDB_DIR}"
fi

if [[ "${USE_FP8}" -eq 1 ]]; then
    EXTRA_ARGS+=" --fp8-recipe blockwise --fp8-format e4m3 --first-last-layers-bf16 --num-layers-at-start-in-bf16 1 --num-layers-at-end-in-bf16 1"
    EXTRA_ARGS+=" --use-vision-backbone-fp8-arch"
fi

if [[ "${USE_DYNAMIC_RES}" -eq 1 ]]; then
    SEQ_LEN=${DYNAMIC_SEQ_LEN:-12288}
    if [[ "${USE_IMAGE_BREAK}" -eq 1 ]]; then
        if [[ "${BATCH}" -eq 0 ]]; then
            EXTRA_ARGS+=" --image-break-token <image_break>"
            SPECIAL_TOKENS+=" <image_break>"
        else
            EXTRA_ARGS+=" --image-break-token \<image_break\>"
            SPECIAL_TOKENS+=" \<image_break\>"
        fi
    fi
    if [[ "${USE_CONV_MERGE}" -eq 1 ]]; then
        EXTRA_ARGS+=" --conv-merging --allow-missing-conv-merge-checkpoint"
    else
        EXTRA_ARGS+=" --pixel-shuffle"
    fi
    EXTRA_ARGS+=" --dynamic-resolution --dynamic-resolution-min-patches 1024 --dynamic-resolution-max-patches 13312"
fi

if [[ "${USE_CPE_EVAL_MODE}" -eq 1 ]]; then
    EXTRA_ARGS+=" --radio-force-cpe-eval-mode"
fi

if [[ "${USE_MTP}" -eq 1 ]]; then
    EXTRA_ARGS+=" --mtp-num-layers 2"
    EXTRA_ARGS+=" --mtp-loss-scaling-factor ${MTP_LOSS_SCALING_FACTOR}"
    EXTRA_ARGS+=" --mtp-use-repeated-layer"
    EXTRA_ARGS+=" --keep-mtp-spec-in-bf16"
else
    EXTRA_ARGS+=" --disable-mtp"
fi

if [[ "${EARLY_EXIT_ITERS}" -gt 0 ]]; then
    EXTRA_ARGS+=" --early-exit-iters ${EARLY_EXIT_ITERS}"
fi

if [[ "${USE_PACKING}" -eq 1 ]]; then
    EXTRA_ARGS+=" --packing-buffer-size ${PBS} --packing-seq-length ${PACKING_SEQ_LEN}"
    if [[ "${USE_BUCKETING}" -eq 1 ]]; then
        EXTRA_ARGS+=" --packing-knapsack-algorithm bucketing_greedy_knapsack"
    else
        EXTRA_ARGS+=" --packing-knapsack-algorithm balanced_greedy_knapsack"
    fi
    if [[ "${LOG_PACKED_SEQUENCE_STATS}" -eq 1 ]]; then
        EXTRA_ARGS+=" --log-packed-sequence-stats"
    fi
fi

if [[ "${ENABLE_TENSORBOARD_TIMERS}" -eq 1 ]]; then
    EXTRA_ARGS+=" --log-timers-to-tensorboard"
fi

VIDEO_MAX_NUM_FRAMES=${VIDEO_MAX_NUM_FRAMES:-64}
VIDEO_TARGET_NUM_PATCHES=${VIDEO_TARGET_NUM_PATCHES:-1024}
VIDEO_AUG_SCALE_FRAMES_UP=${VIDEO_AUG_SCALE_FRAMES_UP:-4}
VIDEO_AUG_SCALE_RESOLUTION_UP=${VIDEO_AUG_SCALE_RESOLUTION_UP:-None}
VIDEO_AUG_SCALE_RESOLUTION_ONLY=${VIDEO_AUG_SCALE_RESOLUTION_ONLY:-1}
EXTRA_ARGS+=" --video-maintain-aspect-ratio --separate-video-embedder"
EXTRA_ARGS+=" --video-target-num-patches ${VIDEO_TARGET_NUM_PATCHES}"
EXTRA_ARGS+=" --video-max-num-frames ${VIDEO_MAX_NUM_FRAMES}"
EXTRA_ARGS+=" --video-temporal-patch-size 2 --video-prompt-version 2"
if [[ "${VIDEO_AUG_SCALE_FRAMES_UP}" != "None" ]]; then
    EXTRA_ARGS+=" --video-aug-scale-frames-up ${VIDEO_AUG_SCALE_FRAMES_UP}"
fi
if [[ "${VIDEO_AUG_SCALE_RESOLUTION_UP}" != "None" ]]; then
    EXTRA_ARGS+=" --video-aug-scale-resolution-up ${VIDEO_AUG_SCALE_RESOLUTION_UP}"
fi
if [[ "${VIDEO_AUG_SCALE_RESOLUTION_ONLY}" -eq 1 ]]; then
    EXTRA_ARGS+=" --video-aug-scale-resolution-only"
fi

if [[ "${USE_LOSS_SCALING}" -eq 1 ]]; then
    EXTRA_ARGS+=" --use-loss-scaling"
fi

# Match the Super 3.5 text SFT recipe by selectively recomputing MoE modules.
EXTRA_ARGS+=" --recompute-granularity selective --recompute-modules moe"
EXTRA_ARGS+=" --recompute-vision --recompute-method-vision block --recompute-granularity-vision full --recompute-vision-num-layers ${VISION_RECOMPUTE_NUM_LAYERS}"
EXTRA_ARGS+=" --log-model-grad-norms --log-model-act-norms --allow-checkpoint-without-temporal-compression ${CUSTOM_ARGS:-}"

CHECKPOINT_CONFIG_ARGS=""
if [[ "${USE_CHECKPOINT_ARGS}" -eq 1 ]]; then
    CHECKPOINT_CONFIG_ARGS="--use-checkpoint-args"
fi

PARALLEL_ARGS=""
if [[ "${USE_SEQUENCE_PARALLEL}" -eq 1 ]]; then
    PARALLEL_ARGS="--sequence-parallel"
fi

MOE_GROUPED_GEMM_ARGS=""
if [[ "${USE_MOE_GROUPED_GEMM}" -eq 1 ]]; then
    MOE_GROUPED_GEMM_ARGS="--moe-grouped-gemm"
fi

CHECKPOINT_ARGS=" \
    --pretrained-checkpoint ${CHECKPOINT_DIR} \
    --load ${FINETUNE_DIR} \
    --ckpt-format ${CKPT_FORMAT} \
"

if [[ "${STRICT_DATALOADER_STATE_LOAD}" -eq 1 ]]; then
    CHECKPOINT_ARGS+=" --strict-dataloader-state-load"
fi

if [[ "${SKIP_SAVE}" -eq 0 ]]; then
    CHECKPOINT_ARGS+=" \
        --save ${FINETUNE_DIR} \
        --dataloader-save ${FINETUNE_DIR}/dataloader \
        --save-interval ${SAVE_INTERVAL} \
        --save-retain-interval ${SAVE_RETAIN_INTERVAL} \
    "
    if [[ "${NO_SAVE_OPTIM}" -eq 1 ]]; then
        CHECKPOINT_ARGS+=" --no-save-optim"
    fi
    if [[ "${NO_SAVE_RNG}" -eq 1 ]]; then
        CHECKPOINT_ARGS+=" --no-save-rng"
    fi
fi

OPTIONS=" \
    ${CHECKPOINT_CONFIG_ARGS} \
    --sft \
    --no-use-tokenizer-model-from-checkpoint-args \
    --transformer-impl transformer_engine \
    --use-te \
    --data-path ${DATA_TRAIN} \
    --train-full-dataset \
    --patch-dim 16 \
    --img-h 512 \
    --img-w 512 \
    --dataloader-type external \
    --language-model-type nemotron6-super \
    ${EXTRA_ARGS} \
    --vision-model-type radio \
    --class-token-len 10 \
    ${SPECIAL_TOKENS} \
    --disable-vision-class-token \
    --prompt-path ${CODE_DIR}/examples/multimodal/manual_prompts.json \
    --eod-mask-loss \
    --image-tag-type internvl \
    --moe-token-dispatcher-type alltoall \
    --enable-experimental \
    --moe-permute-fusion \
    --use-fused-weighted-squared-relu \
    --moe-router-score-function sigmoid \
    ${MOE_GROUPED_GEMM_ARGS} \
    --num-experts 512 \
    --moe-router-topk 22 \
    --moe-aux-loss-coeff ${MOE_AUX_LOSS_COEFF} \
    --moe-router-topk-scaling-factor 5.0 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-shared-expert-intermediate-size 5376 \
    --moe-latent-size 1024 \
    --attention-backend flash \
    --is-hybrid-model \
    --mamba-num-heads 128 \
    --mamba-head-dim 64 \
    --hybrid-layer-pattern '${HYBRID_LAYER_PATTERN}' \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --tiktoken-pattern v2 \
    --distributed-timeout-minutes 120 \
    --use-mcore-models \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --init-method-std 0.014 \
    --position-embedding-type none \
    --squared-relu \
    --num-layers 88 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 2 \
    --ffn-hidden-size 2688 \
    --kv-channels 128 \
    --normalization RMSNorm \
    --attention-dropout ${AD} \
    --hidden-dropout ${HD} \
    --exit-duration-in-mins 220 \
    --tensor-model-parallel-size ${TP} \
    --context-parallel-size ${CP} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --seq-length ${SEQ_LEN} \
    --decoder-seq-length ${DECODER_SEQ_LEN} \
    --max-position-embeddings ${DECODER_SEQ_LEN} \
    --micro-batch-size ${MBZ} \
    --global-batch-size ${BZ} \
    --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style constant \
    --override-opt-param-scheduler \
    --weight-decay ${WEIGHT_DECAY} \
    --clip-grad 1.0 \
    --log-interval ${LI} \
    --eval-iters 0 \
    --eval-interval 99999999999 \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --tokenizer-prompt-format ${TOKENIZER_PROMPT_FORMAT} \
    --thinking-trace-format ultra \
    --tokenizer-keep-history-thinking \
    ${CHECKPOINT_ARGS} \
    --log-progress \
    --timing-log-level ${TIMING_LOG_LEVEL} \
    --timing-log-option minmax \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --logging-level 20 \
    --log-memory-interval 100 \
    --high-priority-stream-groups ep \
    --manual-gc-interval 10 \
    --manual-gc \
    --bf16 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --use-distributed-optimizer \
    --num-workers ${NW} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    ${PARALLEL_ARGS} \
    --allow-large-videos \
"

export WANDB_PROJECT WANDB_ENTITY WANDB_NAME WANDB_DIR
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

run_cmd="python -u ${TRAIN_ENTRYPOINT} ${OPTIONS}"

echo "CODE_DIR=${CODE_DIR}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "DATA_TRAIN=${DATA_TRAIN}"
echo "TOKENIZER_MODEL=${TOKENIZER_MODEL}"
echo "FINETUNE_DIR=${FINETUNE_DIR}"
echo "TP=${TP} CP=${CP} EP=${EP}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "${run_cmd}"
    exit 0
fi

if [[ "${BATCH}" -eq 0 ]]; then
    cd "${CODE_DIR}"
    torchrun --nproc_per_node "${NUM_GPU}" "${TRAIN_ENTRYPOINT}" ${OPTIONS}
else
    if git -C "${CODE_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git -C "${CODE_DIR}" log --oneline -1
    else
        echo "Git metadata unavailable in code snapshot."
    fi
    DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
    srun -l --verbose \
        --mpi=none \
        --container-image "${CONTAINER_IMAGE}" \
        --container-mounts "${CONTAINER_MOUNTS}" \
        --output="${LOGS_DIR}/%x_%j_srun_${DATETIME}.log" \
        sh -c "cd '${CODE_DIR}'; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH; export TRITON_CACHE_DIR=/tmp/triton_cache_\${SLURM_NODEID}; echo ${run_cmd}; ${run_cmd}"
fi
