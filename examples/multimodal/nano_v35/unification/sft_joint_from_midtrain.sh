#!/bin/bash

#SBATCH -A nemotron_omni_vision
#SBATCH -p batch
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --dependency=singleton
#SBATCH --nodes=64
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --gpus-per-node=4
#SBATCH --job-name=nano35_midtrain_100B_lc_joint_sft_20260728_cp32-dss-fix-text-vision-1to1

set -euo pipefail

# Standalone multimodal SFT from the Nano v3.5 midtrain vision-pretrain checkpoint.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_CODE_DIR=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)
if [[ -n "${NANO_V35_CODE_DIR:-}" ]]; then
    CODE_DIR=$(cd -- "${NANO_V35_CODE_DIR}" && pwd)
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/multimodal/train.py" ]]; then
    CODE_DIR=$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)
else
    CODE_DIR="${SCRIPT_CODE_DIR}"
fi
if [[ ! -f "${CODE_DIR}/examples/multimodal/train.py" ]]; then
    echo "ERROR: Could not resolve the Megatron-LM checkout; ${CODE_DIR}/examples/multimodal/train.py is missing." >&2
    exit 1
fi

NANO_V35_PROJECT_ROOT=${NANO_V35_PROJECT_ROOT:-"$(dirname -- "${CODE_DIR}")"}
NANO_V35_RESOURCES=${NANO_V35_RESOURCES:-"${NANO_V35_PROJECT_ROOT}/resources"}

WORKSPACE=${WORKSPACE:-"/lustre/fsw/portfolios/llmservice/users/guyueh/super-3p5-vl/nano-3p5-joint-sft/joint-sft-from-midtrain"}
OUTPUT_BASE=${OUTPUT_BASE:-"${WORKSPACE}/workspace/output"}
CONTAINER_IMAGE="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/containers/pytorch25.11-moe-avlm-editable-energon-super-triton35.sqsh"
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-"/lustre"}

VISION_PRETRAIN_MODEL_NAME=${VISION_PRETRAIN_MODEL_NAME:-"nano35_midtrain_100B_lc_vision_pretrain_0723"}
MODEL_NAME=${MODEL_NAME:-"nano35_midtrain_100B_lc_joint_sft_20260728_cp32-dss-fix-text-vision-1to1"}
TOKENIZER_MODEL="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/repos_nanov35/resources/tokenizer/nano_v35_sft_v10_closethink_unmask_orig6k_vlm"
VISION_PRETRAIN_OUTPUT_BASE=${VISION_PRETRAIN_OUTPUT_BASE:-"/lustre/fsw/portfolios/llmservice/users/guyueh/super-3p5-vl/nano-3p5-joint-sft/vlm_pretrain/workspace/output"}
VISION_PRETRAIN_CKPT_DIR=${VISION_PRETRAIN_CKPT_DIR:-"${VISION_PRETRAIN_OUTPUT_BASE}/${VISION_PRETRAIN_MODEL_NAME}/checkpoints"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${VISION_PRETRAIN_CKPT_DIR}"}
DATA_TRAIN=${DATA_TRAIN:-"${CODE_DIR}/examples/multimodal/nano_v35/unification/sft_joint_from_midtrain.yaml"}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export UB_TIMEOUT=${UB_TIMEOUT:-720}
export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCHINDUCTOR_WORKER_START=${TORCHINDUCTOR_WORKER_START:-fork}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# This recipe is intended to run through Slurm only. Keep argument escaping
# compatible with the srun sh -c launch below.

ENABLE_WANDB=${ENABLE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-"megatron-vlm-v3"}
WANDB_ENTITY=${WANDB_ENTITY:-"adlr"}
WANDB_NAME=${WANDB_NAME:-"${MODEL_NAME}"}
WANDB_RUN_ID=${WANDB_RUN_ID:-"${MODEL_NAME}"}
WANDB_RESUME=${WANDB_RESUME:-"allow"}

OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"
FINETUNE_DIR="${OUTPUT}/checkpoints"
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"
WANDB_DIR="${OUTPUT}/wandb"
TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT:-"${CODE_DIR}/examples/multimodal/train.py"}
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/examples/multimodal:${PYTHONPATH:-}"

mkdir -p "${FINETUNE_DIR}" "${LOGS_DIR}" "${TENSORBOARD_DIR}" "${WANDB_DIR}"

TP=${TP:-2}
CP=${CP:-16}
EP=${EP:-32}
NUM_GPU=${NUM_GPU:-4}
MBZ=${MBZ:-1}
NW=${NW:-8}
AD=${AD:-0.0}
HD=${HD:-0.0}
LI=${LI:-1}
TIMING_LOG_LEVEL=${TIMING_LOG_LEVEL:-1}
ENABLE_TENSORBOARD_TIMERS=${ENABLE_TENSORBOARD_TIMERS:-1}
LOG_PACKED_SEQUENCE_STATS=${LOG_PACKED_SEQUENCE_STATS:-1}
MM_TOKENIZATION_PROFILE=${MM_TOKENIZATION_PROFILE:-0}
MM_TOKENIZATION_PROFILE_INTERVAL=${MM_TOKENIZATION_PROFILE_INTERVAL:-100}
MM_TOKENIZATION_PROFILE_DIR=${MM_TOKENIZATION_PROFILE_DIR:-"${OUTPUT}/tokenization_profile"}
FILESTORE_CACHE_WORKERS=${FILESTORE_CACHE_WORKERS:-2}
TORCH_MULTIPROCESSING_SHARING_STRATEGY=${TORCH_MULTIPROCESSING_SHARING_STRATEGY:-file_system}
DRY_RUN=${DRY_RUN:-0}
EARLY_EXIT_ITERS=${EARLY_EXIT_ITERS:-0}
EXIT_DURATION_MINS=${EXIT_DURATION_MINS:-230}
SKIP_SAVE=${SKIP_SAVE:-0}
NO_SAVE_OPTIM=${NO_SAVE_OPTIM:-0}
NO_SAVE_RNG=${NO_SAVE_RNG:-0}
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
LANGUAGE_RECOMPUTE_MODULES=${LANGUAGE_RECOMPUTE_MODULES:-"core_attn moe_act"}
VISION_RECOMPUTE_NUM_LAYERS=${VISION_RECOMPUTE_NUM_LAYERS:-32}

TOKENIZER_PROMPT_FORMAT=${TOKENIZER_PROMPT_FORMAT:-"nemotron6-moe"}
MAIN_HYBRID_PATTERN=${MAIN_HYBRID_PATTERN:-"MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"}
HYBRID_LAYER_PATTERN=${HYBRID_LAYER_PATTERN:-"${MAIN_HYBRID_PATTERN}"}
MTP_HYBRID_PATTERN=${MTP_HYBRID_PATTERN:-"*E"}
MTP_LOSS_SCALING_FACTOR=${MTP_LOSS_SCALING_FACTOR:-1.5e-4}

SEQ_LEN=${SEQ_LEN:-256}
DECODER_SEQ_LEN=${DECODER_SEQ_LEN:-524288}
PACKING_SEQ_LEN=${PACKING_SEQ_LEN:-${DECODER_SEQ_LEN}}
PBS=${PBS:-10000}
BZ=${BZ:-32}
LR=${LR:-2e-5}
MIN_LR=${MIN_LR:-5e-6}
LR_WARMUP_FRACTION=${LR_WARMUP_FRACTION:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-1e-8}
USE_LOSS_SCALING=${USE_LOSS_SCALING:-1}

if [[ "${MM_TOKENIZATION_PROFILE}" == "1" ]]; then
    mkdir -p "${MM_TOKENIZATION_PROFILE_DIR}"
    echo "MM_TOKENIZATION_PROFILE=${MM_TOKENIZATION_PROFILE}"
    echo "MM_TOKENIZATION_PROFILE_INTERVAL=${MM_TOKENIZATION_PROFILE_INTERVAL}"
    echo "MM_TOKENIZATION_PROFILE_DIR=${MM_TOKENIZATION_PROFILE_DIR}"
fi

SPECIAL_TOKENS=" --special-tokens \<image\> \<img\> \</img\> \<quad\> \</quad\> \<ref\> \</ref\> \<box\> \</box\>"

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
        EXTRA_ARGS+=" --image-break-token \<image_break\>"
        SPECIAL_TOKENS+=" \<image_break\>"
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
    EXTRA_ARGS+=" --mtp-hybrid-override-pattern '${MTP_HYBRID_PATTERN}'"
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
VIDEO_TEMPORAL_PATCH_SIZE=${VIDEO_TEMPORAL_PATCH_SIZE:-2}
VIDEO_AUG_SCALE_FRAMES_UP=${VIDEO_AUG_SCALE_FRAMES_UP:-4}
VIDEO_AUG_SCALE_RESOLUTION_UP=${VIDEO_AUG_SCALE_RESOLUTION_UP:-None}
VIDEO_AUG_SCALE_RESOLUTION_ONLY=${VIDEO_AUG_SCALE_RESOLUTION_ONLY:-1}
EXTRA_ARGS+=" --video-maintain-aspect-ratio --separate-video-embedder"
EXTRA_ARGS+=" --video-target-num-patches ${VIDEO_TARGET_NUM_PATCHES}"
EXTRA_ARGS+=" --video-max-num-frames ${VIDEO_MAX_NUM_FRAMES}"
EXTRA_ARGS+=" --video-temporal-patch-size ${VIDEO_TEMPORAL_PATCH_SIZE} --video-prompt-version 2"
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

EXTRA_ARGS+=" --recompute-granularity selective --recompute-modules ${LANGUAGE_RECOMPUTE_MODULES}"
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

OPTIONS=" \
    ${CHECKPOINT_CONFIG_ARGS} \
    --sft \
    --transformer-impl transformer_engine \
    --use-te \
    --data-path ${DATA_TRAIN} \
    --train-full-dataset \
    --patch-dim 16 \
    --img-h 512 \
    --img-w 512 \
    --dataloader-type external \
    --use-new-dataloader-path \
    --strict-dataloader-state-load \
    --reset-position-ids-from-packed-metadata \
    --language-model-type nemotron6-moe \
    ${EXTRA_ARGS} \
    --vision-model-type radio \
    --class-token-len 10 \
    ${SPECIAL_TOKENS} \
    --disable-vision-class-token \
    --prompt-path ${CODE_DIR}/examples/multimodal/manual_prompts.json \
    --eod-mask-loss \
    --image-tag-type internvl \
    --moe-token-dispatcher-type alltoall \
    --moe-shared-expert-overlap \
    --enable-experimental \
    --moe-permute-fusion \
    --use-fused-weighted-squared-relu \
    --moe-router-score-function sigmoid \
    ${MOE_GROUPED_GEMM_ARGS} \
    --num-experts 128 \
    --moe-router-topk 6 \
    --moe-aux-loss-coeff ${MOE_AUX_LOSS_COEFF} \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-shared-expert-intermediate-size 3712 \
    --attention-backend flash \
    --is-hybrid-model \
    --mamba-num-heads 64 \
    --mamba-head-dim 64 \
    --hybrid-layer-pattern '${HYBRID_LAYER_PATTERN}' \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --tiktoken-pattern v2 \
    --distributed-timeout-minutes 30 \
    --use-mcore-models \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --init-method-std 0.0173 \
    --position-embedding-type none \
    --squared-relu \
    --num-layers 52 \
    --hidden-size 2688 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 2 \
    --ffn-hidden-size 1856 \
    --kv-channels 128 \
    --normalization RMSNorm \
    --attention-dropout ${AD} \
    --hidden-dropout ${HD} \
    --exit-duration-in-mins ${EXIT_DURATION_MINS} \
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
    --lr-warmup-fraction ${LR_WARMUP_FRACTION} \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style cosine \
    --override-opt-param-scheduler \
    --weight-decay ${WEIGHT_DECAY} \
    --clip-grad 1.0 \
    --log-interval ${LI} \
    --eval-iters 0 \
    --eval-interval 99999999999 \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --tokenizer-prompt-format ${TOKENIZER_PROMPT_FORMAT} \
    --tokenizer-keep-history-thinking \
    --thinking-trace-format ultra \
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
    --adam-beta2 0.999 \
    --use-distributed-optimizer \
    --num-workers ${NW} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    ${PARALLEL_ARGS} \
    --allow-large-videos \
"

export WANDB_PROJECT WANDB_ENTITY WANDB_NAME WANDB_DIR WANDB_RUN_ID WANDB_RESUME
export MULTIMODAL_FILESTORE_CACHE_WORKERS="${FILESTORE_CACHE_WORKERS}"
export MM_TOKENIZATION_PROFILE MM_TOKENIZATION_PROFILE_INTERVAL MM_TOKENIZATION_PROFILE_DIR
export TORCH_MULTIPROCESSING_SHARING_STRATEGY
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
if [[ -d /cm/local/apps/slurm/25.11/bin ]]; then
    export PATH="/cm/local/apps/slurm/25.11/bin:${PATH}"
fi

run_cmd="python -u ${TRAIN_ENTRYPOINT} ${OPTIONS}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "${run_cmd}"
    exit 0
fi

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
srun -l --verbose \
    --container-image "${CONTAINER_IMAGE}" \
    --container-mounts "${CONTAINER_MOUNTS}" \
    --output="${LOGS_DIR}/%x_%j_srun_${DATETIME}.log" \
    sh -c "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH; export TRITON_CACHE_DIR=/tmp/triton_cache_\${SLURM_NODEID}; export PYTHONPYCACHEPREFIX=/tmp/python_pycache_\${SLURM_PROCID}; echo ${run_cmd}; ${run_cmd}"
