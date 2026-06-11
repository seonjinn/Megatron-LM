#!/bin/bash
#SBATCH -A nemotron_omni_vision
#SBATCH -p batch
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton
#SBATCH --nodes=64
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --gpus-per-node=8
#SBATCH --job-name=nano_v35_omni_49k_svg_0611

MODEL_NAME=${MODEL_NAME:-nano_v35_omni_49k_svg_0611}

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MSC_CONFIG=${MSC_CONFIG:-"/scratch/fsw/portfolios/llmservice/users/trintamaki/msc_config/msc_config.yaml"}
export UB_TIMEOUT=${UB_TIMEOUT:-720}
export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCHINDUCTOR_WORKER_START=${TORCHINDUCTOR_WORKER_START:-fork}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

USER_NAME=${SLURM_JOB_USER:-${USER:-$(whoami)}}
if command -v srun >/dev/null 2>&1; then
    BATCH=1
else
    BATCH=0
fi

WANDB_API_KEY=${WANDB_API_KEY:-}
WANDB_PROJECT=${WANDB_PROJECT:-"megatron-vlm-v3"}
WANDB_ENTITY=${WANDB_ENTITY:-"adlr"}
WANDB_NAME=${MODEL_NAME}

WORKSPACE="/lustre/fsw/portfolios/llmservice/users/${USER_NAME}/workspace"
SOURCE=$(pwd)
SOURCE=$(echo "${SOURCE}" | sed 's|^//|/|')
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR="${OUTPUT}/checkpoints"
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"
WANDB_DIR="${OUTPUT}/wandb"
CODE_DIR="${SOURCE}"

mkdir -p "${FINETUNE_DIR}" "${LOGS_DIR}" "${TENSORBOARD_DIR}" "${WANDB_DIR}"

CONTAINER_IMAGE=${CONTAINER_IMAGE:-"/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/containers/pytorch25.11-moe-avlm-editable-energon-super.sqsh"}

TP=${TP:-2}
EP=${EP:-32}
NUM_GPU=${NUM_GPU:-8}
MBZ=${MBZ:-1}
NW=${NW:-8}
AD=${AD:-0.0}
HD=${HD:-0.0}
LI=${LI:-5}
DEBUG=${DEBUG:-0}
DRY_RUN=${DRY_RUN:-0}
EARLY_EXIT_ITERS=${EARLY_EXIT_ITERS:-0}
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

TOKENIZER_MODEL=${TOKENIZER_MODEL:-"/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/workspace/nemotron_3_nano_30b_a3b_tokenizer"}
TOKENIZER_PROMPT_FORMAT=${TOKENIZER_PROMPT_FORMAT:-"nemotron6-moe"}
BOOTSTRAP_CKPT=${BOOTSTRAP_CKPT:-"${WORKSPACE}/checkpoints/nano_v35_vlm/nano_v35_moe_tp2_ep32_radio_v4_mtpfix"}

MAIN_HYBRID_PATTERN=${MAIN_HYBRID_PATTERN:-"MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"}
HYBRID_LAYER_PATTERN=${HYBRID_LAYER_PATTERN:-"${MAIN_HYBRID_PATTERN}"}
MTP_HYBRID_PATTERN=${MTP_HYBRID_PATTERN:-"*E"}
MTP_LOSS_SCALING_FACTOR=${MTP_LOSS_SCALING_FACTOR:-1.5e-4}

SEQ_LEN=${SEQ_LEN:-256}
DECODER_SEQ_LEN=${DECODER_SEQ_LEN:-}
PACKING_SEQ_LEN=${PACKING_SEQ_LEN:-}
PBS=${PBS:-}
BZ=${BZ:-}
LR=${LR:-}
MIN_LR=${MIN_LR:-}
LR_WARMUP_FRACTION=${LR_WARMUP_FRACTION:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-}
SAVE_INTERVAL=${SAVE_INTERVAL:-}
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-1e-8}
USE_LOSS_SCALING=${USE_LOSS_SCALING:-1}

INCLUDE_AUDIO=1
INCLUDE_VIDEO=1
ALLOW_MISSING_VISION_PROJECTION=0
ALLOW_MISSING_SOUND=0
FREEZE_ARGS=""
CP_SIZE=${CP_SIZE:-2}
STAGE_EXTRA_ARGS=" --context-parallel-size ${CP_SIZE} --tokenizer-keep-history-thinking "
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${WORKSPACE}/output/nano_v35_omni_16k_svg_0611/checkpoints"}
DATA_TRAIN=${DATA_TRAIN:-"${SOURCE}/examples/multimodal/super/data_config/yamls/sft.long_context.49k.ehsan.v13p77.2x.0330.yaml"}
DECODER_SEQ_LEN=${DECODER_SEQ_LEN:-49152}
PACKING_SEQ_LEN=${PACKING_SEQ_LEN:-${DECODER_SEQ_LEN}}
PBS=${PBS:-500}
BZ=${BZ:-256}
LR=${LR:-1e-6}
MIN_LR=${MIN_LR:-0.0}

DECODER_SEQ_LEN=${DECODER_SEQ_LEN:-16384}
PACKING_SEQ_LEN=${PACKING_SEQ_LEN:-${DECODER_SEQ_LEN}}
PBS=${PBS:-1000}
BZ=${BZ:-512}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-0.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}

if [[ "${BATCH}" -eq 0 ]]; then
    SPECIAL_TOKENS=" --special-tokens <image> <img> </img> <quad> </quad> <ref> </ref> <box> </box>"
    if [[ "${INCLUDE_AUDIO}" -eq 1 ]]; then
        SPECIAL_TOKENS+=" <so_embedding> <so_start> <so_end>"
    fi
else
    SPECIAL_TOKENS=" --special-tokens \<image\> \<img\> \</img\> \<quad\> \</quad\> \<ref\> \</ref\> \<box\> \</box\>"
    if [[ "${INCLUDE_AUDIO}" -eq 1 ]]; then
        SPECIAL_TOKENS+=" \<so_embedding\> \<so_start\> \<so_end\>"
    fi
fi

EXTRA_ARGS=""

if [[ -n "${WANDB_API_KEY}" ]]; then
    EXTRA_ARGS+=" --wandb-project ${WANDB_PROJECT} --wandb-exp-name ${MODEL_NAME} --wandb-save-dir ${WANDB_DIR}"
fi

if [[ "${USE_FP8}" -eq 1 ]]; then
    EXTRA_ARGS+=" --fp8-recipe blockwise --fp8-format e4m3 --first-last-layers-bf16 --num-layers-at-start-in-bf16 1 --num-layers-at-end-in-bf16 1"
    EXTRA_ARGS+=" --use-vision-backbone-fp8-arch "
fi

if [[ "${USE_DYNAMIC_RES}" -eq 1 ]]; then
    SEQ_LEN=12288
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
fi

if [[ "${INCLUDE_VIDEO}" -eq 1 ]]; then
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
fi

SOUND_MODEL_CACHE="${WORKSPACE}/models/parakeet-tdt-0.6b-v2.nemo"
if [[ "${INCLUDE_AUDIO}" -eq 1 ]]; then
    SOUND_MODEL_TYPE=${SOUND_MODEL_TYPE:-"nemo://${SOUND_MODEL_CACHE}"}
    EXTRA_ARGS+=" --sound-model-type ${SOUND_MODEL_TYPE}"
    EXTRA_ARGS+=" --sound-target-rate 16000 --sound-embedding-size 751"
    if [[ "${ALLOW_MISSING_SOUND}" -eq 1 ]]; then
        EXTRA_ARGS+=" --allow-missing-sound-projection-checkpoint --allow-missing-sound-model-checkpoint"
    fi
fi

if [[ "${ALLOW_MISSING_VISION_PROJECTION}" -eq 1 ]]; then
    EXTRA_ARGS+=" --allow-missing-vision-projection-checkpoint"
fi

if [[ "${USE_LOSS_SCALING}" -eq 1 ]]; then
    EXTRA_ARGS+=" --use-loss-scaling"
fi

EXTRA_ARGS+=" --recompute-granularity selective --recompute-modules mlp moe"
EXTRA_ARGS+=" --recompute-vision --recompute-method-vision block --recompute-granularity-vision full --recompute-vision-num-layers 16"
EXTRA_ARGS+=" ${STAGE_EXTRA_ARGS} ${CUSTOM_ARGS:-}"

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
    --use-checkpoint-args \
    --transformer-impl transformer_engine \
    --use-te \
    --data-path ${DATA_TRAIN} \
    --train-full-dataset \
    --patch-dim 16 \
    --img-h 512 \
    --img-w 512 \
    --dataloader-type external \
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
    --moe-grouped-gemm \
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
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size ${TP} \
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
    --weight-decay ${WEIGHT_DECAY} \
    --clip-grad 1.0 \
    --log-interval ${LI} \
    --eval-iters 0 \
    --eval-interval 99999999999 \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --tokenizer-prompt-format ${TOKENIZER_PROMPT_FORMAT} \
    ${CHECKPOINT_ARGS} \
    --log-progress \
    --timing-log-option minmax \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --logging-level 20 \
    --log-memory-interval 100 \
    --bf16 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --use-distributed-optimizer \
    --num-workers ${NW} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --sequence-parallel \
    --allow-large-videos \
    ${FREEZE_ARGS} \
"

export WANDB_ENTITY
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

if [[ "${BATCH}" -eq 0 ]]; then
    cd "${CODE_DIR}"
    torchrun --nproc_per_node "${NUM_GPU}" examples/multimodal/train.py ${OPTIONS}
else
    run_cmd="python -u ${CODE_DIR}/examples/multimodal/train.py ${OPTIONS}"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "${run_cmd}"
        exit 0
    fi

    DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

    if [[ "${INCLUDE_AUDIO}" -eq 1 && ! -f "${SOUND_MODEL_CACHE}" ]]; then
        mkdir -p "$(dirname "${SOUND_MODEL_CACHE}")"
        srun -l --verbose \
            --ntasks=1 \
            --container-image "${CONTAINER_IMAGE}" \
            --container-mounts "/lustre,/scratch" \
            --output="${LOGS_DIR}/%x_%j_${DATETIME}_predownload.log" \
            sh -c "python -c \"from huggingface_hub import hf_hub_download; hf_hub_download('nvidia/parakeet-tdt-0.6b-v2', 'parakeet-tdt-0.6b-v2.nemo', local_dir='$(dirname "${SOUND_MODEL_CACHE}")')\""
    fi

    srun -l --verbose \
        --container-image "${CONTAINER_IMAGE}" \
        --container-mounts "/lustre,/scratch" \
        --output="${LOGS_DIR}/%x_%j_srun_${DATETIME}.log" \
        sh -c "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH; export TRITON_CACHE_DIR=/tmp/triton_cache_\${SLURM_NODEID}; echo ${run_cmd}; ${run_cmd}"
fi
