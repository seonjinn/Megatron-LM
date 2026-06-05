#!/bin/bash

#SBATCH -A llmservice_fm_vision
#SBATCH -p batch
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton
#SBATCH --nodes=64
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --gpus-per-node=8
#SBATCH --job-name=sft_super_omni_49k_svg_newcontainer_0422

set -eo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MSC_CONFIG="/scratch/fsw/portfolios/llmservice/users/trintamaki/msc_config/msc_config.yaml"

export UB_TIMEOUT=720
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN
export TORCHINDUCTOR_WORKER_START=fork

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fix cuDNN sub-library version mismatch: container has cuDNN 9.10.2 in /usr/lib/x86_64-linux-gnu/
# and 9.11.0 in /usr/local/cuda/lib64/. Without this, the 9.11.0 main lib loads 9.10.2 sub-libs
# (e.g. libcudnn_cnn), causing CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED on conv2d ops (sound model).
# We prepend /usr/local/cuda/lib64 to LD_LIBRARY_PATH inside the srun command below.

USER=${SLURM_JOB_USER:-${USER:-$(whoami)}}

# Auto-detect batch or interactive mode.
if command -v srun >/dev/null 2>&1; then
    BATCH=1
else
    BATCH=0
fi

DEBUG=0
USE_TILING=0
USE_DYNAMIC_RES=1
USE_IMAGE_BREAK=0   # Only used if USE_DYNAMIC_RES is 1.
USE_CONV_MERGE=0    # Only used if USE_DYNAMIC_RES is 1.
USE_FP8=0
USE_CPE_EVAL_MODE=1
USE_MTP=${USE_MTP:-1}
MTP_LOSS_SCALING_FACTOR=${MTP_LOSS_SCALING_FACTOR:-1.5e-4}
FREEZE_SOUND=${FREEZE_SOUND:-0}
SKIP_SAVE=${SKIP_SAVE:-0}
USE_LOSS_SCALING=${USE_LOSS_SCALING:-1}
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-1e-8}
USE_PACKING=1
USE_BUCKETING=0
USE_CP=1
CP_SIZE=2

AUDIO_MAX_DURATION_SECONDS=1200

# Conv3d video defaults (w/ allowed overrides via env vars)
VIDEO_MAX_NUM_FRAMES=${VIDEO_MAX_NUM_FRAMES:-256}  # Using conv3d w/ 2-frame compression
SEPARATE_VIDEO_EMBEDDER=${SEPARATE_VIDEO_EMBEDDER:-1}
VIDEO_MAINTAIN_ASPECT_RATIO=${VIDEO_MAINTAIN_ASPECT_RATIO:-1}

# Video augmentation
VIDEO_TARGET_NUM_PATCHES=${VIDEO_TARGET_NUM_PATCHES:-1024}
VIDEO_AUG_SCALE_FRAMES_UP=${VIDEO_AUG_SCALE_FRAMES_UP:-4}
VIDEO_AUG_SCALE_RESOLUTION_UP=${VIDEO_AUG_SCALE_RESOLUTION_UP:-None}
VIDEO_AUG_SCALE_RESOLUTION_ONLY=${VIDEO_AUG_SCALE_RESOLUTION_ONLY:-1}

# Used by examples/multimodal/launch.sh (or can be overridden via exports)
DRY_RUN=${DRY_RUN:-0}
EARLY_EXIT_ITERS=${EARLY_EXIT_ITERS:-0}  # Exit early for testing

if [[ $DEBUG -eq 1 ]]; then
    MBZ=1
    BZ=8
    NW=0
    AD=0.0
    HD=0.0
    LI=1

    EXTRA_ARGS=""

    NUM_GPU=8
    PBS=100
else
    MBZ=1
    BZ=256
    NW=8
    AD=0.0
    HD=0.0
    LI=5
    EXTRA_ARGS=""
    NUM_GPU=8
    PBS=500
fi

SEQ_LEN=256
DECODER_SEQ_LEN=49152

# Remember to update model and job name if running in batch mode!!
if [[ $BATCH -eq 0 ]]; then
    DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
    MODEL_NAME=${MODEL_NAME:-"interactive_sft_super_omni_49k_svg_newcontainer_${DATETIME}"}
    SPECIAL_TOKENS=" --special-tokens <image> <img> </img> <quad> </quad> <ref> </ref> <box> </box> <so_embedding> <so_start> <so_end> "
    DEBUG=1
else
    MODEL_NAME=${MODEL_NAME:-"sft_super_omni_49k_svg_newcontainer_0422"}
    SPECIAL_TOKENS=" --special-tokens \<image\> \<img\> \</img\> \<quad\> \</quad\> \<ref\> \</ref\> \<box\> \</box\> \<so_embedding\> \<so_start\> \<so_end\> "
fi

WANDB_API_KEY=${WANDB_API_KEY}
WANDB_PROJECT=${WANDB_PROJECT:-"megatron-vlm-v3"}
WANDB_ENTITY=${WANDB_ENTITY:-"adlr"}
WANDB_NAME=${MODEL_NAME}

WORKSPACE="/lustre/fsw/portfolios/llmservice/users/${USER}/workspace"
SOURCE=`pwd`
# Clean up double slashes in SOURCE path
SOURCE=$(echo "$SOURCE" | sed 's|^//|/|')
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"
WANDB_DIR="${OUTPUT}/wandb"

# Ensure output directories exist
mkdir -p "${FINETUNE_DIR}" "${LOGS_DIR}" "${TENSORBOARD_DIR}" "${WANDB_DIR}"

CODE_DIR="${SOURCE}"

TP=2
EP=64
MAIN_HYBRID_PATTERN="MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
HYBRID_LAYER_PATTERN="${MAIN_HYBRID_PATTERN}"

# Starting checkpoint: 16k omni SVG output
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/lustre/fsw/portfolios/llmservice/users/${USER}/workspace/output/sft_super_omni_16k_svg_newcontainer_0422_iter_20000/checkpoints"}

TOKENIZER_MODEL=${TOKENIZER_MODEL:-"/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/workspace/nemotron_3_nano_30b_a3b_tokenizer"}

TOKENIZER_PROMPT_FORMAT="nemotron6-moe"
DATA_TRAIN=${DATA_TRAIN:-"${SOURCE}/examples/multimodal/super/data_config/yamls/sft.long_context.49k.ehsan.v13p77.2x.0330.yaml"}

SOUND_MODEL_CACHE="${WORKSPACE}/models/parakeet-tdt-0.6b-v2.nemo"
SOUND_MODEL_TYPE="nemo://${SOUND_MODEL_CACHE}"

if [ -n "${WANDB_API_KEY}" ]; then
    EXTRA_ARGS+=" --wandb-project ${WANDB_PROJECT} --wandb-exp-name ${MODEL_NAME} --wandb-save-dir ${WANDB_DIR}"
fi

if [[ $USE_TILING -eq 1 ]]; then
    EXTRA_ARGS+=" --pixel-shuffle --use-tiling --max-num-tiles 12 --use-thumbnail"
    SEQ_LEN=256
fi

if [[ $USE_FP8 -eq 1 ]]; then
    EXTRA_ARGS+=" --fp8-recipe blockwise --fp8-format e4m3 --first-last-layers-bf16 --num-layers-at-start-in-bf16 1 --num-layers-at-end-in-bf16 1"
    EXTRA_ARGS+=" --use-vision-backbone-fp8-arch "
fi

if [[ $USE_DYNAMIC_RES -eq 1 ]]; then
    SEQ_LEN=12288
    if [[ $USE_IMAGE_BREAK -eq 1 ]]; then
        if [[ $BATCH -eq 0 ]]; then
            EXTRA_ARGS+=" --image-break-token <image_break>"
            SPECIAL_TOKENS+=" <image_break>"
        else
            EXTRA_ARGS+=" --image-break-token \<image_break\>"
            SPECIAL_TOKENS+=" \<image_break\>"
        fi
    fi
    if [[ $USE_CONV_MERGE -eq 1 ]]; then
        EXTRA_ARGS+=" --conv-merging --allow-missing-conv-merge-checkpoint"
    else
        EXTRA_ARGS+=" --pixel-shuffle"
    fi
    EXTRA_ARGS+=" --dynamic-resolution --dynamic-resolution-min-patches 1024 --dynamic-resolution-max-patches 13312"
fi

if [[ $USE_CPE_EVAL_MODE -eq 1 ]]; then
    EXTRA_ARGS+=" --radio-force-cpe-eval-mode"  # Only CPE in eval mode (not entire vision encoder)
fi

if [[ $USE_MTP -eq 1 ]]; then
    HYBRID_LAYER_PATTERN="${MAIN_HYBRID_PATTERN}"
    EXTRA_ARGS+=" --mtp-num-layers 2"
    EXTRA_ARGS+=" --mtp-hybrid-override-pattern '*E'"
    EXTRA_ARGS+=" --mtp-loss-scaling-factor ${MTP_LOSS_SCALING_FACTOR}"
    EXTRA_ARGS+=" --mtp-use-repeated-layer"
    EXTRA_ARGS+=" --keep-mtp-spec-in-bf16"
else
    HYBRID_LAYER_PATTERN="${MAIN_HYBRID_PATTERN}"
fi

if [[ $EARLY_EXIT_ITERS -gt 0 ]]; then
    EXTRA_ARGS+=" --early-exit-iters ${EARLY_EXIT_ITERS} "
fi

if [[ "$VIDEO_MAINTAIN_ASPECT_RATIO" -eq 1 ]]; then
    EXTRA_ARGS+=" --video-maintain-aspect-ratio "
fi

if [[ $SEPARATE_VIDEO_EMBEDDER -eq 1 ]]; then
    EXTRA_ARGS+=" --separate-video-embedder "
fi

# online packing & bucketing
if [[ $USE_PACKING -eq 1 ]]; then
    EXTRA_ARGS+=" --packing-buffer-size ${PBS} --packing-seq-length ${DECODER_SEQ_LEN}"
    if [[ $USE_BUCKETING -eq 1 ]]; then
        EXTRA_ARGS+=" --packing-knapsack-algorithm bucketing_greedy_knapsack "
    else
        EXTRA_ARGS+=" --packing-knapsack-algorithm balanced_greedy_knapsack "
    fi
fi

# Context parallelism for long sequences
if [[ $USE_CP -eq 1 ]]; then
    EXTRA_ARGS+=" --context-parallel-size ${CP_SIZE}"
fi

if [[ $FREEZE_SOUND -eq 1 ]]; then
    EXTRA_ARGS+=" --freeze-sound-model --freeze-sound-projection "
fi

if [[ $USE_LOSS_SCALING -eq 1 ]]; then
    EXTRA_ARGS+=" --use-loss-scaling "
fi

# Video augmentation
if [[ "$VIDEO_AUG_SCALE_FRAMES_UP" != "None" ]]; then
    EXTRA_ARGS+=" --video-aug-scale-frames-up ${VIDEO_AUG_SCALE_FRAMES_UP} "
fi
if [[ "$VIDEO_AUG_SCALE_RESOLUTION_UP" != "None" ]]; then
    EXTRA_ARGS+=" --video-aug-scale-resolution-up ${VIDEO_AUG_SCALE_RESOLUTION_UP} "
fi
if [[ $VIDEO_AUG_SCALE_RESOLUTION_ONLY -eq 1 ]]; then
    EXTRA_ARGS+=" --video-aug-scale-resolution-only "
fi

CHECKPOINT_ARGS=" \
    --pretrained-checkpoint ${CHECKPOINT_DIR} \
    --load ${FINETUNE_DIR} \
    --ckpt-format torch \
"

if [[ $SKIP_SAVE -eq 0 ]]; then
    CHECKPOINT_ARGS+=" \
        --save ${FINETUNE_DIR} \
        --dataloader-save ${FINETUNE_DIR}/dataloader \
        --save-interval 10000 \
    "
fi

# LM (Mamba block) recompute — more aggressive for longer video sequences
# EXTRA_ARGS+=" --recompute-granularity selective --recompute-modules core_attn mlp layernorm moe_act moe "
# Vision (GPT block) recompute
# EXTRA_ARGS+=" --recompute-vision --recompute-method-vision block --recompute-granularity-vision full --recompute-vision-num-layers 32 "
# EXTRA_ARGS+=" --recompute-vision-projection "
# Sound model recompute
# EXTRA_ARGS+=" --recompute-sound "
# EXTRA_ARGS+=" --recompute-sound-projection "

# LM recompute
EXTRA_ARGS+=" --recompute-granularity selective --recompute-modules mlp moe "

# Vision recompute
EXTRA_ARGS+=" --recompute-vision --recompute-method-vision block --recompute-granularity-vision full --recompute-vision-num-layers 32 "
EXTRA_ARGS+=" --recompute-vision-projection "

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
    --moe-grouped-gemm \
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
    --hybrid-layer-pattern '${HYBRID_LAYER_PATTERN}' \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --tiktoken-pattern v2 \
    --distributed-timeout-minutes 60 \
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
    --lr-warmup-fraction 0.01 \
    --lr 1e-6 \
    --min-lr 0.0 \
    --lr-decay-style cosine \
    --weight-decay 0.05 \
    --clip-grad 1.0 \
    --log-interval ${LI} \
    --eval-iters 0 \
    --eval-interval 99999999999 \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --tokenizer-prompt-format ${TOKENIZER_PROMPT_FORMAT} \
    ${CHECKPOINT_ARGS} \
    --log-progress  \
    --timing-log-option minmax \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --logging-level 20 \
    --log-memory-interval 500 \
    --bf16 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --use-distributed-optimizer \
    --num-workers ${NW} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --sequence-parallel \
    --allow-large-videos \
    --sound-model-type ${SOUND_MODEL_TYPE} \
    --sound-target-rate 16000 \
    --sound-embedding-size 751 \
    --video-target-num-patches ${VIDEO_TARGET_NUM_PATCHES} \
    --video-min-num-frames 8 \
    --video-max-num-frames ${VIDEO_MAX_NUM_FRAMES} \
    --video-temporal-patch-size 2 \
    --video-prompt-version 2 \
    --tokenizer-keep-history-thinking \
"

export WANDB_ENTITY=$WANDB_ENTITY  # Not passed in via command line args, only env vars
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

# Interactive or batch mode
if [[ $BATCH -eq 0 ]]; then
    cd ${CODE_DIR}
    torchrun --nproc_per_node ${NUM_GPU} examples/multimodal/train.py ${OPTIONS}
else
    run_cmd="python -u ${CODE_DIR}/examples/multimodal/train.py ${OPTIONS}"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo ${run_cmd}
        exit 0
    fi

    DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

    # Download the NeMo sound model once to shared Lustre if not already cached.
    # Uses a single task to avoid HuggingFace HTTP 429 rate-limiting.
    if [ ! -f "${SOUND_MODEL_CACHE}" ]; then
        mkdir -p "$(dirname ${SOUND_MODEL_CACHE})"
        srun -l --verbose \
        --ntasks=1 \
        --container-image /lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/containers/pytorch25.11-moe-avlm-editable-energon-super.sqsh \
        --container-mounts "/lustre,/scratch" \
        --output=${LOGS_DIR}/%x_%j_${DATETIME}_predownload.log \
        sh -c "python -c \"from huggingface_hub import hf_hub_download; hf_hub_download('nvidia/parakeet-tdt-0.6b-v2', 'parakeet-tdt-0.6b-v2.nemo', local_dir='$(dirname ${SOUND_MODEL_CACHE})'); print('Sound model download complete')\""
    fi

    # Need TRITON_CACHE_DIR expanded inside srun b/c sbatch runs on node 0
    # Need to use orig (non-symlink'ed) path for container image or slurm fails to launch
    srun -l --verbose \
    --container-image /lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/containers/pytorch25.11-moe-avlm-editable-energon-super.sqsh \
    --container-mounts "/lustre,/scratch" \
    --output=${LOGS_DIR}/%x_%j_srun_$DATETIME.log \
    sh -c "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH; export TRITON_CACHE_DIR=/tmp/triton_cache_\${SLURM_NODEID}; \
    export USE_PRECISION_AWARE_OPTIMIZER=1; \
    export AUDIO_MAX_DURATION_SECONDS=${AUDIO_MAX_DURATION_SECONDS}; \
    echo ${run_cmd}; ${run_cmd}"

    set +x
fi
