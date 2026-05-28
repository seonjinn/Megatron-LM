#!/bin/bash

#SBATCH -A llmservice_fm_vision
# batch_block1 + llmservice_fm_vision; match sft_moe_long_context_49k_conv3d.v13p77.2x.higher_lr.0331.sh
# Slurm headers (--mem=0 + --exclusive) so the scheduler bills/allocates full node RAM + CPUs with 8 GPUs.
# QOS / mem: Slurm FAQ "Partition/QoS Troubleshooting" — per-user limits combine mem/CPU/GPU; --mem=0
# raises billed memory (use chained submit script to avoid many large pending jobs).
# https://nvidia.atlassian.net/wiki/spaces/HWINFCSSUP/pages/2441646090/Slurm+FAQ+Troubleshooting
#SBATCH -p batch_block1
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton
#SBATCH --nodes=64
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --gpus-per-node=8
#SBATCH --job-name=sft_moe_long_context_49k_conv3d_v13p77_2x_0331

# !! Example launch command !!
# examples/multimodal/launch.sh --name sft_omni_video_0225 --sbatch examples/multimodal/v3_omni_staged_conv3d/sft_video.sh --num-jobs 5

# Strict mode: exit immediately on failure (-e), treat unset vars as error (-u), mark any failures as whole pipeline (-o pipefail)
# Combined these ensure that the job is reliably marked as failed so we can use `--dependency afterok:<jobid>`
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MSC_CONFIG="/lustre/fsw/portfolios/llmservice/users/matthieul/msc_config/msc_config.yaml"

export UB_TIMEOUT=720
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN
export TORCHINDUCTOR_WORKER_START=fork
#export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"/tmp/triton_cache_\${SLURM_NODEID}"}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

USER=${SLURM_JOB_USER:-${USER}}

# Auto-detect batch or interactive mode.
which srun
BATCH=$((1-$?))

DEBUG=0
INTERACTIVE=0
DRY_RUN=0
EARLY_EXIT_ITERS=0

# Defaults, hard-coded
USE_TILING=0
USE_DYNAMIC_RES=1
USE_IMAGE_BREAK=0   # Only used if USE_DYNAMIC_RES is 1.
USE_CONV_MERGE=0    # Only used if USE_DYNAMIC_RES is 1.
USE_FP8=0
USE_PACKING=1
USE_BUCKETING=0
USE_PRECISION_AWARE_OPTIMIZER=1

USE_CPE_EVAL_MODE=1
USE_VISION_ENCODER_EVAL_MODE=0
AUDIO_MAX_DURATION_SECONDS=1200
NUM_EPOCHS=${NUM_EPOCHS:-4}
# ITERS_PER_EPOCH=${ITERS_PER_EPOCH:-9169}
# TRAIN_ITERS=$((NUM_EPOCHS * ITERS_PER_EPOCH))
USE_CP=1
CP_SIZE=2 #2 #8 #16 #2

# # Remember to update model and job name if running in batch mode!!
# MODEL_NAME=${MODEL_NAME:-"sft_omni_video_0225"}
# DATA_TRAIN=${DATA_TRAIN:-"__CODE_DIR__/examples/multimodal/v3_omni_staged_conv3d/sft_recipe_video.yaml"}

# Conv3d video defaults (w/ allowed overrides via env vars)
VIDEO_MAX_NUM_FRAMES=${VIDEO_MAX_NUM_FRAMES:-256}  # Using conv3d w/ 2-frame compression
SEPARATE_VIDEO_EMBEDDER=${SEPARATE_VIDEO_EMBEDDER:-1}
VIDEO_MAINTAIN_ASPECT_RATIO=${VIDEO_MAINTAIN_ASPECT_RATIO:-1}

# Video augmentation
VIDEO_TARGET_NUM_PATCHES=${VIDEO_TARGET_NUM_PATCHES:-1024}
# This will effectively scale `video_target_num_patches` down by 1-4x, between 256-1024
# TODO: Change this setting to something like `--video-scale-resolution-down 4`
VIDEO_AUG_SCALE_FRAMES_UP=${VIDEO_AUG_SCALE_FRAMES_UP:-4}
VIDEO_AUG_SCALE_RESOLUTION_UP=${VIDEO_AUG_SCALE_RESOLUTION_UP:-None}
VIDEO_AUG_SCALE_RESOLUTION_ONLY=${VIDEO_AUG_SCALE_RESOLUTION_ONLY:-1}

EXTRA_ARGS=""

if [[ "$VIDEO_AUG_SCALE_FRAMES_UP" != "None" ]]; then
    EXTRA_ARGS+=" --video-aug-scale-frames-up ${VIDEO_AUG_SCALE_FRAMES_UP} "
fi
if [[ "$VIDEO_AUG_SCALE_RESOLUTION_UP" != "None" ]]; then
    EXTRA_ARGS+=" --video-aug-scale-resolution-up ${VIDEO_AUG_SCALE_RESOLUTION_UP} "
fi
if [[ $VIDEO_AUG_SCALE_RESOLUTION_ONLY -eq 1 ]]; then
    EXTRA_ARGS+=" --video-aug-scale-resolution-only "
fi



if [[ $DEBUG -eq 1 || $INTERACTIVE -eq 1 ]]; then
    MBZ=1
    BZ=16
    NW=0
    AD=0.0
    HD=0.0
    LI=1

    EP=2
    HYBRID_PATTERN="MEM*EMEM*EMEM*EMEM*EMEME"
    NUM_LAYERS=24

    EXTRA_ARGS+=" --freeze-LM"

    NUM_GPU=${SLURM_GPUS_ON_NODE:-8}
    PBS=100
    LR=1e-6
    LR_WARMUP_FRAC=0.01
    MIN_LR=0.0
    WD=0.05
    CG=1.0
    LR_DECAY_STYLE="cosine"
    MOE_AUX_LOSS_COEFF=1e-8
else
    MBZ=1
    BZ=256 #128
    NW=8
    AD=0.0
    HD=0.0
    LI=5
    HYBRID_PATTERN="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
    NUM_LAYERS=52

    # EXTRA_ARGS=" --pretrained-checkpoint ${CHECKPOINT_DIR}"
    NUM_GPU=8 #8
    PBS=500 #1000 #1000 #5000 #1000 #500 #20000 #1000 #1000 #50 #100 #200 #500 #1000
    LR=1e-6 #5e-5 #1e-6 #1e-5 #5e-6 #4e-5
    LR_WARMUP_FRAC=0.01
    MIN_LR=0.0 #1e-7 #0.0
    WD=0.05 # weight decay
    CG=1.0 # clip grad
    LR_DECAY_STYLE="cosine" # lr decay style
    INIT_METHOD_STD=0.014 # init method std
    MOE_AUX_LOSS_COEFF=1e-8 #1e-8
fi

SEQ_LEN=256
DECODER_SEQ_LEN=49152
# DECODER_SEQ_LEN=81920

TP=2
EP=32


# # Used by examples/multimodal/launch.sh (or can be overridden via exports)
# DEBUG=${DEBUG:-0}  # Sets DEBUG_RANK, megatron attaches debugger and waits; requires interactive session
# DRY_RUN=${DRY_RUN:-0}  # Prints launch command and exits
# EARLY_EXIT_ITERS=${EARLY_EXIT_ITERS:-0}  # Exit early for testing

# if [[ $DEBUG -eq 1 ]]; then
#     INTERACTIVE=0
#     MODEL_NAME="${MODEL_NAME}_debug"
#     WORKSPACE="/lustre/fsw/portfolios/llmservice/users/${USER}/workspace/debug"
# else
#     INTERACTIVE=${INTERACTIVE:-$(which srun >/dev/null 2>&1 && echo 0 || echo 1)}
#     WORKSPACE="/lustre/fsw/portfolios/llmservice/users/${USER}/workspace"
# fi

# if [[ $INTERACTIVE -eq 1 ]]; then
#     MODEL_NAME="interactive_${MODEL_NAME}"
#     SPECIAL_TOKENS="--special-tokens <image> <img> </img> <quad> </quad> <ref> </ref> <box> </box> <so_embedding> <so_start> <so_end>"
# else
#     SPECIAL_TOKENS="--special-tokens \<image\> \<img\> \</img\> \<quad\> \</quad\> \<ref\> \</ref\> \<box\> \</box\> \<so_embedding\> \<so_start\> \<so_end\>"
# fi

# Remember to update model and job name if running in batch mode!!
if [[ $BATCH -eq 0 ]]; then
    DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
    MODEL_NAME="interactive_sft_moe_long_context_49k_${DATETIME}"
    SPECIAL_TOKENS=" --special-tokens <image> <img> </img> <quad> </quad> <ref> </ref> <box> </box> <so_embedding> <so_start> <so_end> "
    DEBUG=1
else
    if [[ $USE_CP -eq 1 ]]; then
        MODEL_NAME="sft_moe_long_context_conv3d_v13p77_2x_nodes${SLURM_NNODES}-seq${DECODER_SEQ_LEN}-lr${LR}-0331"
    else
        MODEL_NAME="sft_moe_long_context_conv3d_v13p77_2x_nodes${SLURM_NNODES}-seq${DECODER_SEQ_LEN}-lr${LR}-0331"
    fi
    SPECIAL_TOKENS=" --special-tokens \<image\> \<img\> \</img\> \<quad\> \</quad\> \<ref\> \</ref\> \<box\> \</box\> \<so_embedding\> \<so_start\> \<so_end\> "
fi

WORKSPACE="/lustre/fsw/portfolios/llmservice/users/${USER}/workspace"



SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"
WANDB_DIR="${OUTPUT}/wandb"

# WANDB_API_KEY=${WANDB_API_KEY}
# WANDB_PROJECT=${WANDB_PROJECT:-"megatron-vlm-v3"}
# WANDB_ENTITY=${WANDB_ENTITY:-"adlr"}
# WANDB_NAME=${MODEL_NAME}

WANDB_API_KEY="9db2ded47edc63ecb92f98626c13c615e0f385e4"
WANDB_PROJECT=${WANDB_PROJECT:-"megatron-omni-v3"}
WANDB_ENTITY=${WANDB_ENTITY:-"adlr"}
WANDB_NAME=${MODEL_NAME}



# Ensure output directories exist
mkdir -p "${FINETUNE_DIR}" "${LOGS_DIR}" "${TENSORBOARD_DIR}" "${WANDB_DIR}"

# Snapshot the source code into the OUTPUT directory on first run, and always run from the snapshot thereafter
# NOTE: Don't recommend this method anymore, code isn't copied until run executes (after queuing)
#       See examples/multimodal/launch.sh instead, which copies when the run is launched
# if [[ $INTERACTIVE -eq 0 && $DEBUG -eq 0 && $DRY_RUN -eq 0 ]]; then
#     CODE_SNAPSHOT_DIR="${OUTPUT}/code_snapshot"
#     if [[ ! -d "${CODE_SNAPSHOT_DIR}" ]]; then
#         echo "[info] Creating code snapshot at ${CODE_SNAPSHOT_DIR} from ${SOURCE}"
#         rsync -a --delete \
#             --exclude "__pycache__" \
#             --exclude "*.pyc" \
#             --exclude "wandb/" \
#             "${SOURCE}/" "${CODE_SNAPSHOT_DIR}/"
#     fi
#     CODE_DIR="${CODE_SNAPSHOT_DIR}"
# else
#     CODE_DIR="${SOURCE}"
# fi
if [[ $DEBUG -eq 0 ]]; then
    CODE_SNAPSHOT_DIR="${OUTPUT}/code_snapshot"
    CODE_DIR="${SOURCE}"
    if [[ ! -d "${CODE_SNAPSHOT_DIR}" ]]; then
        echo "[info] Creating code snapshot at ${CODE_SNAPSHOT_DIR} from ${SOURCE}"
        rsync -a --delete \
            --exclude "__pycache__" \
            --exclude "*.pyc" \
            "${SOURCE}/" "${CODE_SNAPSHOT_DIR}/"
    fi
    CODE_DIR="${CODE_SNAPSHOT_DIR}"
else
    CODE_DIR="${SOURCE}"
fi



if [ -n "${WANDB_API_KEY}" ]; then
    EXTRA_ARGS+=" --wandb-project ${WANDB_PROJECT} --wandb-exp-name ${MODEL_NAME} --wandb-save-dir ${WANDB_DIR} --wandb-resume-same-run"
fi



# Starting checkpoint: omni SFT checkpoint (update this path)
# CHECKPOINT_DIR=/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/workspace/output/matthieu_checkpoints/sft_omni_13p70_0224/checkpoints # iter 12067
# CHECKPOINT_DIR=/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/workspace/output/matthieu_checkpoints/sft_recipe_13p70_new_audio_0303/checkpoints # iter 20000
# CHECKPOINT_DIR=/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/workspace/output/matthieu_checkpoints/sft_recipe_13p70_0320/checkpoints # iter 7211
CHECKPOINT_DIR=/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/workspace/output/matthieu_checkpoints/sft_recipe_13p77_0329/checkpoints # iter 7000 (16K SFT)


# New tokenizer 1/26/25
TOKENIZER_MODEL="/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/nemotron_3_nano_30b_a3b_tokenizer"
TOKENIZER_PROMPT_FORMAT="nemotron6-moe"

# # Resolve DATA_TRAIN placeholder now that CODE_DIR is set
# DATA_TRAIN="${DATA_TRAIN/__CODE_DIR__/${CODE_DIR}}"


# DATA_TRAIN="${SOURCE}/examples/multimodal/avlm/data_config/sft.long_context.49k.ehsan.0320.yaml"
# DATA_TRAIN="${SOURCE}/examples/multimodal/avlm/data_config/sft.long_context.49k.ehsan.v13p77.0330.yaml"
DATA_TRAIN="${SOURCE}/examples/multimodal/avlm/long_context_49k_ga/sft.long_context.49k.ehsan.v13p77.2x.0330.yaml"


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

if [[ $USE_VISION_ENCODER_EVAL_MODE -eq 1 ]]; then
    EXTRA_ARGS+=" --radio-force-eval-mode"  # Entire vision encoder in eval mode (eval CPE, no dropout)
fi

if [[ $USE_CPE_EVAL_MODE -eq 1 ]]; then
    EXTRA_ARGS+=" --radio-force-cpe-eval-mode"  # Only CPE in eval mode (not entire vision encoder)
fi


# EXTRA_ARGS+=" --packing-buffer-size 3247 --packing-seq-length ${DECODER_SEQ_LEN} --packing-knapsack-algorithm balanced_greedy_knapsack "
if [[ $USE_PACKING -eq 1 ]]; then
    EXTRA_ARGS+=" --packing-buffer-size ${PBS} --packing-seq-length ${DECODER_SEQ_LEN}"
    if [[ $USE_BUCKETING -eq 1 ]]; then
        EXTRA_ARGS+=" --packing-knapsack-algorithm bucketing_greedy_knapsack "
    else
        EXTRA_ARGS+=" --packing-knapsack-algorithm balanced_greedy_knapsack "
    fi
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

if [[ $USE_CP -eq 1 ]]; then
    EXTRA_ARGS+=" --context-parallel-size ${CP_SIZE}"
fi




# LM (Mamba block) recompute — more aggressive for longer video sequences
EXTRA_ARGS+=" --recompute-granularity selective --recompute-modules core_attn mlp layernorm moe_act moe "
# Vision (GPT block) recompute
EXTRA_ARGS+=" --recompute-vision --recompute-method-vision block --recompute-granularity-vision full --recompute-vision-num-layers 32 "
EXTRA_ARGS+=" --recompute-vision-projection "
# Sound model recompute
EXTRA_ARGS+=" --recompute-sound "
EXTRA_ARGS+=" --recompute-sound-projection "



SOUND_MODEL_TYPE="nemo://nvidia/parakeet-tdt-0.6b-v2"

# TRAIN_SAMPLES=$((TRAIN_ITERS * BZ))
OPTIONS=" \
    --use-checkpoint-args \
    --transformer-impl transformer_engine \
    --use-te \
    --data-path ${DATA_TRAIN} \
    --patch-dim 16 \
    --img-h 512 \
    --img-w 512 \
    --dataloader-type external \
    --language-model-type nemotron6-moe \
    ${EXTRA_ARGS} \
    --vision-model-type radio \
    --use-loss-scaling \
    ${SPECIAL_TOKENS} \
    --disable-vision-class-token \
    --prompt-path ${SOURCE}/examples/multimodal/manual_prompts.json \
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
    --is-hybrid-model \
    --mamba-num-heads 64 \
    --mamba-head-dim 64 \
    --hybrid-override-pattern ${HYBRID_PATTERN} \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --tiktoken-pattern v2 \
    --distributed-timeout-minutes 60 \
    --use-mcore-models \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --init-method-std ${INIT_METHOD_STD} \
    --position-embedding-type none \
    --squared-relu \
    --num-layers ${NUM_LAYERS} \
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
    --train-full-dataset \
    --lr-warmup-fraction ${LR_WARMUP_FRAC} \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --weight-decay ${WD} \
    --clip-grad ${CG} \
    --lr-decay-style ${LR_DECAY_STYLE} \
    --log-interval ${LI} \
    --eval-iters 0 \
    --eval-interval 99999999999 \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --tokenizer-prompt-format ${TOKENIZER_PROMPT_FORMAT} \
    --pretrained-checkpoint ${CHECKPOINT_DIR} \
    --load ${FINETUNE_DIR} \
    --save ${FINETUNE_DIR} \
    --dataloader-save ${FINETUNE_DIR}/dataloader \
    --save-interval 200 \
    --ckpt-format torch \
    --log-progress  \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --bf16 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --use-distributed-optimizer \
    --num-workers ${NW} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --sequence-parallel \
    --allow-large-videos \
    --class-token-len 10 \
    --sound-model-type ${SOUND_MODEL_TYPE} \
    --sound-target-rate 16000 \
    --sound-embedding-size 751 \
    --video-target-num-patches ${VIDEO_TARGET_NUM_PATCHES} \
    --video-min-num-frames 8 \
    --video-max-num-frames ${VIDEO_MAX_NUM_FRAMES} \
    --video-temporal-patch-size 2 \
    --video-prompt-version 2 \
"
    # --tokenizer-keep-history-thinking \
    # --train-iters ${TRAIN_ITERS} \
    # --train-samples ${TRAIN_SAMPLES} \


export WANDB_ENTITY=$WANDB_ENTITY  # Not passed in via command line args, only env vars
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

if [[ $DRY_RUN -eq 1 ]]; then
    run_cmd="python -u ${CODE_DIR}/examples/multimodal/train.py ${OPTIONS}"
    echo "${run_cmd}"
    exit 0
fi

# Debug, interactive, or submit_job mode
if [[ $DEBUG -eq 1 ]]; then
    if [[ $SLURM_NNODES -gt 1 ]]; then
        echo "ERROR: Expected single-node debugging when using DEBUG_RANK environment variable."
        exit 1
    fi

    DEBUG_RANK=${DEBUG_RANK:-0}
    DEBUG_CMD="ONE_LOGGER_JOB_CATEGORY=test \
    CUDA_LAUNCH_BLOCKING=1 \
    DEBUG_RANK=${DEBUG_RANK} \
    WANDB_MODE=disabled \
    python -Xfrozen_modules=off \
    -m torch.distributed.run \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    ${CODE_DIR}/examples/multimodal/train.py ${OPTIONS}"

    echo -e "Debugging with options: $OPTIONS\n"
    eval "$DEBUG_CMD"

elif [[ $INTERACTIVE -eq 1 ]]; then
    torchrun --nproc_per_node ${NUM_GPU} ${CODE_DIR}/examples/multimodal/train.py ${OPTIONS}

else
    # run_cmd="python -u ${CODE_DIR}/examples/multimodal/train.py ${OPTIONS}"
    run_cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
    export USE_PRECISION_AWARE_OPTIMIZER=1; \
    export WANDB_API_KEY=${WANDB_API_KEY}; \
    export WANDB_ENTITY=${WANDB_ENTITY}; \
    export AUDIO_MAX_DURATION_SECONDS=${AUDIO_MAX_DURATION_SECONDS}; \
    export HF_HOME=/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/cache/huggingface; \
    export HF_HUB_CACHE=/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/cache/huggingface/hub; \
    export TRANSFORMERS_CACHE=/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/cache/transformers; \
    export HF_DATASETS_CACHE=/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/cache/datasets; \
    export TRITON_CACHE_DIR=/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/cache/triton; \
    mkdir -p /lustre/fsw/portfolios/llmservice/users/ehosseiniasl/cache/huggingface; \
    mkdir -p /lustre/fsw/portfolios/llmservice/users/ehosseiniasl/cache/transformers; \
    mkdir -p /lustre/fsw/portfolios/llmservice/users/ehosseiniasl/cache/datasets; \
    mkdir -p /lustre/fsw/portfolios/llmservice/users/ehosseiniasl/cache/triton; \
    python -u ${SOURCE}/examples/multimodal/train.py ${OPTIONS}"


    DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

    srun -l --verbose \
    --container-image /lustre/fsw/portfolios/llmservice/users/matthieul/docker/pytorch25.06-moe-avlm-editable-energon-732-mamba-fix.sqsh \
    --container-mounts "/lustre" \
    --output=${LOGS_DIR}/%x_%j_srun_$DATETIME.log \
    sh -c "export TRITON_CACHE_DIR=/tmp/triton_cache_\${SLURM_NODEID}; echo ${run_cmd}; ${run_cmd}"

    set +x
fi
