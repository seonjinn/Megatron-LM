#!/bin/bash

#SBATCH -A llmservice_fm_vision
#SBATCH -p batch_block1,batch_large,batch_long
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton
#SBATCH --nodes=64
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --gpus-per-node=8
#SBATCH --job-name=sft_moe_rl_llm_2e_bs_x2_radio_v4_h_rc2_conv3d_vf32a_videoaug_v1377_0324

# !! Example launch command !!
# examples/multimodal/launch.sh \
# --name sft_moe_rl_llm_2e_bs_x2_radio_v4_h_rc2_conv3d_vf32a_videoaug_v1377_0324 \
# --sbatch examples/multimodal/v3_conv3d/sft_moe_rl_llm_eval_mode_radio_v4_two_epochs_bs_x2_videoaug.sh \
# --num-jobs 10

# Strict mode: exit immediately on failure (-e), treat unset vars as error (-u), mark any failures as whole pipeline (-o pipefail)
# Combined these ensure that the job is reliably marked as failed so we can use `--dependency afterok:<jobid>`
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MSC_CONFIG="/lustre/fsw/portfolios/llmservice/users/matthieul/msc_config/msc_config.yaml"

export UB_TIMEOUT=720
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN
export TORCHINDUCTOR_WORKER_START=fork
#export TORCH_NCCL_AVOID_RECORD_STREAMS=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

USER=${SLURM_JOB_USER:-${USER}}

# Defaults, hard-coded
USE_TILING=0
USE_DYNAMIC_RES=1
USE_IMAGE_BREAK=0   # Only used if USE_DYNAMIC_RES is 1.
USE_CONV_MERGE=0    # Only used if USE_DYNAMIC_RES is 1.
USE_FP8=0
USE_CPE_EVAL_MODE=1

# Remember to update model and job name if running in batch mode!!
MODEL_NAME=${MODEL_NAME:-"sft_moe_rl_llm_2e_bs_x2_radio_v4_h_rc2_conv3d_vf32a_videoaug_v1377_0324"}
DATA_TRAIN=${DATA_TRAIN:-"${SOURCE}/examples/multimodal/v3_baseline/1377_video_text.yaml"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/lustre/fsw/portfolios/llmservice/users/amalasanjayd/workspace/output/pretrain_vision_adaptor_v1376_cleaned_sft05_0320/checkpoints"}

# Defaults, w/ allowed overrides (via env vars)
VIDEO_MAX_NUM_FRAMES=${VIDEO_MAX_NUM_FRAMES:-64}  # Using conv3d w/ 2-frame compression
SEPARATE_VIDEO_EMBEDDER=${SEPARATE_VIDEO_EMBEDDER:-1}
VIDEO_MAINTAIN_ASPECT_RATIO=${VIDEO_MAINTAIN_ASPECT_RATIO:-1}
VIDEO_TARGET_NUM_PATCHES=${VIDEO_TARGET_NUM_PATCHES:-1024}

# This will effectively scale `video_target_num_patches` down by 1-4x, between 256-1024
# TODO: Change this setting to something like `--video-scale-resolution-down 4`
VIDEO_AUG_SCALE_FRAMES_UP=${VIDEO_AUG_SCALE_FRAMES_UP:-4}
VIDEO_AUG_SCALE_RESOLUTION_UP=${VIDEO_AUG_SCALE_RESOLUTION_UP:-None}
VIDEO_AUG_SCALE_RESOLUTION_ONLY=${VIDEO_AUG_SCALE_RESOLUTION_ONLY:-1}

# Used by examples/multimodal/launch.sh (or can be overridden via exports)
DEBUG=${DEBUG:-0}  # Sets DEBUG_RANK, megatron attaches debugger and waits; requires interactive session
DRY_RUN=${DRY_RUN:-0}  # Prints launch command and exits
EARLY_EXIT_ITERS=${EARLY_EXIT_ITERS:-0}  # Exit early for testing

if [[ $DEBUG -eq 1 ]]; then
    # Debugging launches the debugger, so it's a non-interactive launch in terms of escaping "<>"
    INTERACTIVE=0

    # Append _debug to MODEL_NAME and WORKSPACE to more easily delete all debug runs
    MODEL_NAME="${MODEL_NAME}_debug"
    WORKSPACE="/lustre/fsw/portfolios/llmservice/users/${USER}/workspace/debug"

else
    # If we've SSH'd into the allocation from a new terminal, `srun` will be on path still,
    #   so need to explicitly pass in `INTERACTIVE=1 <script>` (allow override here)
    INTERACTIVE=${INTERACTIVE:-$(which srun >/dev/null 2>&1 && echo 0 || echo 1)}

    # Normal workspace
    WORKSPACE="/lustre/fsw/portfolios/llmservice/users/${USER}/workspace"
fi

if [[ $INTERACTIVE -eq 1 ]]; then
    MODEL_NAME="interactive_${MODEL_NAME}"
    SPECIAL_TOKENS="--special-tokens <image> <img> </img> <quad> </quad> <ref> </ref> <box> </box>"
else
    SPECIAL_TOKENS="--special-tokens \<image\> \<img\> \</img\> \<quad\> \</quad\> \<ref\> \</ref\> \<box\> \</box\>"
fi

WANDB_API_KEY=${WANDB_API_KEY}
WANDB_PROJECT=${WANDB_PROJECT:-"megatron-vlm-v3"}
WANDB_ENTITY=${WANDB_ENTITY:-"adlr"}
WANDB_NAME=${MODEL_NAME}

SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"
WANDB_DIR="${OUTPUT}/wandb"

# Ensure output directories exist
mkdir -p "${FINETUNE_DIR}" "${LOGS_DIR}" "${TENSORBOARD_DIR}" "${WANDB_DIR}"

# Snapshot the source code into the OUTPUT directory on first run, and always run from the snapshot thereafter
# NOTE: Don't recommend this method anymore, code isn't copied until run executes (after queuing)
#       See examples/multimodal/launch.sh instead, which copies when the run is launched
if [[ $INTERACTIVE -eq 0 && $DEBUG -eq 0 && $DRY_RUN -eq 0 ]]; then
    CODE_SNAPSHOT_DIR="${OUTPUT}/code_snapshot"
    if [[ ! -d "${CODE_SNAPSHOT_DIR}" ]]; then
        echo "[info] Creating code snapshot at ${CODE_SNAPSHOT_DIR} from ${SOURCE}"
        rsync -a --delete \
            --exclude "__pycache__" \
            --exclude "*.pyc" \
            --exclude "wandb/" \
            "${SOURCE}/" "${CODE_SNAPSHOT_DIR}/"
    fi
    CODE_DIR="${CODE_SNAPSHOT_DIR}"
else
    CODE_DIR="${SOURCE}"
fi

TP=2
EP=32

# New tokenizer 1/26/25
TOKENIZER_MODEL="/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/nemotron_3_nano_30b_a3b_tokenizer"
TOKENIZER_PROMPT_FORMAT="nemotron6-moe"

if [[ $DEBUG -eq 1 || $INTERACTIVE -eq 1 ]]; then
    MBZ=1
    BZ=16
    NW=0
    AD=0.0
    HD=0.0
    LI=1

    EXTRA_ARGS=""

    NUM_GPU=$SLURM_GPUS_ON_NODE
else
    MBZ=1
    BZ=256
    NW=8
    AD=0.0
    HD=0.0
    LI=5
    EXTRA_ARGS=""
    NUM_GPU=8
fi

SEQ_LEN=256
DECODER_SEQ_LEN=16384

if [ -n "${WANDB_API_KEY}" ]; then
    EXTRA_ARGS+=" --wandb-project ${WANDB_PROJECT} --wandb-exp-name ${MODEL_NAME} --wandb-save-dir ${WANDB_DIR} --wandb-resume-same-run"
fi

if [[ $USE_TILING -eq 1 ]]; then
    EXTRA_ARGS+=" --pixel-shuffle --use-tiling --max-num-tiles 12 --use-thumbnail"
    SEQ_LEN=256
fi

if [[ $USE_FP8 -eq 1 ]]; then
    # Recipe 1: More accurate but not the fastest.
    EXTRA_ARGS+=" --fp8-recipe blockwise --fp8-format e4m3 --first-last-layers-bf16 --num-layers-at-start-in-bf16 1 --num-layers-at-end-in-bf16 1"
    # Recipes 2 and 3: Faster but metrics can become a bit noisier. Still the difference to bf16 should be small < 1%.
    #EXTRA_ARGS+=" --fp8-recipe blockwise --fp8-format e4m3 "
    #EXTRA_ARGS+=" --fp8-recipe blocwise --fp8-format e4m3 --fp8-param-gather "
    EXTRA_ARGS+=" --use-vision-backbone-fp8-arch "
fi

if [[ $USE_DYNAMIC_RES -eq 1 ]]; then
    SEQ_LEN=12288
    if [[ $USE_IMAGE_BREAK -eq 1 ]]; then
        if [[ $INTERACTIVE -eq 1 ]]; then
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


if [[ $EARLY_EXIT_ITERS -gt 0 ]]; then
    EXTRA_ARGS+=" --early-exit-iters ${EARLY_EXIT_ITERS} "
fi


if [[ "$VIDEO_MAINTAIN_ASPECT_RATIO" -eq 1 ]]; then
    EXTRA_ARGS+=" --video-maintain-aspect-ratio "
fi

if [[ $SEPARATE_VIDEO_EMBEDDER -eq 1 ]]; then
    EXTRA_ARGS+=" --separate-video-embedder "
fi

EXTRA_ARGS+=" --packing-buffer-size 3247 --packing-seq-length ${DECODER_SEQ_LEN} --packing-knapsack-algorithm balanced_greedy_knapsack "

if [[ $VIDEO_MAX_NUM_FRAMES -ge 32 ]]; then
    # Need more activation recomputation for increased max frames; getting CUDA OOM w/ conv3d and 32 frames
    # LM (Mamba block) recompute
    EXTRA_ARGS+=" --recompute-granularity selective --recompute-modules core_attn mlp layernorm moe_act moe "
    # Vision (GPT block) recompute
    EXTRA_ARGS+=" --recompute-vision --recompute-method-vision block --recompute-granularity-vision full --recompute-vision-num-layers 32 "
else
    # Can afford less activation recomputation for fewer max frames
    # LM (Mamba block) recompute
    EXTRA_ARGS+=" --recompute-granularity selective --recompute-modules core_attn layernorm moe_act "
    # Vision (GPT block) recompute
    EXTRA_ARGS+=" --recompute-vision --recompute-method-vision block --recompute-granularity-vision full --recompute-vision-num-layers 28 "
fi

if [[ "$VIDEO_AUG_SCALE_FRAMES_UP" != "None" ]]; then
    EXTRA_ARGS+=" --video-aug-scale-frames-up ${VIDEO_AUG_SCALE_FRAMES_UP} "
fi
if [[ "$VIDEO_AUG_SCALE_RESOLUTION_UP" != "None" ]]; then
    EXTRA_ARGS+=" --video-aug-scale-resolution-up ${VIDEO_AUG_SCALE_RESOLUTION_UP} "
fi
if [[ $VIDEO_AUG_SCALE_RESOLUTION_ONLY -eq 1 ]]; then
    EXTRA_ARGS+=" --video-aug-scale-resolution-only "
fi

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
    --moe-aux-loss-coeff 1e-8 \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-shared-expert-intermediate-size 3712 \
    --is-hybrid-model \
    --mamba-num-heads 64 \
    --mamba-head-dim 64 \
    --hybrid-override-pattern MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --tiktoken-pattern v2 \
    --distributed-timeout-minutes 20 \
    --use-mcore-models \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --init-method-std 0.014 \
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
    --train-full-dataset \
    --lr-warmup-fraction 0.1 \
    --lr 5e-5 \
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
    --pretrained-checkpoint ${CHECKPOINT_DIR} \
    --load ${FINETUNE_DIR} \
    --save ${FINETUNE_DIR} \
    --dataloader-save ${FINETUNE_DIR}/dataloader \
    --save-interval 5000 \
    --ckpt-format torch \
    --bf16 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --use-distributed-optimizer \
    --num-workers ${NW} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --sequence-parallel \
    --allow-large-videos \
    --class-token-len 10 \
    --tokenizer-keep-history-thinking \
    --log-model-grad-norms \
    --log-model-act-norms \
    --video-target-num-patches ${VIDEO_TARGET_NUM_PATCHES} \
    --video-max-num-frames ${VIDEO_MAX_NUM_FRAMES} \
    --video-temporal-patch-size 2 \
    --video-prompt-version 2 \
    --allow-checkpoint-without-temporal-compression \
"

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

    DEBUG_RANK=${DEBUG_RANK:-0}  # Default to rank 0
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
    run_cmd="python -u ${CODE_DIR}/examples/multimodal/train.py ${OPTIONS}"

    DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

    # Need TRITON_CACHE_DIR expanded inside srun b/c sbatch runs on node 0
    srun -l --verbose \
    --container-image /lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/containers/pytorch25.06-moe-avlm-editable-energon.sqsh \
    --container-mounts "/lustre,/home" \
    --output=${LOGS_DIR}/%x_%j_srun_$DATETIME.log \
    sh -c "export TRITON_CACHE_DIR=/tmp/triton_cache_\${SLURM_NODEID}; echo ${run_cmd}; ${run_cmd}"

    set +x
fi
