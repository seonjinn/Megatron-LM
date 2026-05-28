#!/bin/bash

#SBATCH -A llmservice_fm_vision
#SBATCH -p batch_block1,batch_large,batch_long
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton
#SBATCH --nodes=32
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --job-name=sft_nm_5p5_h_9b_radio_v4_so400m_rc3_tiling_1230

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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

USER=${SLURM_JOB_USER:-${USER}}

# Hard-coded
USE_IMAGE_BREAK=0   # Only used if USE_DYNAMIC_RES is 1.
USE_CONV_MERGE=0    # Only used if USE_DYNAMIC_RES is 1.
USE_PRECISION_AWARE_OPTIMIZER=1
USE_CP=0

# Can be overridden via exports
# NOTE: To run RADIO-H (not So400m), export modified MODEL_NAME, VISION_MODEL_TYPE, CHECKPOINT_DIR
#   export MODEL_NAME=sft_nm_5p5_h_9b_radio_v4_h_rc3_tiling_1230
#   export CHECKPOINT_DIR=/lustre/fsw/portfolios/llmservice/users/cmccarthy/workspace/output/pretrain_nm_5p5_h_9b_radio_v4_h_rc3_tiling_1230
#   export VISION_MODEL_TYPE=radio
#   export USE_FP8=1
MODEL_NAME=${MODEL_NAME:-"sft_nm_5p5_h_9b_radio_v4_so400m_rc3_tiling_1230"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/lustre/fsw/portfolios/llmservice/users/cmccarthy/workspace/output/pretrain_nm_5p5_h_9b_radio_v4_so400m_rc3_tiling_1230"}
VISION_MODEL_TYPE=${VISION_MODEL_TYPE:-"radio-so400m"}
USE_TILING=${USE_TILING:-1}
USE_DYNAMIC_RES=${USE_DYNAMIC_RES:-0}
USE_CPE_EVAL_MODE=${USE_CPE_EVAL_MODE:-0}
OVERWRITE_CODE_SNAPSHOT=${OVERWRITE_CODE_SNAPSHOT:-0}
USE_FP8=${USE_FP8:-0}  # Not supported for So400m-RC3
DRY_RUN=${DRY_RUN:-0}  # Prints launch command and exits
DEBUG=${DEBUG:-0}  # Sets DEBUG_RANK, requires interactive session
EARLY_EXIT_ITERS=${EARLY_EXIT_ITERS:-0}

if [[ $DEBUG -eq 1 ]]; then
    # Debugging launches the debugger, so it's a non-interactive launch in terms of escaping "<>"
    INTERACTIVE=0

    # Append _debug to MODEL_NAME and WORKSPACE to more easily delete all debug runs
    MODEL_NAME="${MODEL_NAME}_debug"
    WORKSPACE="/lustre/fsw/portfolios/llmservice/users/${USER}/workspace/debug"

else
    # Auto-detect interactive mode (if srun is not defined, we're interactive)
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
if [[ $INTERACTIVE -eq 0 && $DEBUG -eq 0 && $DRY_RUN -eq 0 ]]; then
    CODE_SNAPSHOT_DIR="${OUTPUT}/code_snapshot"
    if [[ ! -d "${CODE_SNAPSHOT_DIR}" || $OVERWRITE_CODE_SNAPSHOT -eq 1 ]]; then
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

TP=4

DATA_TRAIN="/lustre/fsw/portfolios/llmservice/users/matthieul/eagle_recipe_online_packing/final_recipe/eagle_sft_v13.52.no.text.yaml"

if [[ $DEBUG -eq 1 || $INTERACTIVE -eq 1 ]]; then
    MBZ=1
    BZ=8
    NW=0
    AD=0.0
    HD=0.0
    LI=1

    EXTRA_ARGS=""

    NUM_GPU=$SLURM_GPUS_ON_NODE
else
    MBZ=1
    BZ=128
    NW=8
    AD=0.0
    HD=0.0
    LI=5
    EXTRA_ARGS=""
    NUM_GPU=8
fi

SEQ_LEN=1024
DECODER_SEQ_LEN=16384

if [ -n "${WANDB_API_KEY}" ]; then
    EXTRA_ARGS+=" --wandb-project ${WANDB_PROJECT} --wandb-exp-name ${MODEL_NAME} --wandb-save-dir ${WANDB_DIR} --wandb-resume-same-run"
fi

if [[ $USE_TILING -eq 1 ]]; then
    EXTRA_ARGS+=" --pixel-shuffle --use-tiling --max-num-tiles 12 --use-thumbnail "
    SEQ_LEN=256
fi

if [[ $USE_FP8 -eq 1 ]]; then
    # Recipe 1: More accurate but not the fastest.
    EXTRA_ARGS+=" --fp8-recipe blockwise --fp8-format e4m3 --first-last-layers-bf16 --num-layers-at-start-in-bf16 1 --num-layers-at-end-in-bf16 1 "
    # Recipes 2 and 3: Faster but metrics can become a bit noisier. Still the difference to bf16 should be small < 1%.
    #EXTRA_ARGS+=" --fp8-recipe blockwise --fp8-format e4m3 "
    #EXTRA_ARGS+=" --fp8-recipe blocwise --fp8-format e4m3 --fp8-param-gather "
    EXTRA_ARGS+=" --use-vision-backbone-fp8-arch "
fi

if [[ $USE_PRECISION_AWARE_OPTIMIZER -eq 1 ]]; then
    EXTRA_ARGS+=" --use-precision-aware-optimizer --main-grads-dtype bf16 --main-params-dtype fp16 --exp-avg-dtype fp16 --exp-avg-sq-dtype fp16 "
fi

if [[ $USE_CP -eq 1 ]]; then
    # TODO: Loss scaling is not enabled for context parallel yet. Implementation exists but not committed yet.
    EXTRA_ARGS+=" --context-parallel-size 2 --sequence-parallel "
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
    elif [[ $NO_PIXEL_SHUFFLE -eq 0 ]]; then
        EXTRA_ARGS+=" --pixel-shuffle"
    fi
    EXTRA_ARGS+=" --dynamic-resolution --dynamic-resolution-min-patches 1024 --dynamic-resolution-max-patches 13312 --apply-data-augment"
fi

if [[ $USE_CPE_EVAL_MODE -eq 1 ]]; then
    EXTRA_ARGS+=" --radio-force-cpe-eval-mode"  # Only CPE in eval mode (not entire vision encoder)
fi

if [[ $EARLY_EXIT_ITERS -gt 0 ]]; then
    EXTRA_ARGS+=" --early-exit-iters ${EARLY_EXIT_ITERS} "
fi

EXTRA_ARGS+=" --packing-buffer-size 3247 --packing-seq-length ${DECODER_SEQ_LEN} --packing-knapsack-algorithm balanced_greedy_knapsack "

EXTRA_ARGS+=" --recompute-granularity full --recompute-method block --recompute-num-layers 56 --recompute-vision "

# LM (Mamba block) recompute
# EXTRA_ARGS+=" --recompute-granularity selective --recompute-modules core_attn mlp layernorm "
# Vision (GPT block) recompute
# EXTRA_ARGS+=" --recompute-vision --recompute-method-vision block --recompute-granularity-vision full --recompute-vision-num-layers 32 "


OPTIONS=" \
    --use-checkpoint-args \
    --disable-bias-linear \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model /lustre/fsw/portfolios/llmservice/users/ksapra/checkpoints/prunedmodel/first_9b_sft_pruned_v0 \
    --make-vocab-size-divisible-by 16512 \
    --transformer-impl transformer_engine \
    --normalization RMSNorm \
    --group-query-attention \
    --num-query-groups 8 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --attention-dropout ${AD} \
    --hidden-dropout ${HD} \
    --untie-embeddings-and-output-weights \
    --position-embedding-type none \
    --hybrid-override-pattern M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M- \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --squared-relu \
    --norm-epsilon 1e-05 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size 1 \
    --num-layers 56 \
    --hidden-size 4480 \
    --ffn-hidden-size 15680 \
    --kv-channels 128 \
    --num-attention-heads 40 \
    --use-distributed-optimizer \
    --use-te \
    --num-workers ${NW} \
    --exit-duration-in-mins 230 \
    --seq-length ${SEQ_LEN} \
    --decoder-seq-length ${DECODER_SEQ_LEN} \
    --max-position-embeddings ${DECODER_SEQ_LEN} \
    --train-full-dataset \
    --lr-warmup-fraction 0.1 \
    --micro-batch-size ${MBZ} \
    --global-batch-size ${BZ} \
    --lr 2e-5 \
    --min-lr 0.0 \
    --lr-decay-style cosine \
    --log-interval ${LI} \
    --eval-iters 0 \
    --eval-interval 100000 \
    --data-path ${DATA_TRAIN} \
    --prompt-path ${SOURCE}/examples/multimodal/manual_prompts.json \
    --save-interval 2000 \
    --save ${FINETUNE_DIR} \
    --load ${FINETUNE_DIR} \
    --pretrained-checkpoint ${CHECKPOINT_DIR} \
    --dataloader-save ${FINETUNE_DIR}/dataloader \
    --split 100,0,0 \
    --clip-grad 1.0 \
    --weight-decay 0.05 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --init-method-std 0.014 \
    --bf16 \
    --eod-mask-loss \
    --patch-dim 16 \
    --img-h 512 \
    --img-w 512 \
    --dataloader-type external \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --language-model-type nemotron5-hybrid-9b \
    ${EXTRA_ARGS} \
    --distributed-timeout-minutes 60 \
    --vision-model-type radio \
    --tokenizer-prompt-format nemotron-h-5p5-reasoning \
    --use-loss-scaling \
    --packing-seq-length ${DECODER_SEQ_LEN} \
    ${SPECIAL_TOKENS} \
    --ckpt-format torch \
    --image-tag-type internvl \
    --eos-id 15 \
    --disable-vision-class-token \
    --use-vision-backbone-fp8-arch \
    --is-hybrid-model \
    --mamba-head-dim 80 \
    --mamba-num-heads 128 \
    --mamba-state-dim 128 \
    --use-loss-scaling \
"

export WANDB_ENTITY=$WANDB_ENTITY  # Not passed in via command line args, only env vars
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

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
    ${SOURCE}/examples/multimodal/train.py ${OPTIONS}"

    echo -e "Debugging with options: $OPTIONS\n"
    eval "$DEBUG_CMD"

elif [[ $INTERACTIVE -eq 1 ]]; then
    torchrun --nproc_per_node ${NUM_GPU} ${CODE_DIR}/examples/multimodal/train.py ${OPTIONS}

else
    run_cmd="python -u ${CODE_DIR}/examples/multimodal/train.py ${OPTIONS}"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "${run_cmd}"
        exit 0
    fi

    DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

    # Need TRITON_CACHE_DIR expanded inside srun b/c sbatch runs on node 0
    srun -l --verbose \
    --container-image /lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/containers/pytorch25.06-moe-avlm-editable-energon.sqsh \
    --container-mounts "/lustre,/home" \
    --output=${LOGS_DIR}/%x_%j_$DATETIME.log \
    sh -c "export TRITON_CACHE_DIR=/tmp/triton_cache_\${SLURM_NODEID}; echo ${run_cmd}; ${run_cmd}"

    set +x
fi
