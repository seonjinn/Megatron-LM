#!/bin/bash

#SBATCH -A llmservice_fm_vision
#SBATCH -p batch_block1,batch_large,batch_long
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton
#SBATCH --nodes=32
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --gpus-per-node=8
#SBATCH --job-name=pretrain_moe_rl_llm_vision_eval_mode_radio_v4_so400m_rc3_1230

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
USE_TILING=0
USE_DYNAMIC_RES=1
USE_IMAGE_BREAK=0   # Only used if USE_DYNAMIC_RES is 1.
USE_CONV_MERGE=0    # Only used if USE_DYNAMIC_RES is 1.
USE_FP8=0  # Not supported with current RC3 checkpoint
USE_VISION_ENCODER_EVAL_MODE=1

# Can be overridden via exports
# NOTE: Debug doesn't work with >TP8 currently, but leaving it here for completeness
# NOTE: To run RADIO-H (not So400m), export modified MODEL_NAME, VISION_MODEL_TYPE, CHECKPOINT_DIR
#   export MODEL_NAME=pretrain_moe_rl_llm_vision_eval_mode_radio_v4_h_rc3_1230
#   export CHECKPOINT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nano-v3-rc2-12-03-25-7-15-pm-c-radio_v4-h-rc3-tp2-ep32
#   export VISION_MODEL_TYPE=radio
MODEL_NAME=${MODEL_NAME:-"pretrain_moe_rl_llm_vision_eval_mode_radio_v4_so400m_rc3_1230"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nano-v3-rc2-12-03-25-7-15-pm-c-radio_v4-so400m-rc3-tp2-ep32"}
VISION_MODEL_TYPE=${VISION_MODEL_TYPE:-"radio-so400m"}
DRY_RUN=${DRY_RUN:-0}  # Prints launch command and exits
DEBUG=${DEBUG:-0}  # Sets DEBUG_RANK, requires interactive session
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
BASE_LR=${BASE_LR:-"1e-3"}
MIN_LR=${MIN_LR:-"1e-5"}
DYNAMIC_RES_DATA_AUG=${DYNAMIC_RES_DATA_AUG:-1}

if [[ $DEBUG -eq 1 ]]; then
    # Debugging launches the debugger, so it's a non-interactive launch in terms of escaping "<>"
    INTERACTIVE=0

    # Append _debug to MODEL_NAME and WORKSPACE to more easily delete all debug runs
    MODEL_NAME="${MODEL_NAME}_debug"
    WORKSPACE="/lustre/fsw/portfolios/llmservice/users/${USER}/workspace/debug"

else
    # Auto-detect interactive mode (if srun is not defined, we're interactive)
    INTERACTIVE=$(which srun >/dev/null 2>&1 && echo 0 || echo 1)

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
if [[ $DEBUG -eq 0 ]]; then
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

# New tokenizer 10/20.
TOKENIZER_MODEL="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/hf-transformers/hub/models--nvidia--Nemotron-Nano-3-30B-A3.5B-dev-1016/snapshots/bb271274159f07461e919379311e32802e5ec36b/"
TOKENIZER_PROMPT_FORMAT="nemotron6-moe"

DATA_TRAIN="${CODE_DIR}/examples/multimodal/v2/data_config/pretrain_dataset_commercial_sft_extended.yaml"

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
    BZ=512
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
    if [[ $DYNAMIC_RES_DATA_AUG -eq 1 ]]; then
        EXTRA_ARGS+=" --apply-data-augment"
    fi
fi

if [[ $USE_VISION_ENCODER_EVAL_MODE -eq 1 ]]; then
    EXTRA_ARGS+=" --radio-force-eval-mode"  # Entire vision encoder in eval mode (eval CPE, no dropout)
fi


OPTIONS=" \
    --use-checkpoint-args \
    --transformer-impl transformer_engine \
    --use-te \
    --data-path ${DATA_TRAIN} \
    --freeze-ViT \
    --freeze-LM \
    --patch-dim 16 \
    --img-h 512 \
    --img-w 512 \
    --dataloader-type external \
    --language-model-type nemotron6-moe \
    ${EXTRA_ARGS} \
    --allow-missing-vision-projection-checkpoint \
    --vision-model-type ${VISION_MODEL_TYPE} \
    --use-loss-scaling \
    ${SPECIAL_TOKENS} \
    --disable-vision-class-token \
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
    --moe-aux-loss-coeff 1e-6 \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-shared-expert-intermediate-size 3712 \
    --attention-backend flash \
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
    --init-method-std 0.02 \
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
    --lr ${BASE_LR} \
    --min-lr ${MIN_LR} \
    --lr-warmup-fraction 0.1 \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --lr-decay-style cosine \
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
    --save-interval ${SAVE_INTERVAL} \
    --ckpt-format torch \
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
    --ddp-num-buckets 8 \
    --ddp-pad-buckets-for-high-nccl-busbw \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --manual-gc \
    --num-workers ${NW} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --sequence-parallel \
    --class-token-len 10 \
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
    ${CODE_DIR}/examples/multimodal/train.py ${OPTIONS}"

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
    --output=${LOGS_DIR}/%x_%j_srun_$DATETIME.log \
    sh -c "export TRITON_CACHE_DIR=/tmp/triton_cache_\${SLURM_NODEID}; echo ${run_cmd}; ${run_cmd}"

    set +x
fi
