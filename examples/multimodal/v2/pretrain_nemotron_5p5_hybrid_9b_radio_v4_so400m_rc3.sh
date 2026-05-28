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
#SBATCH --job-name=pretrain_nm_5p5_h_9b_radio_v4_so400m_rc3_tiling_1230

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

# Can be overridden via exports
# NOTE: To run RADIO-H (not So400m), export modified MODEL_NAME, VISION_MODEL_TYPE, CHECKPOINT_DIR
#   export MODEL_NAME=pretrain_nm_5p5_h_9b_radio_v4_h_rc3_tiling_1230
#   export CHECKPOINT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nemotron_5p5_9b_v2_c_radio_v4_h_rc3_tp4
#   export VISION_MODEL_TYPE=radio
#   export USE_FP8=1
MODEL_NAME=${MODEL_NAME:-"pretrain_nm_5p5_h_9b_radio_v4_so400m_rc3_tiling_1230"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nemotron_5p5_9b_v2_c_radio_v4_so400m_rc3_tp4"}
VISION_MODEL_TYPE=${VISION_MODEL_TYPE:-"radio-so400m"}
USE_TILING=${USE_TILING:-1}
USE_DYNAMIC_RES=${USE_DYNAMIC_RES:-0}
USE_VISION_ENCODER_EVAL_MODE=${USE_VISION_ENCODER_EVAL_MODE:-0}
USE_FP8=${USE_FP8:-0}  # Not supported for So400m-RC3
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

TP=4


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
    BZ=1024
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
    EXTRA_ARGS+=" --pixel-shuffle --use-tiling --max-num-tiles 12 --use-thumbnail"
    SEQ_LEN=256
fi

# FP8 padding along the sequence dimension doesn't work with RADIO-v4-So400m-RC3
# Getting the following:
#  4: [rank4]:   File "/usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/module/layernorm_linear.py", line 147, in forward
#  4: [rank4]:     assert_dim_for_fp8_exec(inputmat, weight)
#  4: [rank4]:   File "/usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/utils.py", line 437, in assert_dim_for_fp8_exec
#  4: [rank4]:     assert math.prod(tensor.shape[:-1]) % 8 == 0 and tensor.shape[-1] % 16 == 0, (
#  4: [rank4]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#  4: [rank4]: AssertionError: FP8 execution requires the product of all dimensions except the last to be divisible by 8 and the last dimension to be divisible by 16, but got tensor with dims=[1076, 1152]
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
    --disable-bias-linear \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model /lustre/fsw/portfolios/llmservice/users/ksapra/checkpoints/prunedmodel/first_9b_sft_pruned_v0 \
    --tokenizer-prompt-format nemotron-h-5p5-reasoning \
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
    --lr-warmup-samples 102400 \
    --micro-batch-size ${MBZ} \
    --global-batch-size ${BZ} \
    --lr 2e-4 \
    --min-lr 0.0 \
    --lr-decay-style cosine \
    --log-interval ${LI} \
    --eval-iters 0 \
    --eval-interval 100000 \
    --data-path ${DATA_TRAIN} \
    --prompt-path ${CODE_DIR}/examples/multimodal/manual_prompts.json \
    --save-interval ${SAVE_INTERVAL} \
    --save ${FINETUNE_DIR} \
    --load ${FINETUNE_DIR} \
    --dataloader-save ${FINETUNE_DIR}/dataloader \
    --pretrained-checkpoint ${CHECKPOINT_DIR} \
    --split 100,0,0 \
    --clip-grad 1.0 \
    --weight-decay 1e-2 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --init-method-std 0.02 \
    --bf16 \
    --eod-mask-loss \
    --freeze-ViT \
    --freeze-LM \
    --patch-dim 16 \
    --img-h 512 \
    --img-w 512 \
    --dataloader-type external \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --language-model-type nemotron5-hybrid-9b \
    ${EXTRA_ARGS} \
    --distributed-timeout-minutes 60 \
    --allow-missing-vision-projection-checkpoint \
    --vision-model-type ${VISION_MODEL_TYPE} \
    --use-loss-scaling \
    ${SPECIAL_TOKENS} \
    --ckpt-format torch \
    --image-tag-type internvl \
    --disable-vision-class-token \
    --eos-id 15 \
    --attention-backend flash \
    --is-hybrid-model \
    --mamba-head-dim 80 \
    --mamba-num-heads 128 \
    --mamba-state-dim 128 \
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
