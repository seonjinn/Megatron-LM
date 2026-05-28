#!/bin/bash

#SBATCH -A llmservice_nemo_mlops
# DFW: batch
# OCI-IAD: batch_block1,batch_block3,batch_block4,backfill_block1,backfill_block2,backfill_block3,backfill_block4
# OCI-ORD: grizzly,polar,polar3,polar4
# OCI-NRT: batch_block1,backfill,batch_large,batch_long
#SBATCH -p batch_block1,batch_block3,batch_block4,backfill_block1,backfill_block2,backfill_block3,backfill_block4
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --gpus-per-node=8
#SBATCH --job-name=stage1_nm_5p5_h_9b_cradio_parakeet_vlm2_branch_0825

# Please launch this script from megatron-lm root.
# --------------------------
# 1. Auto-resume setup
# --------------------------
# Grab first/second arguments or fallback to defaults

ORIG_SLURM_JOB_ID=$1
if [[ "$ORIG_SLURM_JOB_ID" == "" ]]; then
  ORIG_SLURM_JOB_ID=$SLURM_JOB_ID
fi
echo "ORIGINAL JOB ID: $ORIG_SLURM_JOB_ID"

PREV_SLURM_JOB_ID=$2
if [[ "$PREV_SLURM_JOB_ID" != "" ]]; then
    # Check the status of the previous job
    PREV_STATUS=$(sacct -j "${PREV_SLURM_JOB_ID}" -P -n -o State | head -n 1)
    if [[ "${PREV_STATUS}" != "TIMEOUT" && "${PREV_STATUS}" != "PREEMPTED" && "${PREV_STATUS}" != "NODE_FAIL" ]]; then
        echo "PREVIOUS JOB ${PREV_SLURM_JOB_ID} FINISHED WITH ${PREV_STATUS}. Not resuming."
        exit 1
    fi
    echo "PREVIOUS JOB ${PREV_SLURM_JOB_ID} FINISHED WITH ${PREV_STATUS}. Resuming..."
fi
echo "PREVIOUS JOB ID: $PREV_SLURM_JOB_ID"

# Schedule a re-run of this script *if* it fails (afternotok)
sbatch \
  --dependency=afternotok:"${SLURM_JOB_ID}" \
  "$0" \
  "${ORIG_SLURM_JOB_ID}" \
  "${SLURM_JOB_ID}"


export CUDA_DEVICE_MAX_CONNECTIONS=1
export MSC_CONFIG="/lustre/fsw/portfolios/llmservice/users/matthieul/msc_config/msc_config.yaml"

USER=$SLURM_JOB_USER

# Auto-detect batch or interactive mode.
which srun
BATCH=$((1-$?))

DEBUG=0
USE_TILING=1
USE_DYNAMIC_RES=0
USE_FP8=0
USE_PRECISION_AWARE_OPTIMIZER=0
USE_CP=0
USE_NEMO=1
USE_PACKING=1
USE_ROTE=0

if [[ $DEBUG -eq 1 ]]; then
    MBZ=1
    BZ=2
    NW=2
    AD=0.0
    HD=0.0
    LI=1
    PBS=128

    NONDETERMINISTIC_ATTN=1

    NUM_GPU=8
else
    MBZ=1
    BZ=2048 #1024
    NW=8
    LI=5
    AD=0.0
    HD=0.0
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
    PBS=4000

    NUM_GPU=8
    TP=4
fi

SEQ_LEN=1024
DECODER_SEQ_LEN=16384

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

# online packing
if [[ $USE_PACKING -eq 1 ]]; then
    EXTRA_ARGS+=" --packing-buffer-size ${PBS} --packing-seq-length ${DECODER_SEQ_LEN} --packing-knapsack-algorithm balanced_greedy_knapsack "
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

    if [[ $BATCH -eq 0 ]]; then
        IMAGE_BREAK_TOKEN="--image-break-token <image_break>"
        SPECIAL_TOKENS+=" <image_break>"
    else
        IMAGE_BREAK_TOKEN="--image-break-token \<image_break\>"
        SPECIAL_TOKENS+=" \<image_break\>"
    fi
    EXTRA_ARGS+=" ${IMAGE_BREAK_TOKEN} --dynamic-resolution --dynamic-resolution-min-patches 1024 --conv-merging "
fi

if [[ $USE_NEMO -eq 1 ]]; then
    # SOUND_MODEL_TYPE="nemo://nvidia/parakeet-tdt-0.6b-v3"
    SOUND_MODEL_TYPE="nemo://nvidia/parakeet-tdt-0.6b-v2"
else
    SOUND_MODEL_TYPE="hf://nithinraok/parakeet-tdt-0.6b-v2-hf"
fi

if [[ $USE_ROTE -eq 1 ]]; then
    EXTRA_ARGS+=" --use-video-rote --use-sound-rote "
fi

EXTRA_ARGS+=" --recompute-granularity full --recompute-method block --recompute-num-layers 62 --recompute-vision --recompute-sound "

# Remember to update model and job name if running in batch mode!!
if [[ $BATCH -eq 0 ]]; then
    DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
    MODEL_NAME="interactive_stage1_nm_5p5_h_9b_cradio_parakeet_${DATETIME}"
    SPECIAL_TOKENS=" --special-tokens <image> <img> </img> <quad> </quad> <ref> </ref> <box> </box> <so_embedding> <so_start> <so_end> "
    DEBUG=1
else
    MODEL_NAME="draco-mcore-nm_5p5_h_9b-cradio-parakeet-nemo-stage1.5-alm-nodes${SLURM_NNODES}-seq${DECODER_SEQ_LEN}-bz${BZ}-tp${TP}-vlm2-branch-1007"
    SPECIAL_TOKENS=" --special-tokens \<image\> \<img\> \</img\> \<quad\> \</quad\> \<ref\> \</ref\> \<box\> \</box\> \<im_embedding\> \<im_start\> \<im_end\> \<video\> \<vi_embedding\> \<vi_start\> \<vi_end\> \<video-sound\> \<vis_embedding\> \<vis_start\> \<vis_end\> \<sound\> \<so_embedding\> \<so_start\> \<so_end\> "
fi

WORKSPACE="/lustre/fsw/portfolios/llmservice/users/${USER}/workspace"

SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR=${WORKSPACE}/tensorboard/${MODEL_NAME}

CHECKPOINT_DIR="/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/avlm_halfduplex/mcore_results/draco-mcore-nm_5p5_h_9b-cradio-parakeet-nemo-stage1-alm-nodes16-seq24576-bz2048-tp4-vlm2-branch-1001/checkpoints" # stage 1

DATA_TRAIN="${SOURCE}/examples/multimodal/avlm/data_config/stage1p5_commercial_alm_blend.yaml"


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
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 56 \
    --recompute-vision \
    --sound-model-type ${SOUND_MODEL_TYPE}  \
    --sound-target-rate 16000 \
    --allow-missing-sound-projection-checkpoint \
    --allow-missing-sound-model-checkpoint \
    --sound-embedding-size 751 \
    --sound-clip-duration 60 \
    --freeze-LM \
    --freeze-ViT \
    --allow-large-videos \
"

export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NONDETERMINISTIC_ATTN}

# Interactive or batch mode
if [[ $BATCH -eq 0 ]]; then
    torchrun --nproc_per_node ${NUM_GPU} examples/multimodal/train.py ${OPTIONS}
else
    run_cmd="cd ${SOURCE}; \
    export PYTHONPATH=\${PYTHONPATH}:${SOURCE}; \
    export HF_HOME=/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/cache/huggingface; \
    export TRANSFORMERS_CACHE=/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/cache/transformers; \
    export HF_DATASETS_CACHE=/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/cache/datasets; \
    export TRITON_CACHE_DIR=/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/cache/triton; \
    mkdir -p /lustre/fsw/portfolios/llmservice/users/ehosseiniasl/cache/huggingface; \
    mkdir -p /lustre/fsw/portfolios/llmservice/users/ehosseiniasl/cache/transformers; \
    mkdir -p /lustre/fsw/portfolios/llmservice/users/ehosseiniasl/cache/datasets; \
    mkdir -p /lustre/fsw/portfolios/llmservice/users/ehosseiniasl/cache/triton; \
    pip install 'megatron-energon[av_decode]@git+https://github.com/NVIDIA/Megatron-Energon.git@feature/cache_to_file'; \
    python -u examples/multimodal/train.py ${OPTIONS}"

    DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

    srun -l --verbose \
    --container-image /lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/containers/megatron-dev-img-05142025-pytorch-dev-te-cd37379-editable-energon-mamba-fix-vlmeval-av.sqsh \
    --container-mounts "/lustre" \
    --output=${LOGS_DIR}/%x_%j_$DATETIME.log \
    sh -c "${run_cmd}"

    set +x
fi
