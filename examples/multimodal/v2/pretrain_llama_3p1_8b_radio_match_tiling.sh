#!/bin/bash

#SBATCH -A llmservice_fm_vision
#SBATCH -p batch_block1,backfill,batch_large,batch_long
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton
#SBATCH --nodes=32
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --gpus-per-node=8
#SBATCH --job-name=pretrain_llama_3p1_8b_cradio_v3_match_tiling_0730

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MSC_CONFIG="/lustre/fsw/portfolios/llmservice/users/matthieul/msc_config/msc_config.yaml"

USER=$SLURM_JOB_USER

# Auto-detect batch or interactive mode.
which srun
BATCH=$((1-$?))

DEBUG=0
USE_TILING=0
USE_MATCH_TILING=1
USE_DYNAMIC_RES=0
USE_PP=0
USE_FP8=1

# Remember to update model and job name if running in batch mode!!
if [[ $BATCH -eq 0 ]]; then
    DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
    MODEL_NAME="interactive_pretrain_llama_3p1_8b_cradio_v3_match_tiling_${DATETIME}"
    SPECIAL_TOKENS="--special-tokens <image> <img> </img> <quad> </quad> <ref> </ref> <box> </box>"
    DEBUG=1
else
    MODEL_NAME="pretrain_llama_3p1_8b_cradio_v3_match_tiling_0730"
    SPECIAL_TOKENS="--special-tokens \<image\> \<img\> \</img\> \<quad\> \</quad\> \<ref\> \</ref\> \<box\> \</box\>"
fi

WORKSPACE="/lustre/fsw/portfolios/llmservice/users/${USER}/workspace"
SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

TP=4

CHECKPOINT_DIR="/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/llama_3p1_8b_c-radio-vlm_v1_rc3-no-extra-state"

DATA_TRAIN="${SOURCE}/examples/multimodal/v2/data_config/pretrain_dataset_commercial.yaml"

if [[ $DEBUG -eq 1 ]]; then
    MBZ=1
    BZ=4
    NW=0
    AD=0.0
    HD=0.0
    LI=1

    EXTRA_ARGS="--deterministic-mode --use-cpu-initialization"
    EXTRA_ARGS=""

    NONDETERMINISTIC_ATTN=0
    NONDETERMINISTIC_ATTN=1

    NUM_GPU=8
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
    MBZ=1
    BZ=1024
    NW=8
    AD=0.0
    HD=0.0
    LI=5
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
    NUM_GPU=8
fi

SEQ_LEN=1024
DECODER_SEQ_LEN=16384

if [[ $USE_TILING -eq 1 ]]; then
    EXTRA_ARGS+=" --pixel-shuffle --use-tiling --max-num-tiles 12 --use-thumbnail"
    SEQ_LEN=256
fi

if [[ $USE_MATCH_TILING -eq 1 ]]; then
    SEQ_LEN=12288

    if [[ $BATCH -eq 0 ]]; then
        IMAGE_BREAK_TOKEN="--image-break-token <image_break>"
        SPECIAL_TOKENS+=" <image_break>"
    else
        IMAGE_BREAK_TOKEN="--image-break-token \<image_break\>"
        SPECIAL_TOKENS+=" \<image_break\>"
    fi
    EXTRA_ARGS+=" ${IMAGE_BREAK_TOKEN} --dynamic-resolution --match-tiling-dynamic-resolution --conv-merging --allow-missing-conv-merge-checkpoint"
    EXTRA_ARGS+=" --max-num-tiles 12 --use-thumbnail"
fi

if [[ $USE_PP -eq 1 ]]; then
    EXTRA_ARGS+=" --pipeline-model-parallel-size 1 --encoder-pipeline-model-parallel-size 1"
    NUM_GPU=8
fi

if [[ $USE_FP8 -eq 1 ]]; then
    EXTRA_ARGS+=" --fp8-recipe blockwise --fp8-format e4m3 --first-last-layers-bf16 --num-layers-at-start-in-bf16 1 --num-layers-at-end-in-bf16 1"
    #EXTRA_ARGS+=" --fp8-recipe blockwise --fp8-format e4m3 "
    #EXTRA_ARGS+=" --fp8-recipe blocwise --fp8-format e4m3 --fp8-param-gather "
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
    EXTRA_ARGS+=" ${IMAGE_BREAK_TOKEN} --dynamic-resolution --dynamic-resolution-min-patches 1024 --conv-merging --allow-missing-conv-merge-checkpoint"
fi


OPTIONS=" \
    --use-checkpoint-args \
    --disable-bias-linear \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model /lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/ \
    --transformer-impl transformer_engine \
    --normalization RMSNorm \
    --group-query-attention \
    --num-query-groups 8 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --attention-dropout ${AD} \
    --hidden-dropout ${HD} \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 500000 \
    --use-rope-scaling \
    --swiglu \
    --tensor-model-parallel-size ${TP}  \
    --pipeline-model-parallel-size 1  \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --use-distributed-optimizer \
    --use-te \
    --num-workers ${NW} \
    --exit-duration-in-mins 230 \
    --seq-length ${SEQ_LEN} \
    --decoder-seq-length ${DECODER_SEQ_LEN} \
    --max-position-embeddings 131072 \
    --train-full-dataset \
    --lr-warmup-samples 102400 \
    --micro-batch-size ${MBZ} \
    --global-batch-size ${BZ} \
    --lr 2e-4 \
    --min-lr 0.0 \
    --lr-decay-style cosine \
    --log-interval ${LI} \
    --eval-iters 0 \
    --eval-interval 999999999 \
    --data-path ${DATA_TRAIN} \
    --prompt-path ${SOURCE}/examples/multimodal/manual_prompts.json \
    --save-interval 5000 \
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
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --eod-mask-loss \
    --freeze-ViT \
    --freeze-LM \
    --patch-dim 16 \
    --img-h 512 \
    --img-w 512 \
    --dataloader-type external \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --language-model-type=llama3.1_8b \
    ${EXTRA_ARGS} \
    --distributed-timeout-minutes 60 \
    --allow-missing-vision-projection-checkpoint \
    --vision-model-type radio \
    --tokenizer-prompt-format llama3p1 \
    --use-loss-scaling \
    ${SPECIAL_TOKENS} \
    --ckpt-format torch \
    --image-tag-type internvl \
    --force-system-message \
    --disable-vision-class-token \
    --inference-max-seq-length ${DECODER_SEQ_LEN} \
    --use-area-weighted-aspect-ratio \
    --use-vision-backbone-fp8-arch \
"

export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NONDETERMINISTIC_ATTN}

# Interactive or batch mode
if [[ $BATCH -eq 0 ]]; then
    torchrun --nproc_per_node ${NUM_GPU} examples/multimodal/train.py ${OPTIONS}
else
    run_cmd="python -u ${SOURCE}/examples/multimodal/train.py ${OPTIONS}"

    DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

    srun -l --verbose \
    --container-image /lustre/fsw/portfolios/llmservice/users/amalasanjayd/containers/megatron-lm/megatron-dev-0806.sqsh \
    --container-mounts "/lustre" \
    --output=${LOGS_DIR}/%x_%j_$DATETIME.log \
    sh -c "echo ${run_cmd}; ${run_cmd}"

    set +x
fi
