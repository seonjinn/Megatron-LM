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
#SBATCH --job-name=pretrain_nm_5p5_h_12b_1007

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MSC_CONFIG="/lustre/fsw/portfolios/llmservice/users/matthieul/msc_config/msc_config.yaml"

USER=$SLURM_JOB_USER

# Auto-detect batch or interactive mode.
which srun
BATCH=$((1-$?))

DEBUG=0


# Remember to update model and job name if running in batch mode!!
if [[ $BATCH -eq 0 ]]; then
    DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
    MODEL_NAME="interactive_pretrain_nemotron_5p5_hybrid_12b_cradio_${DATETIME}"
    SPECIAL_TOKENS="--special-tokens <image> <img> </img> <quad> </quad> <ref> </ref> <box> </box>"
    DEBUG=1
else
    MODEL_NAME="pretrain_nm_5p5_h_12b_1007"
    SPECIAL_TOKENS="--special-tokens \<image\> \<img\> \</img\> \<quad\> \</quad\> \<ref\> \</ref\> \<box\> \</box\>"
fi

WORKSPACE="/lustre/fsw/portfolios/llmservice/users/${USER}/workspace"
SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

TP=8
CHECKPOINT_DIR="/lustre/fsw/portfolios/llmservice/users/amalasanjayd/checkpoints/nemotron_5p5_12b_0804/mcore_tp8_vlm"
# TP=4
# CHECKPOINT_DIR="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/n5p5_12b_sft_0718_cradio_vlm_v1_rc3_tp4"

# Run data_scripts/convert.py to produce the pretrain dataset.
# Pretrain dataset #2 (5% SFT data). GA model was trained with this dataset.
# Remove commit 774ca92cb425241b6460335db2471c4a9290b9e2 to reproduce this dataset.
DATA_TRAIN="${SOURCE}/examples/multimodal/v2/data_config/pretrain_dataset_commercial_sft_extended.yaml"
# Pretrain dataset #2 with fixes (account for repetitions and remove raw[json] errors).
# DATA_TRAIN="${SOURCE}/examples/multimodal/v2/data_config/pretrain_dataset_commercial_sft_extended2.yaml"

if [[ $DEBUG -eq 1 ]]; then
    MBZ=1
    BZ=8
    NW=0
    AD=0.0
    HD=0.0
    LI=1

    EXTRA_ARGS="--deterministic-mode --use-cpu-initialization"
    EXTRA_ARGS=""

    NUM_GPU=8

    SAVE_INTERVAL=10000
else
    MBZ=1
    BZ=1024
    NW=8
    AD=0.0
    HD=0.0
    LI=5
    EXTRA_ARGS=""
    NUM_GPU=8

    SAVE_INTERVAL=5000
fi

SEQ_LEN=1024
DECODER_SEQ_LEN=16384

USE_TILING=1
if [[ $USE_TILING -eq 1 ]]; then
    EXTRA_ARGS+=" --pixel-shuffle --use-tiling --max-num-tiles 12 --use-thumbnail"
    SEQ_LEN=256
fi

USE_FP8=1
if [[ $USE_FP8 -eq 1 ]]; then
    # Recipe 1: More accurate but not the fastest.
    EXTRA_ARGS+=" --fp8-recipe blockwise --fp8-format e4m3 --first-last-layers-bf16 --num-layers-at-start-in-bf16 1 --num-layers-at-end-in-bf16 1"
    # Recipes 2 and 3: Faster but metrics can become a bit noisier. Still the difference to bf16 should be small < 1%.
    #EXTRA_ARGS+=" --fp8-recipe blockwise --fp8-format e4m3 "
    #EXTRA_ARGS+=" --fp8-recipe blocwise --fp8-format e4m3 --fp8-param-gather "
    EXTRA_ARGS+=" --use-vision-backbone-fp8-arch "
fi

USE_DYNAMIC_RES=0
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

# --tokenizer-model /lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/models--nvidia--Nemotron-H-8B-Base-8K/snapshots/281935db305672111f043428fe4982969876613c/ \

OPTIONS=" \
    --use-checkpoint-args \
    --disable-bias-linear \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model /lustre/fsw/portfolios/llmservice/users/amalasanjayd/checkpoints/nano-v2-sft-lr5e-6-128k-nollama-thinkfix-ep2/checkpoints/nano-v2-sft-lr5e-6-128k-nollama-thinkfix-ep2/iter_0006000/ \
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
    --hybrid-override-pattern M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M- \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --squared-relu \
    --norm-epsilon 1e-05 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size 1 \
    --num-layers 62 \
    --hidden-size 5120 \
    --ffn-hidden-size 20480 \
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
    --lr 2e-4 \
    --min-lr 0.0 \
    --lr-decay-style cosine \
    --log-interval ${LI} \
    --eval-iters 0 \
    --eval-interval 100000 \
    --data-path ${DATA_TRAIN} \
    --prompt-path ${SOURCE}/examples/multimodal/manual_prompts.json \
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
    --language-model-type nemotron5-hybrid-12b \
    ${EXTRA_ARGS} \
    --distributed-timeout-minutes 60 \
    --allow-missing-vision-projection-checkpoint \
    --vision-model-type radio \
    --use-loss-scaling \
    ${SPECIAL_TOKENS} \
    --ckpt-format torch \
    --image-tag-type internvl \
    --disable-vision-class-token \
    --eos-id 15 \
    --attention-backend flash \
    --use-vision-backbone-fp8-arch \
    --is-hybrid-model \
    --mamba-head-dim 80 \
    --mamba-num-heads 128 \
    --mamba-state-dim 128 \
"

export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

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
