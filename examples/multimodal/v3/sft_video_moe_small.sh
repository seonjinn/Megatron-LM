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
#SBATCH --job-name=sft_video_moe_small_0917

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MSC_CONFIG="/lustre/fsw/portfolios/llmservice/users/matthieul/msc_config/msc_config.yaml"

export UB_TIMEOUT=720
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN
export TORCHINDUCTOR_WORKER_START=fork

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

USER=$SLURM_JOB_USER

# Auto-detect batch or interactive mode.
which srun
BATCH=$((1-$?))

DEBUG=0
USE_TILING=1
USE_DYNAMIC_RES=0
USE_FP8=0
USE_CP=1


# Remember to update model and job name if running in batch mode!!
if [[ $BATCH -eq 0 ]]; then
    DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
    MODEL_NAME="interactive_sft_video_moe_small_${DATETIME}"
    SPECIAL_TOKENS="--special-tokens <image> <img> </img> <quad> </quad> <ref> </ref> <box> </box>"
    DEBUG=1
else
    MODEL_NAME="sft_video_moe_small_0924"
    SPECIAL_TOKENS="--special-tokens \<image\> \<img\> \</img\> \<quad\> \</quad\> \<ref\> \</ref\> \<box\> \</box\>"
fi

WORKSPACE="/lustre/fsw/portfolios/llmservice/users/${USER}/workspace"
SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

TP=2
EP=4
CHECKPOINT_DIR="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/output/pretrain_moe_small_0928/checkpoints"

# New tokenizer.
TOKENIZER_MODEL="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/hf-transformers/hub/models--nvidia--NVIDIA-Nemotron-Nano-31B-A3-v3/snapshots/22c64afb03eb7c27c5d1fa30042b68a1eed53f33/"
TOKENIZER_PROMPT_FORMAT="nemotron6-moe"

# Old tokenizer.
#TOKENIZER_MODEL="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/nemotron6-moe-tokenizer/"
#TOKENIZER_PROMPT_FORMAT="nemotron6-moe-old"

DATA_TRAIN="/lustre/fsw/portfolios/llmservice/users/matthieul/eagle_recipe_online_packing/final_recipe/25pct.13.51.25pct.txt.multi.img.video.yaml"

if [[ $DEBUG -eq 1 ]]; then
    MBZ=1
    BZ=4
    NW=0
    AD=0.0
    HD=0.0
    LI=1

    EXTRA_ARGS=""

    NUM_GPU=8
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
DECODER_SEQ_LEN=49152
VIDEO_MAX_NUM_FRAMES=128     # Values > 0 enable video max num frames.

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

    if [[ $BATCH -eq 0 ]]; then
        IMAGE_BREAK_TOKEN="--image-break-token <image_break>"
        SPECIAL_TOKENS+=" <image_break>"
    else
        IMAGE_BREAK_TOKEN="--image-break-token \<image_break\>"
        SPECIAL_TOKENS+=" \<image_break\>"
    fi
    EXTRA_ARGS+=" ${IMAGE_BREAK_TOKEN} --dynamic-resolution --dynamic-resolution-min-patches 1024 --conv-merging --allow-missing-conv-merge-checkpoint"
fi

if [[ $USE_CP -eq 1 ]]; then
    EXTRA_ARGS+=" --context-parallel-size 2 --sequence-parallel "
fi

EXTRA_ARGS+=" --packing-buffer-size 128 --packing-seq-length ${DECODER_SEQ_LEN} --packing-knapsack-algorithm balanced_greedy_knapsack "
# LM (Mamba block) recompute
#EXTRA_ARGS+=" --recompute-granularity full --recompute-method block --recompute-num-layers 52 "
EXTRA_ARGS+=" --recompute-granularity selective --recompute-modules core_attn moe_act layernorm mlp moe "
# Vision (GPT block) recompute
EXTRA_ARGS+=" --recompute-vision --recompute-method-vision block --recompute-granularity-vision full --recompute-vision-num-layers 32 "

EXTRA_ARGS+=" --video-min-num-frames 8 --video-max-num-frames ${VIDEO_MAX_NUM_FRAMES} "

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
    --moe-token-dispatcher-type flex \
    --moe-enable-deepep \
    --moe-shared-expert-overlap \
    --enable-experimental \
    --moe-permute-fusion \
    --use-fused-weighted-squared-relu \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl native \
    --moe-router-score-function sigmoid \
    --moe-grouped-gemm \
    --num-experts 128 \
    --moe-router-topk 6 \
    --moe-aux-loss-coeff 1e-4 \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-shared-expert-intermediate-size 3712 \
    --attention-backend flash \
    --is-hybrid-model \
    --mamba-num-heads 64 \
    --mamba-head-dim 64 \
    --hybrid-override-pattern MEM*EMEM*EMEM*EMEM*EMEME \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --tiktoken-pattern v2 \
    --distributed-timeout-minutes 20 \
    --use-mcore-models \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --init-method-std 0.0173 \
    --position-embedding-type none \
    --squared-relu \
    --num-layers 24 \
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
    --lr-wsd-decay-samples 10000 \
    --lr-wsd-decay-style minus_sqrt \
    --override-opt_param-scheduler \
    --lr 2e-4 \
    --min-lr 2e-6 \
    --weight-decay 0.05 \
    --clip-grad 1.0 \
    --lr-decay-style WSD \
    --log-interval ${LI} \
    --eval-iters 0 \
    --eval-interval 99999999999 \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --tokenizer-prompt-format ${TOKENIZER_PROMPT_FORMAT} \
    --load ${FINETUNE_DIR} \
    --save ${FINETUNE_DIR} \
    --dataloader-save ${FINETUNE_DIR}/dataloader \
    --save-interval 2000 \
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
    --adam-beta2 0.95 \
    --use-distributed-optimizer \
    --ddp-num-buckets 8 \
    --ddp-pad-buckets-for-high-nccl-busbw \
    --manual-gc \
    --num-workers ${NW} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --sequence-parallel \
    --allow-large-videos \
"

# --overlap-param-gather \
# --overlap-grad-reduce \
# --pretrained-checkpoint ${CHECKPOINT_DIR} \

#     --disable-gloo-process-groups \
# --tp-comm-overlap \
# --sequence-parallel \
#     --pretrained-checkpoint ${CHECKPOINT_DIR} \

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
