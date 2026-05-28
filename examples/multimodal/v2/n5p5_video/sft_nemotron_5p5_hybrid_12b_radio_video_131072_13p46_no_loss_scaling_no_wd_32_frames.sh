#!/bin/bash

#SBATCH -A llmservice_fm_vision
#SBATCH -p batch_block1,batch_large,batch_long
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton
#SBATCH --nodes=64
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --job-name=sft_nm_5p5_h_12b_6k_cradio_vlm_v1_rc3_video_13p46_trimmed_131072_no_loss_scaling_no_wd_32_frames_0826

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MSC_CONFIG="/lustre/fsw/portfolios/llmservice/users/matthieul/msc_config/msc_config.yaml"
export TOKENIZERS_PARALLELISM="false"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

USER=$SLURM_JOB_USER

# Auto-detect batch or interactive mode.
which srun
BATCH=$((1-$?))

DEBUG=0
USE_TILING=1
USE_PACKING=0
USE_ONLINE_PACKING=1
USE_DYNAMIC_RES=0
USE_FP8=1
USE_PRECISION_AWARE_OPTIMIZER=1

# Video options.
SEQ_LEN=256     # Vision encoder per image.
DECODER_SEQ_LEN=131072
VIDEO_MAX_NUM_FRAMES=32     # Values > 0 enable video max num frames.
USE_CP=1

# Remember to update model and job name if running in batch mode!!
if [[ $BATCH -eq 0 ]]; then
    DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
    MODEL_NAME="interactive_sft_nemotron_5p5_hybrid_12b_cradio_vlm_v1_rc3_video_${DATETIME}"
    SPECIAL_TOKENS="--special-tokens <image> <img> </img> <quad> </quad> <ref> </ref> <box> </box>"
    DEBUG=1
else
    MODEL_NAME="sft_nm_5p5_h_12b_6k_cradio_vlm_v1_rc3_video_13p46_trimmed_131072_no_loss_scaling_no_wd_32_frames_0826"
    SPECIAL_TOKENS="--special-tokens \<image\> \<img\> \</img\> \<quad\> \</quad\> \<ref\> \</ref\> \<box\> \</box\>"
fi

WORKSPACE="/lustre/fsw/portfolios/llmservice/users/${USER}/workspace"
SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

# Ensure output directories exist
mkdir -p "${FINETUNE_DIR}" "${LOGS_DIR}" "${TENSORBOARD_DIR}"

# Snapshot the source code into the OUTPUT directory on first run, and always run from the snapshot thereafter
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

DATA_TRAIN="/lustre/fsw/portfolios/llmservice/users/matthieul/eagle_recipe_online_packing/final_recipe/merged_13p46_lc_video_v1_v2_1.0_videoMult_1.0_trimmed_0827.yaml"

if [[ $DEBUG -eq 1 ]]; then
    MBZ=1
    BZ=1
    NW=0
    AD=0.0
    HD=0.0
    LI=1

    NONDETERMINISTIC_ATTN=1
    # CUDA_VISIBLE_DEVICES=0,1,2,3
    NUM_GPU=8
    PACKING_BUFFER_SIZE=1000
else
    MBZ=1
    BZ=16
    NW=8
    AD=0.0
    HD=0.0
    LI=5
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
    NUM_GPU=8
    PACKING_BUFFER_SIZE=4034
fi

# TODO
# - Updated proper checkpoint from Amala.
# - Proper TP=4 checkpoint for debugging.
if [[ $DEBUG -eq 1 ]]; then
    TP=2
    DECODER_SEQ_LEN=16384
else
    TP=8
    CHECKPOINT_DIR="/lustre/fsw/portfolios/llmservice/users/amalasanjayd/workspace/output/pretrain_nm_5p5_h_12b0804_v1340_0804/checkpoints"
    EXTRA_ARGS=" --pretrained-checkpoint ${CHECKPOINT_DIR} "
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

if [[ $USE_PACKING -eq 1 ]]; then
    EXTRA_ARGS+=" --packing-seq-length ${DECODER_SEQ_LEN} "
fi

if [[ $USE_ONLINE_PACKING -eq 1 ]]; then
    EXTRA_ARGS+=" --packing-buffer-size ${PACKING_BUFFER_SIZE} --packing-seq-length ${DECODER_SEQ_LEN} --packing-knapsack-algorithm balanced_greedy_knapsack "
fi

if [[ $USE_PRECISION_AWARE_OPTIMIZER -eq 1 ]]; then
    EXTRA_ARGS+=" --use-precision-aware-optimizer --main-grads-dtype bf16 --main-params-dtype fp16 --exp-avg-dtype fp16 --exp-avg-sq-dtype fp16 "
fi

if [[ $USE_CP -eq 1 ]]; then
    EXTRA_ARGS+=" --context-parallel-size 4 --sequence-parallel "
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

EXTRA_ARGS+=" --recompute-granularity full --recompute-method block --recompute-num-layers 16 --recompute-vision --recompute-vision-num-layers 16 "
EXTRA_ARGS+=" --video-min-num-frames 8 --video-max-num-frames ${VIDEO_MAX_NUM_FRAMES} "

OPTIONS=" \
    --use-checkpoint-args \
    --disable-bias-linear \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model /lustre/fsw/portfolios/llmservice/users/amalasanjayd/checkpoints/nano-v2-sft-lr5e-6-128k-nollama-thinkfix-ep2/checkpoints/nano-v2-sft-lr5e-6-128k-nollama-thinkfix-ep2/iter_0006000/ \
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
    --lr 2e-5 \
    --min-lr 0.0 \
    --lr-decay-style cosine \
    --log-interval ${LI} \
    --eval-iters 0 \
    --eval-interval 100000 \
    --data-path ${DATA_TRAIN} \
    --prompt-path ${CODE_DIR}/examples/multimodal/manual_prompts.json \
    --save-interval 2000 \
    --save ${FINETUNE_DIR} \
    --load ${FINETUNE_DIR} \
    --dataloader-save ${FINETUNE_DIR}/dataloader \
    --split 100,0,0 \
    --clip-grad 1.0 \
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
    --language-model-type nemotron5-hybrid-12b \
    ${EXTRA_ARGS} \
    --distributed-timeout-minutes 60 \
    --vision-model-type radio \
    --tokenizer-prompt-format nemotron-h-5p5-reasoning \
    --packing-seq-length ${DECODER_SEQ_LEN} \
    ${SPECIAL_TOKENS} \
    --ckpt-format torch \
    --image-tag-type internvl \
    --eos-id 12 \
    --disable-vision-class-token \
    --use-vision-backbone-fp8-arch \
    --is-hybrid-model \
    --mamba-head-dim 80 \
    --mamba-num-heads 128 \
    --mamba-state-dim 128 \
    --allow-large-videos \
    --force-system-message \
"

export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NONDETERMINISTIC_ATTN}

# Interactive or batch mode
if [[ $BATCH -eq 0 ]]; then
    cd "${CODE_DIR}"
    torchrun --nproc_per_node ${NUM_GPU} examples/multimodal/train.py ${OPTIONS}
else
    run_cmd="cd ${CODE_DIR}; python -u examples/multimodal/train.py ${OPTIONS}"

    DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

    srun -l --verbose \
    --container-image /lustre/fsw/portfolios/llmservice/users/matthieul/docker/megatron-dev-img-05142025-pytorch-dev-te-cd37379-editable-energon-mamba-fix-vlmeval-pad-conv.sqsh \
    --container-mounts "/lustre" \
    --output=${LOGS_DIR}/%x_%j_$DATETIME.log \
    sh -c "${run_cmd}"

    set +x
fi
