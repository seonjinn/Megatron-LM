#!/bin/bash

#SBATCH -A llmservice_fm_vision
#SBATCH -p batch
#SBATCH -t 00:30:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --gpus-per-node=8
#SBATCH --job-name=test_pretrain_super_a1_svg_newcontainer

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MSC_CONFIG="/scratch/fsw/portfolios/llmservice/users/trintamaki/msc_config/msc_config.yaml"

export UB_TIMEOUT=720
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
USE_TILING=0
USE_DYNAMIC_RES=1
USE_IMAGE_BREAK=0   # Only used if USE_DYNAMIC_RES is 1.
USE_CONV_MERGE=0    # Only used if USE_DYNAMIC_RES is 1.
USE_FP8=0
USE_VISION_ENCODER_EVAL_MODE=1
USE_MTP=1
USE_PACKING=1
USE_BUCKETING=0

# Conv3d video defaults (w/ allowed overrides via env vars)
VIDEO_MAX_NUM_FRAMES=${VIDEO_MAX_NUM_FRAMES:-32}  # Using conv3d w/ 2-frame compression
SEPARATE_VIDEO_EMBEDDER=${SEPARATE_VIDEO_EMBEDDER:-1}
VIDEO_MAINTAIN_ASPECT_RATIO=${VIDEO_MAINTAIN_ASPECT_RATIO:-1}

# Early exit for testing
EARLY_EXIT_ITERS=5

if [[ $DEBUG -eq 1 ]]; then
    MBZ=1
    BZ=8
    NW=0
    AD=0.0
    HD=0.0
    LI=1

    EXTRA_ARGS=""

    NUM_GPU=8
    PBS=128
else
    MBZ=1
    BZ=16
    NW=2
    AD=0.0
    HD=0.0
    LI=1
    EXTRA_ARGS=""
    NUM_GPU=8
    PBS=128
fi

SEQ_LEN=256
DECODER_SEQ_LEN=16384

# Remember to update model and job name if running in batch mode!!
if [[ $BATCH -eq 0 ]]; then
    DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
    MODEL_NAME="interactive_test_pretrain_super_a1_svg_newcontainer_${DATETIME}"
    SPECIAL_TOKENS=" --special-tokens <image> <img> </img> <quad> </quad> <ref> </ref> <box> </box> <so_embedding> <so_start> <so_end> "
    DEBUG=1
else
    MODEL_NAME="test_pretrain_super_a1_svg_newcontainer"
    SPECIAL_TOKENS=" --special-tokens \<image\> \<img\> \</img\> \<quad\> \</quad\> \<ref\> \</ref\> \<box\> \</box\> \<so_embedding\> \<so_start\> \<so_end\> "
fi

WANDB_API_KEY=""
WANDB_PROJECT=""
WANDB_ENTITY=""
WANDB_NAME=${MODEL_NAME}

WORKSPACE="/lustre/fsw/portfolios/llmservice/users/${USER}/workspace"
SOURCE=`pwd`
# Clean up double slashes in SOURCE path
SOURCE=$(echo "$SOURCE" | sed 's|^//|/|')
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"
WANDB_DIR="${OUTPUT}/wandb"

# Ensure output directories exist
mkdir -p "${FINETUNE_DIR}" "${LOGS_DIR}" "${TENSORBOARD_DIR}" "${WANDB_DIR}"

CODE_DIR="${SOURCE}"

# Reduced parallelism for 2-node test (16 GPUs): TP=2, EP=8, DP_data=1
TP=2
EP=8

# Starting checkpoint: SVG SFT stage output (sft_super_final_ckpt_conv3d_radiov4_1377_0402)
CHECKPOINT_DIR="/lustre/fsw/portfolios/llmservice/users/cmccarthy/workspace/output/sft_super_final_ckpt_conv3d_radiov4_1377_0402/checkpoints"

TOKENIZER_MODEL="/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/workspace/nemotron_3_nano_30b_a3b_tokenizer"
TOKENIZER_PROMPT_FORMAT="nemotron6-moe"

DATA_TRAIN="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/megatron-lm/a1_yamls/audio_stage1.yaml"

SOUND_MODEL_TYPE="nemo://nvidia/parakeet-tdt-0.6b-v2"

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

if [[ $USE_MTP -eq 1 ]]; then
    EXTRA_ARGS+=" --mtp-spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec"
    EXTRA_ARGS+=" --mtp-num-layers 2"
    EXTRA_ARGS+=" --mtp-hybrid-override-pattern *E"
    EXTRA_ARGS+=" --mtp-loss-scaling-factor 1.5e-4"
    EXTRA_ARGS+=" --mtp-use-repeated-layer"
    EXTRA_ARGS+=" --keep-mtp-spec-in-bf16"
else
    EXTRA_ARGS+=" --disable-mtp"
fi

EXTRA_ARGS+=" --early-exit-iters ${EARLY_EXIT_ITERS} "

if [[ "$VIDEO_MAINTAIN_ASPECT_RATIO" -eq 1 ]]; then
    EXTRA_ARGS+=" --video-maintain-aspect-ratio "
fi

if [[ $SEPARATE_VIDEO_EMBEDDER -eq 1 ]]; then
    EXTRA_ARGS+=" --separate-video-embedder "
fi

# online packing & bucketing
if [[ $USE_PACKING -eq 1 ]]; then
    EXTRA_ARGS+=" --packing-buffer-size ${PBS} --packing-seq-length ${DECODER_SEQ_LEN}"
    if [[ $USE_BUCKETING -eq 1 ]]; then
        EXTRA_ARGS+=" --packing-knapsack-algorithm bucketing_greedy_knapsack "
    else
        EXTRA_ARGS+=" --packing-knapsack-algorithm balanced_greedy_knapsack "
    fi
fi

# LM (Mamba block) recompute
EXTRA_ARGS+=" --recompute-granularity selective --recompute-modules core_attn mlp layernorm moe_act moe "
# Vision (GPT block) recompute
EXTRA_ARGS+=" --recompute-vision --recompute-method-vision block --recompute-granularity-vision full --recompute-vision-num-layers 32 "
# Sound model recompute
EXTRA_ARGS+=" --recompute-sound "

OPTIONS=" \
    --use-checkpoint-args \
    --transformer-impl transformer_engine \
    --use-te \
    --data-path ${DATA_TRAIN} \
    --train-full-dataset \
    --patch-dim 16 \
    --img-h 512 \
    --img-w 512 \
    --dataloader-type external \
    --language-model-type nemotron6-super \
    ${EXTRA_ARGS} \
    --vision-model-type radio \
    --class-token-len 10 \
    --use-loss-scaling \
    ${SPECIAL_TOKENS} \
    --disable-vision-class-token \
    --prompt-path ${CODE_DIR}/examples/multimodal/manual_prompts.json \
    --eod-mask-loss \
    --image-tag-type internvl \
    --moe-token-dispatcher-type alltoall \
    --enable-experimental \
    --moe-permute-fusion \
    --use-fused-weighted-squared-relu \
    --moe-router-score-function sigmoid \
    --moe-grouped-gemm \
    --num-experts 512 \
    --moe-router-topk 22 \
    --moe-aux-loss-coeff 1e-8 \
    --moe-router-topk-scaling-factor 5.0 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-shared-expert-intermediate-size 5376 \
    --moe-latent-size 1024 \
    --attention-backend flash \
    --is-hybrid-model \
    --mamba-num-heads 128 \
    --hybrid-override-pattern MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --tiktoken-pattern v2 \
    --distributed-timeout-minutes 20 \
    --use-mcore-models \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --init-method-std 0.014 \
    --position-embedding-type none \
    --squared-relu \
    --num-layers 88 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 2 \
    --ffn-hidden-size 2688 \
    --kv-channels 128 \
    --normalization RMSNorm \
    --attention-dropout ${AD} \
    --hidden-dropout ${HD} \
    --exit-duration-in-mins 25 \
    --tensor-model-parallel-size ${TP} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --seq-length ${SEQ_LEN} \
    --decoder-seq-length ${DECODER_SEQ_LEN} \
    --max-position-embeddings ${DECODER_SEQ_LEN} \
    --micro-batch-size ${MBZ} \
    --global-batch-size ${BZ} \
    --lr-warmup-fraction 0.1 \
    --lr 1e-3 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
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
    --adam-beta2 0.999 \
    --use-distributed-optimizer \
    --num-workers ${NW} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --sequence-parallel \
    --allow-large-videos \
    --sound-model-type ${SOUND_MODEL_TYPE} \
    --sound-target-rate 16000 \
    --sound-embedding-size 751 \
    --allow-missing-sound-projection-checkpoint \
    --allow-missing-sound-model-checkpoint \
    --freeze-LM \
    --freeze-ViT \
    --freeze-sound-model \
    --video-target-num-patches 1024 \
    --video-max-num-frames ${VIDEO_MAX_NUM_FRAMES} \
    --video-temporal-patch-size 2 \
    --video-prompt-version 2 \
"

export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

# Interactive or batch mode
if [[ $BATCH -eq 0 ]]; then
    cd ${CODE_DIR}
    torchrun --nproc_per_node ${NUM_GPU} examples/multimodal/train.py ${OPTIONS}
else
    run_cmd="python -u ${CODE_DIR}/examples/multimodal/train.py ${OPTIONS}"

    DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

    srun -l --verbose \
    --container-image /lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/containers/pytorch25.11-moe-avlm-editable-energon-super.sqsh \
    --container-mounts "/lustre,/scratch" \
    --output=${LOGS_DIR}/%x_%j_srun_$DATETIME.log \
    sh -c "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH; export TRITON_CACHE_DIR=/tmp/triton_cache_\${SLURM_NODEID}; echo ${run_cmd}; ${run_cmd}"

    set +x
fi
