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
#SBATCH --job-name=ultra_long_context_0222

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MSC_CONFIG="/lustre/fsw/portfolios/llmservice/users/matthieul/msc_config/msc_config.yaml"

export UB_TIMEOUT=720
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN
export TORCHINDUCTOR_WORKER_START=fork
#export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"/tmp/triton_cache_\${SLURM_NODEID}"}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

USER=$SLURM_JOB_USER

# Auto-detect batch or interactive mode.
which srun
BATCH=$((1-$?))

DEBUG=0
USE_DYNAMIC_RES=1
USE_IMAGE_BREAK=0   # Only used if USE_DYNAMIC_RES is 1.
USE_CONV_MERGE=0    # Only used if USE_DYNAMIC_RES is 1.
USE_FP8=0
USE_CPE_EVAL_MODE=1
VIDEO_MAX_NUM_FRAMES=128     # Values > 0 enable video max num frames.
USE_CP=1
USE_MEGA_LONG_MODE=1
USE_ULTRA_LONG_MODE=0

if [ $USE_MEGA_LONG_MODE -eq 1 ] && [ $USE_ULTRA_LONG_MODE -eq 1 ]; then
    echo "Error: USE_MEGA_LONG_MODE and USE_ULTRA_LONG_MODE cannot both be enabled simultaneously."
    exit 1
fi

# Remember to update model and job name if running in batch mode!!
if [[ $BATCH -eq 0 ]]; then
    DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
    MODEL_NAME="interactive_ultra_long_context_${DATETIME}"
    SPECIAL_TOKENS="--special-tokens <image> <img> </img> <quad> </quad> <ref> </ref> <box> </box>"
    DEBUG=1
else
    MODEL_NAME="ultra_long_context_0222"
    SPECIAL_TOKENS="--special-tokens \<image\> \<img\> \</img\> \<quad\> \</quad\> \<ref\> \</ref\> \<box\> \</box\>"
fi

WANDB_API_KEY=${WANDB_API_KEY}
WANDB_PROJECT=${WANDB_PROJECT:-"megatron-vlm-v3"}
WANDB_ENTITY=${WANDB_ENTITY:-"hwinf_dcm"}
WANDB_NAME=${MODEL_NAME}

WORKSPACE="/lustre/fsw/portfolios/llmservice/users/${USER}/workspace"
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
if [[ $DEBUG -eq 0 ]]; then
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
else
    CODE_DIR="${SOURCE}"
fi

TP=2
EP=32
CHECKPOINT_DIR="/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/sft_moe_video_lower_lr_lower_load_rl_llm_eval_mode_radio_v4_1214/checkpoints"
CHECKPOINT_DIR="/lustre/fsw/portfolios/llmservice/users/amalasanjayd/workspace/output/sft_moe_rl_llm_eval_mode_radio_v4_v1365_0126/checkpoints/"
CHECKPOINT_DIR="/lustre/fsw/portfolios/llmservice/users/amalasanjayd/workspace/output/sft_moe_rl_llm_eval_mode_radio_v4_v1365_text_0129/checkpoints/"
#CHECKPOINT_DIR="/lustre/fsw/portfolios/llmservice/users/amalasanjayd/workspace/output/sft_moe_rl_llm_eval_mode_radio_v4_v1369_keephist_0215/checkpoints"
#CHECKPOINT_DIR="/lustre/fsw/portfolios/llmservice/users/amalasanjayd/workspace/output/sft_moe_rl_llm_eval_mode_radio_v4_v1367_0205/checkpoints/"

# New tokenizer 10/20.
TOKENIZER_MODEL="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/megatron-lm/nano-v3-tokenizer-vlm" # "/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/hf-transformers/hub/models--nvidia--Nemotron-Nano-3-30B-A3.5B-dev-1016/snapshots/bb271274159f07461e919379311e32802e5ec36b/"
TOKENIZER_PROMPT_FORMAT="nemotron6-moe"

# DATA_TRAIN="/lustre/fsw/portfolios/llmservice/users/matthieul/eagle_recipe_online_packing/final_recipe/eagle_sft_v13.52.no.text.yaml"
DATA_TRAIN="${SOURCE}/examples/multimodal/v3_baseline/ultra_long_context.yaml"

if [[ $DEBUG -eq 1 ]]; then
    MBZ=1
    BZ=8
    NW=0
    AD=0.0
    HD=0.0
    LI=1

    EXTRA_ARGS=""

    NUM_GPU=8
else
    MBZ=1
    BZ=128

    if [[ $USE_ULTRA_LONG_MODE -eq 1 ]]; then
        BZ=$((BZ/2))
    fi

    NW=4
    AD=0.0
    HD=0.0
    LI=5
    EXTRA_ARGS=""
    NUM_GPU=8
fi

PBS=1024

SEQ_LEN=256
DECODER_SEQ_LEN=49152
if [[ $USE_MEGA_LONG_MODE -eq 1 ]]; then
    DECODER_SEQ_LEN=122880
elif [[ $USE_ULTRA_LONG_MODE -eq 1 ]]; then
    DECODER_SEQ_LEN=245760
fi

if [ -n "${WANDB_API_KEY}" ]; then
    EXTRA_ARGS+=" --wandb-project ${WANDB_PROJECT} --wandb-exp-name ${MODEL_NAME} --wandb-save-dir ${WANDB_DIR} --wandb-resume-same-run"
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
    EXTRA_ARGS+=" --dynamic-resolution --dynamic-resolution-min-patches 1024 --dynamic-resolution-max-patches 13312 --apply-data-augment"
fi

if [[ $USE_CPE_EVAL_MODE -eq 1 ]]; then
    EXTRA_ARGS+=" --radio-force-cpe-eval-mode"  # Only CPE in eval mode (not entire vision encoder)
fi

if [[ $USE_CP -eq 1 ]]; then
    CP_SIZE=2
    if [[ $USE_MEGA_LONG_MODE -eq 1 ]]; then
        CP_SIZE=8
    elif [[ $USE_ULTRA_LONG_MODE -eq 1 ]]; then
        CP_SIZE=16
    fi
    EXTRA_ARGS+=" --context-parallel-size ${CP_SIZE}"
fi

EXTRA_ARGS+=" --packing-buffer-size ${PBS} --packing-seq-length ${DECODER_SEQ_LEN} --packing-knapsack-algorithm balanced_greedy_knapsack "
# LM (Mamba block) recompute
EXTRA_ARGS+=" --recompute-granularity selective --recompute-modules core_attn mlp layernorm moe_act moe "
# core_attn moe_act layernorm mlp moe
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
    --lr 5e-6 \
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
"

# --tokenizer-keep-history-thinking \

export WANDB_ENTITY=$WANDB_ENTITY  # Not passed in via command line args, only env vars
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
    --container-image /lustre/fsw/portfolios/llmservice/users/matthieul/docker/pytorch25.06-moe-avlm-editable-energon-732-mamba-fix.sqsh \
    --container-mounts "/lustre" \
    --output=${LOGS_DIR}/%x_%j_$DATETIME.log \
    sh -c "echo ${run_cmd}; ${run_cmd}"

    set +x
fi
