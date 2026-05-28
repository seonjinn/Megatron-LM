#!/bin/bash

#SBATCH -A llmservice_fm_vision
#SBATCH -p batch_block1
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton
#SBATCH --nodes=32
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --job-name=sft_llama_3p1_8b_radio_rc3_v13_28_sft1_reasoning_newtokenizer

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

USER=$SLURM_JOB_USER

# Auto-detect batch or interactive mode.
which srun
BATCH=$((1-$?))

DEBUG=0
USE_TILING=1

# Remember to update model and job name if running in batch mode!!
if [[ $BATCH -eq 0 ]]; then
    DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
    MODEL_NAME="interactive_sft_llama_3p1_8b_radio_${DATETIME}"
    SPECIAL_TOKENS="--special-tokens <image> <img> </img> <quad> </quad> <ref> </ref> <box> </box>"
    DEBUG=1
else
    MODEL_NAME="sft_llama_3p1_8b_radio_rc3_v13_28_sft1_reasoning_newtokenizer"
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

CHECKPOINT_DIR="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/ksapra/checkpoints/pretrain_llama_nemotron_reasoning_8b_commercial_0521/checkpoints/"

# DATA_TRAIN="/lustre/fsw/portfolios/llmservice/users/kchumachenko/eagle_recipe/eagle_sft_v13.16_sft1/wds/out.yaml"
#DATA_TRAIN="/lustre/fsw/portfolios/llmservice/users/ikarmanov/eagle_recipe/eagle_sft_v13.22/wds/recipe.yaml"
#DATA_TRAIN="/lustre/fsw/portfolios/llmservice/users/ikarmanov/eagle_recipe/eagle_sft_v13.24/wds/recipe.yaml"
DATA_TRAIN="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/eagle_sft_v13.28/eagle_sft_v13.28/wds/recipe.yaml"

SEQ_LEN=1024
DECODER_SEQ_LEN=16384

if [[ $DEBUG -eq 1 ]]; then
    MBZ=1
    BZ=2
    NW=0
    AD=0.0
    HD=0.0
    LI=1
    EVAL_INTERVAL=99999

    # EXTRA_ARGS="--deterministic-mode --use-cpu-initialization"

    NONDETERMINISTIC_ATTN=1

    NUM_GPU=8
    # NUM_GPU=4
    # export CUDA_VISIBLE_DEVICES=0,1,2,3
else
    MBZ=1
    BZ=128
    NW=8
    AD=0.0
    HD=0.0
    LI=5
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
    NUM_GPU=8
    EVAL_INTERVAL=2000
fi

if [[ $USE_TILING -eq 1 ]]; then
    EXTRA_ARGS+=" --pixel-shuffle --use-tiling --max-num-tiles 12 --use-thumbnail"
    SEQ_LEN=256
fi

OPTIONS=" \
    --use-checkpoint-args \
    --disable-bias-linear \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model /lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/ksapra/checkpoints/Llama-3.1-Nemotron-Nano-8B-v1 \
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
    --pipeline-model-parallel-size 1 \
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
    --lr-warmup-fraction 0.03 \
    --micro-batch-size ${MBZ} \
    --global-batch-size ${BZ} \
    --lr 2e-5 \
    --min-lr 0.0 \
    --lr-decay-style cosine \
    --log-interval ${LI} \
    --eval-iters 10 \
    --eval-interval ${EVAL_INTERVAL} \
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
    --language-model-type=llama_nemotron_8b \
    ${EXTRA_ARGS} \
    --distributed-timeout-minutes 60 \
    --vision-model-type radio \
    --tokenizer-prompt-format llama_nemotron_8b \
    --use-loss-scaling \
    --packing-seq-length ${DECODER_SEQ_LEN} \
    ${SPECIAL_TOKENS} \
    --ckpt-format torch \
    --image-tag-type internvl \
    --disable-vision-class-token \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 32 \
    --recompute-vision \
"

export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NONDETERMINISTIC_ATTN}

#pip install git+https://github.com/NVIDIA/Megatron-Energon.git#egg=megatron-energon[av_decode]
#pip uninstall fastseek
# Interactive or batch mode
if [[ $BATCH -eq 0 ]]; then
    torchrun --nproc_per_node ${NUM_GPU} examples/multimodal/train.py ${OPTIONS}
else
    run_cmd="cd ${SOURCE}; python -u examples/multimodal/train.py ${OPTIONS}"

    DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

    srun -l --verbose \
    --container-image /lustre/fsw/portfolios/llmservice/users/matthieul/docker/megatron-dev-img-05142025-pytorch-dev-te-cd37379-energon-710-mamba-fix-vlmeval.sqsh \
    --container-mounts "/lustre" \
    --output=${LOGS_DIR}/%x_%j_$DATETIME.log \
    sh -c "${run_cmd}"

    set +x
fi
