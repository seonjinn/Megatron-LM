#!/bin/bash

#SBATCH -A llmservice_fm_vision
# DFW: batch
# OCI-IAD: batch_block1,batch_block3,batch_block4,backfill_block1,backfill_block2,backfill_block3,backfill_block4
# OCI-ORD: grizzly,polar,polar3,polar4
#SBATCH -p batch_block1,backfill,batch_large,batch_long
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --gpus-per-node=8
#SBATCH --job-name=llama3p1-8b-radio-nvwhisper-mlp-stage1-0627

# Please launch this script from megatron-lm root.

export CUDA_DEVICE_MAX_CONNECTIONS=1

USER=$SLURM_JOB_USER

# Auto-detect batch or interactive mode.
which srun
BATCH=$((1-$?))

DEBUG=0
USE_PACKING=1
USE_ROTE=0

if [[ $BATCH -eq 0 ]]; then
    DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
    MODEL_NAME="interactive-mcore-llama3p1-8b-radio-nvwhisper-mlp-stage1-${DATETIME}"
    SPECIAL_TOKENS=" --special-tokens <image> <img> </img> <quad> </quad> <ref> </ref> <box> </box> <s> <im_embedding> <im_start> <im_end> <video> <vi_embedding> <vi_start> <vi_end> <video-sound> <vis_embedding> <vis_start> <vis_end> <sound> <so_embedding> <so_start> <so_end> "
    DEBUG=1
else
    MODEL_NAME="mcore-llama3p1-8b-radio-nvwhisper-mlp-stage1-0627"
    SPECIAL_TOKENS=" --special-tokens \<image\> \<img\> \</img\> \<quad\> \</quad\> \<ref\> \</ref\> \<box\> \</box\> \<s\> \<im_embedding\> \<im_start\> \<im_end\> \<video\> \<vi_embedding\> \<vi_start\> \<vi_end\> \<video-sound\> \<vis_embedding\> \<vis_start\> \<vis_end\> \<sound\> \<so_embedding\> \<so_start\> \<so_end\> "
fi

WORKSPACE=/lustre/fsw/portfolios/llmservice/users/${USER}/workspace
SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

# Latest doc intelligence model:
CHECKPOINT_DIR="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/output/kchumachenko_sft_llama_3p1_8b_radio_rc3_v13_16_sft1_0509/checkpoints/"

DATA_TRAIN="${SOURCE}/examples/multimodal/avlm/stage1_blend.yaml"

if [[ $DEBUG -eq 1 ]]; then
    MBZ=1
    BZ=8
    NW=0
    LI=1
    AD=0.0
    HD=0.0
    EXTRA_ARGS=""
    ALLOW_NONDETERMINISTIC=1
    PBS=128

    export CUDA_VISIBLE_DEVICES=0,1,2,3
    NUM_GPU=4
    #NUM_GPU=8
else
    MBZ=1
    BZ=2048
    NW=8
    LI=5
    AD=0.0
    HD=0.0
    EXTRA_ARGS=""
    ALLOW_NONDETERMINISTIC=1
    PBS=128

    NUM_GPU=8
fi

SEQ_LEN=256              # Image embeddings sequence length.
DECODER_SEQ_LEN=24576    # Language model sequence length.

EXTRA_ARGS+=" --pixel-shuffle --use-tiling --max-num-tiles 12 --use-thumbnail "

if [[ $USE_PACKING -eq 1 ]]; then
    EXTRA_ARGS+=" --packing-seq-length ${DECODER_SEQ_LEN} --packing-buffer-size ${PBS} "
fi

if [[ $USE_ROTE -eq 1 ]]; then
    EXTRA_ARGS+=" --use-video-rote --use-sound-rote "
fi

OPTIONS=" \
    --disable-vision-class-token \
    --swiglu \
    --use-distributed-optimizer \
    --num-workers ${NW} \
    --normalization RMSNorm \
    --num-attention-heads 32 \
    --num-layers 32 \
    --hidden-size 4096 \
    --exit-duration-in-mins 230 \
    --group-query-attention \
    --num-query-groups 8 \
    --ffn-hidden-size 14336 \
    --seq-length ${SEQ_LEN} \
    --decoder-seq-length ${DECODER_SEQ_LEN} \
    --max-position-embeddings ${DECODER_SEQ_LEN} \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model /lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/ \
    --tokenizer-prompt-format llama3p1 \
    --vocab-size 128512 \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 5000000 \
    --use-rope-scaling \
    --disable-bias-linear \
    --tensor-model-parallel-size 4 \
    --language-model-type llama3.1_8b \
    --vision-model-type radio \
    --sound-model-type hf:///lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/NV-Whisper \
    --sound-target-rate 16000 \
    --num-frames 8 \
    --micro-batch-size ${MBZ} \
    --global-batch-size ${BZ} \
    --train-full-dataset \
    --lr-warmup-fraction 0.03 \
    --lr 1e-4 \
    --min-lr 2.5e-6 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 0.01 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --attention-dropout ${AD} \
    --hidden-dropout ${HD} \
    --untie-embeddings-and-output-weights \
    --eod-mask-loss \
    --bf16 \
    --tensorboard-dir=${TENSORBOARD_DIR} \
    --freeze-LM \
    --freeze-ViT \
    --freeze-sound-model \
    --img-h 512 \
    --img-w 512 \
    --patch-dim 16 \
    --data-path ${DATA_TRAIN} \
    --dataloader-type external \
    --split 100,0,0 \
    --prompt-path ${SOURCE}/examples/multimodal/nvlm/nvlm_prompts.json \
    --log-interval ${LI} \
    --save-interval 2000 \
    --eval-interval 500 \
    --eval-iters 10 \
    ${EXTRA_ARGS} \
    ${SPECIAL_TOKENS} \
    --save ${FINETUNE_DIR} \
    --load ${FINETUNE_DIR} \
    --dataloader-save ${FINETUNE_DIR}/dataloader \
    --allow-missing-sound-projection-checkpoint \
    --allow-missing-sound-model-checkpoint \
    --pretrained-checkpoint ${CHECKPOINT_DIR} \
    --use-te \
    --ckpt-format torch \
    --image-tag-type nvlm \
    --allow-large-videos \
"

export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${ALLOW_NONDETERMINISTIC}

# Interactive or batch mode
if [[ $BATCH -eq 0 ]]; then
    torchrun --nproc_per_node ${NUM_GPU} examples/multimodal/train.py ${OPTIONS}
else
    run_cmd="cd ${SOURCE}; python -u examples/multimodal/train.py ${OPTIONS}"

    DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

    srun -l --verbose \
    --container-image /lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/containers/megatron-dev-img-05142025-pytorch-dev-te-cd37379-editable-energon-mamba-fix-vlmeval-audio.sqsh \
    --container-mounts "/lustre" \
    --output=${LOGS_DIR}/%x_%j_$DATETIME.log \
    sh -c "${run_cmd}"

    set +x
fi

