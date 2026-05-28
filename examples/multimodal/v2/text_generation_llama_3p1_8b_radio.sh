#!/bin/bash

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=0

INPUT_IMAGE_PATH="placeholder"
GROUNDTRUTH_PATH="placeholder"
NUM_FRAMES=1
TP=1
#OUT_SEQ_LEN=1024
OUT_SEQ_LEN=16384
#INFERENCE_MAX_SEQ_LEN=2560
INFERENCE_MAX_SEQ_LEN=18384
MAX_NUM_TILES=12

while [[ $# -gt 0 ]]; do
    case $1 in
        --tensor-model-parallel-size)
            TP="$2"
            shift
            shift
            ;;
        --input-image-path)
            INPUT_IMAGE_PATH="$2"
            shift
            shift
            ;;
        --num-frames)
            NUM_FRAMES="$2"
            shift
            shift
            ;;
        --out-seq-length)
            OUT_SEQ_LEN="$2"
            shift
            shift
            ;;
        --inference-max-seq-length)
            INFERENCE_MAX_SEQ_LEN="$2"
            shift
            shift
            ;;
        --max-num-tiles)
            MAX_NUM_TILES="$2"
            shift
            shift
            ;;
        -g|--groundtruth-path)
            GROUNDTRUTH_PATH="$2"
            shift
            shift
            ;;
        -o|--output-path)
            OUTPUT_PATH="$2"
            shift
            shift
            ;;
        -m|--model-path)
            MODEL_PATH="$2"
            shift
            shift
            ;;
        --task)
            TASK="$2"
            shift
            shift
            ;;
        -g|--gt-path)
            GROUNDTRUTH_PATH="$2"
            shift
            shift
            ;;
        -*|--*)
            echo "Invalid option $1"
            exit 1
            ;;
    esac
done

# Please modify these as needed.
NUM_PARTITIONS=0
START=0
END=0

USE_TILING=1

SEQ_LEN=1024
DECODER_SEQ_LEN=16384

EXTRA_ARGS=""

if [[ $USE_TILING -eq 1 ]]; then
    EXTRA_ARGS+=" --pixel-shuffle --use-tiling --max-num-tiles ${MAX_NUM_TILES} --use-thumbnail"
    SEQ_LEN=256
fi

# Automatic distributed setup (works for both single- and multi-node cases). If the script
# is launched under Slurm, we derive the necessary parameters from the Slurm
# environment variables; otherwise we fall back to sensible single-node defaults.
if [[ -z "${MASTER_ADDR}" ]]; then
    if [[ -n "${SLURM_JOB_NODELIST}" ]]; then
        MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
    else
        MASTER_ADDR="localhost"
    fi
fi

MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-${SLURM_NNODES:-1}}
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}

# Print a summary of the launch configuration for easier debugging.
echo "[text_generation_llama_3p1_8b_radio] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NNODES=${NNODES} NODE_RANK=${NODE_RANK} GPUS_PER_NODE=${GPUS_PER_NODE}"

for PARTITION_ID in $( eval echo {$START..$END} )
do
    # python examples/multimodal/run_text_generation.py \
    torchrun \
        --nproc_per_node ${GPUS_PER_NODE} \
        --nnodes ${NNODES} \
        --node_rank ${NODE_RANK} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        examples/multimodal/run_text_generation.py \
        --attention-softmax-in-fp32 \
        --transformer-impl transformer_engine \
        --use-te \
        --use-checkpoint-args \
        --normalization RMSNorm \
        --language-model-type=llama_nemotron_8b \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --position-embedding-type rope \
        --rotary-percent 1.0 \
        --rotary-base 500000 \
        --use-rope-scaling \
        --swiglu \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size 1 \
        --group-query-attention \
        --num-query-groups 8 \
        --num-layers 32 \
        --hidden-size 4096 \
        --ffn-hidden-size 14336 \
        --num-attention-heads 32 \
        --max-position-embeddings 131072 \
        --no-masked-softmax-fusion \
        --load ${MODEL_PATH} \
        --tokenizer-type MultimodalTokenizer \
        --tokenizer-model /lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/ksapra/checkpoints/Llama-3.1-Nemotron-Nano-8B-v1 \
        --tokenizer-prompt-format llama_nemotron_8b \
        --bf16 \
        --micro-batch-size 1 \
        --seq-length ${SEQ_LEN} \
        --decoder-seq-length ${DECODER_SEQ_LEN} \
        --out-seq-length ${OUT_SEQ_LEN} \
        --inference-max-seq-length ${INFERENCE_MAX_SEQ_LEN} \
        --temperature 1.0 \
        --img-h 512 \
        --img-w 512 \
        --patch-dim 16 \
        --seed 153 \
        --top_k 1 \
        --no-load-rng \
        --no-load-optim \
        --input-image-path ${INPUT_IMAGE_PATH} \
        --num-partitions ${NUM_PARTITIONS} \
        --partition-id ${PARTITION_ID} \
        --output-path ${OUTPUT_PATH} \
        --gt-path ${GROUNDTRUTH_PATH} \
        --task ${TASK} \
        ${EXTRA_ARGS} \
        --vision-model-type radio \
        --num-frames ${NUM_FRAMES} \
        --special-tokens "<image>" "<img>" "</img>" "<quad>" "</quad>" "<ref>" "</ref>" "<box>" "</box>" \
        --ckpt-format torch \
        --image-tag-type internvl \
        --disable-vision-class-token \
        --force-system-message \
        --exit-on-missing-checkpoint
done
