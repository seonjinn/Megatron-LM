#!/bin/bash

MCORE_TORCH_CKPT_DIR=$1
MCORE_DIST_CKPT_DIR=$2
TARGET_TP=$3

export UB_TIMEOUT=720
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN
export TORCHINDUCTOR_WORKER_START=fork

########################### AUTO RESUME ##################################

# ORIG_SLURM_JOB_ID=$1
# if [[ "$ORIG_SLURM_JOB_ID" == "" ]]; then
#   ORIG_SLURM_JOB_ID=$SLURM_JOB_ID
# fi
# echo ORIG JOBID $ORIG_SLURM_JOB_ID

# PREV_SLURM_JOB_ID=$2
# if [[ "$PREV_SLURM_JOB_ID" != "" ]]; then
#     PREV_STATUS=`sacct -j $PREV_SLURM_JOB_ID -P -n -o State | head -n 1`
#     if [[ "$PREV_STATUS" != "TIMEOUT" && "$PREV_STATUS" != "PREEMPTED" && "$PREV_STATUS" != "NODE_FAIL" ]]; then
#         echo "PREVIOUS JOB $PREV_SLURM_JOB_ID FINISHED WITH $PREV_STATUS STATUS. EXIT."
#         exit 1
#     fi
#     echo "PREVIOUS JOB $PREV_SLURM_JOB_ID FINISHED WITH $PREV_STATUS STATUS. RESUMING..."
# fi
# echo PREV JOBID $PREV_SLURM_JOB_ID
# sbatch -J "${SLURM_JOB_NAME}" --dependency=afternotok:"${SLURM_JOB_ID}" "$0" "${ORIG_SLURM_JOB_ID}" "${SLURM_JOB_ID}"

########################### CHANGE #######################################

REPO_DIR="/lustre/fsw/portfolios/llmservice/users/ameyasunilm/codebases/nano-v2-megatron-lm"
OUTPUT_ROOT="/lustre/fsw/portfolios/llmservice/users/matthieul/repos_rebase/megatron-lm-main/experiments"
IMAGE_PATH="/lustre/fsw/portfolios/llmservice/users/kezhik/images/adlr+megatron-lm+pytorch+nemotron5p5-apr2025-nvrx-patchedte+datasets.sqsh"

BLEND_PATH="/lustre/fsw/portfolios/llmservice/users/ameyasunilm/projects/nano_v2/data/sft1/sft1_128kpacked_multilingual.json"
PRETRAINED_CKPT="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/data/adlr-nlp-sharing/nemotron5.5/12b_hybrid/checkpoints/lc_extension-sp_emb_reinit"
TOKENIZER_MODEL="/lustre/fsw/portfolios/llmservice/users/kezhik/images/Nemotron-H-4B-Instruct"
PROMPT_FORMAT="nemotron-h-aligned"

WANDB_PROJECT="Nano-V2-SFT"
NAME="nano-v2-sft-stage1-multilingual-lr5e-6-128k-seqpacked-0711"

export HF_HOME="/lustre/fsw/portfolios/llmservice/users/ameyasunilm/hf_cache"
export WANDB_API_KEY="fc9eb03df9400c4953caa40fc59b2ffa7724dd82"
export WANDB_RESUME="allow"
export WANDB_RUN_ID=${NAME}

TRAIN_SAMPLES=2340000
LR_DECAY_SAMPLES=234000
LR_WARMUP_SAMPLES=78000
LR_DECAY_STYLE="constant"
LR=5e-6
MIN_LR=5e-7
GBS=64
LOG_INTERVAL=10
EVAL_INTERVAL=300
MAX_SEQ_LENGTH=131072

##########################################################################

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

# if [ -n "${SLURM_JOB_ID:-}" ] ; then
#     SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
#     ENV_LOG_FILENAME=${NAME}_${SLURM_JOB_ID}_${DATETIME}.env.log
# else
#     SCRIPT_PATH=$(realpath "$0")
#     ENV_LOG_FILENAME=${NAME}_${DATETIME}.env.log
# fi

# SCRIPT_DIR=$(dirname ${SCRIPT_PATH})

RUN_DIR="${OUTPUT_ROOT}/${NAME}"; mkdir -p ${RUN_DIR}
LOGS_DIR="${RUN_DIR}/logs/${NAME}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints/${NAME}"
DATACACHE_DIR="${RUN_DIR}/../data_cache"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard/${NAME}"

# Mamba triton cache.
export TRITON_CACHE_DIR="/lustre/fsw/portfolios/llmservice/users/matthieul/repos_rebase/megatron-lm-main/triton_cache"
export TRITON_CACHE_MANAGER="megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"

mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

##################################################################

# echo "<< START PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo "IMAGE_PATH=${IMAGE_PATH}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo "OUTPUT_ROOT=${OUTPUT_ROOT}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo "SCRIPT_DIR=${SCRIPT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo "REPO_DIR=${REPO_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo "RUN_DIR=${RUN_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo "LOGS_DIR=${LOGS_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo "DATACACHE_DIR=${DATACACHE_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo "SCRIPT_DIR=${SCRIPT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo "REPO_DIR=${REPO_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo "<< END PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

# echo "<< START GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo "GIT LOG" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# git -C ${REPO_DIR} log --oneline -1 |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo "GIT STATUS" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# git -C ${REPO_DIR} status --porcelain --branch |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo "GIT DIFF" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# git -C ${REPO_DIR} diff |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo "<< END GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

# echo "<< START ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# env |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
# echo "<< END ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

########################### CHANGE #######################################
# --context-parallel-size 2 \
# --disable-gloo-process-groups \

options=" \
    --adam-beta2 0.98 \
    \
    --sft \
    --sft-tokenizer-prompt-format ${PROMPT_FORMAT} \
    --tokenizer-type SFTTokenizer \
    --tokenizer-model /lustre/fsw/portfolios/llmservice/users/kezhik/images/Nemotron-H-4B-Instruct \
    \
    --log-interval ${LOG_INTERVAL} \
    --micro-batch-size 1 \
    --global-batch-size ${GBS} \
    --train-samples ${TRAIN_SAMPLES} \
    --lr-decay-samples ${LR_DECAY_SAMPLES} \
    --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
    --lr-decay-style ${LR_DECAY_STYLE} \
    --lr ${LR}  \
    --min-lr ${MIN_LR} \
    \
    --seq-length ${MAX_SEQ_LENGTH} \
    --max-position-embeddings ${MAX_SEQ_LENGTH} \
    --eval-iters 2 \
    --eval-interval ${EVAL_INTERVAL} \
    --weight-decay 0.0 \
    --save-interval ${EVAL_INTERVAL} \
    --pretrained-checkpoint ${PRETRAINED_CKPT} \
    --tensor-model-parallel-size ${TARGET_TP} \
    \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-exp-name ${NAME} \
    \
    --fp8-recipe blockwise \
    --fp8-format e4m3 \
    --first-last-layers-bf16 \
    --num-layers-at-start-in-bf16 2 \
    --num-layers-at-end-in-bf16 2 \
    --fp8-param-gather \
    --attention-backend flash \
    --is-hybrid-model \
    --mamba-head-dim 80 \
    --hybrid-override-pattern M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M- \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --per-split-data-args-path ${BLEND_PATH} \
    --tiktoken-pattern v2 \
    --distributed-timeout-minutes 10 \
    --use-mcore-models \
    --data-cache-path ${DATACACHE_DIR} \
    --no-mmap-bin-files \
    --sequence-parallel \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --init-method-std 0.0125 \
    --position-embedding-type none \
    --squared-relu \
    --num-layers 62 \
    --hidden-size 5120 \
    --num-attention-heads 40 \
    --group-query-attention \
    --num-query-groups 8 \
    --ffn-hidden-size 20480 \
    --kv-channels 128 \
    --normalization RMSNorm \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 5750 \
    --pipeline-model-parallel-size 1 \
    --clip-grad 1.0 \
    --load ${MCORE_TORCH_CKPT_DIR} \
    --save ${MCORE_DIST_CKPT_DIR} \
    --ckpt-format torch \
    --ckpt-fully-parallel-save \
    --ckpt-fully-parallel-load \
    --ckpt-assume-constant-structure \
    --log-progress  \
    --timing-log-option minmax \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --log-energy \
    --bf16 \
    --adam-beta1 0.9 \
    --use-distributed-optimizer \
    --ddp-num-buckets 5 \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --no-create-attention-mask-in-dataloader \
    --manual-gc \
    --num-workers 1 \
    --log-straggler \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 \
    --check-weight-hash-across-dp-replicas-interval 20000 \
    --rerun-mode disabled \
    --ckpt-convert-format torch_dist \
    --ckpt-convert-save ${MCORE_DIST_CKPT_DIR} \
    --ckpt-step 1 \
    --no-load-optim \
    --no-load-rng"

torchrun --nproc_per_node ${TARGET_TP} pretrain_mamba.py ${options}
