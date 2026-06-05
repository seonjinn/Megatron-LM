#!/bin/bash

#SBATCH -p batch
#SBATCH --account=llmservice_nemotron_super
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=8
#SBATCH --time=3:45:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --dependency=singleton
#SBATCH --job-name=smy-super-v3-TP8-CP8-EP8-N256-revised-notool-jan8


export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
# export TORCHINDUCTOR_WORKER_START=fork
export QUANTIZATION_TYPE_DEBUG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16

# Debug: See NCCL operations during checkpoint load
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

export HF_HOME="/lustre/fsw/portfolios/llmservice/users/soumyes/sft-runs/hf_home"

NAME=${SLURM_JOB_NAME}

OUTPUT_ROOT="/lustre/fsw/portfolios/llmservice/users/soumyes/sft-runs"
MEGATRON_LM_DIR=/scratch/fsw/portfolios/llmservice/users/abukharin/sft/code/skip_nvfp4/duncan-hybrid-packed-sequence-for-training-moe-june2025
IMAGE=/scratch/fsw/portfolios/llmservice/users/abukharin/containers/pytorch-2506-py3-mamba_seq_pack_fixed_3p1.sqsh

# 2. WANDB_API_KEY: Use your own API key from wandb.ai (REQUIRED)
# export WANDB_API_KEY="ab7c9da76db9ffe53f44df501978d69b1b241bda"
# WANDB_PROJECT="super-v3-sft"

RUN_DIR="${OUTPUT_ROOT}"
LOGS_DIR="/scratch/fsw/portfolios/llmservice/users/tpoon/repos_super/megatron-lm/temp/logs/${NAME}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints/${NAME}"
DATACACHE_DIR="${RUN_DIR}/data_cache/${NAME}"
TENSORBOARD_DIR="/scratch/fsw/portfolios/llmservice/users/tpoon/repos_super/megatron-lm/temp/tensorboard/${NAME}"


mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

# Use node-local temp directory for triton/inductor cache to avoid race conditions on shared filesystem
export TRITON_CACHE_DIR="/tmp/triton-cache-${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor-cache-${SLURM_JOB_ID:-$$}"

# Reduce parallel compilation threads to minimize race conditions
export TORCHINDUCTOR_COMPILE_THREADS=1

# Add /sbin to PATH for ldconfig (needed by Triton)
export PATH="/sbin:/usr/sbin:$PATH"


DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
    ENV_LOG_FILENAME=${NAME}_${SLURM_JOB_ID}_${DATETIME}.env.log
else
    SCRIPT_PATH=$(realpath "$0")
    ENV_LOG_FILENAME=${NAME}_${DATETIME}.env.log
fi

SCRIPT_DIR=$(dirname ${SCRIPT_PATH})

################################################################
### Log environment
################################################################
echo "<< START PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "IMAGE=${IMAGE}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
#echo "OUTPUT_ROOT=${OUTPUT_ROOT}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "MEGATRON_LM_DIR=${MEGATRON_LM_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "RUN_DIR=${RUN_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "LOGS_DIR=${LOGS_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "DATACACHE_DIR=${DATACACHE_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "SCRIPT_DIR=${SCRIPT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT LOG" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} log --oneline -1 |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT STATUS" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} status --porcelain --branch |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT DIFF" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} diff |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
env |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}


#--result-rejected-tracker-filename ${RESULT_REJECTED_TRACKER_FILENAME} \
#--iterations-to-skip ${ITERATIONS_TO_SKIP} \
#--rerun-mode validate_results \
#
#--enable-experimental \
#--moe-shared-expert-overlap \

	# Can not use with FP4

	# MXFP8
	#--moe-router-padding-for-fp8 \
	#--fp8-format e4m3 \
	#--fp8-recipe mxfp8 \
	#--fp8-param-gather \
	#--reuse-grad-buf-for-mxfp8-param-ag \

	# Additional options
	#--recompute-modules layernorm moe_act \
	#
	#--recompute-granularity selective \
	#--recompute-modules moe \
	#
	#--tp-comm-overlap \

	# Short context, use
	# --enable-cuda-graph \
	# Long context, use
	# --recompute-granularity selective \
	# --recompute-modules moe \

	# NVFP4 args
	# --keep-mtp-spec-in-bf16 \
	# --keep-mamba-stack-attention-linear-in-bf16 \
	# --keep-mamba-out-proj-in-mxfp8 \
	# --keep-moe-latent-projections-in-bf16 \
	# --first-last-layers-bf16 \
	# --num-layers-at-start-in-bf16 0 \
	# --num-layers-at-end-in-bf16 14 \
	# --fp4-format e2m1 \
	# --fp4-recipe nvfp4 \

	# checkpoint load fix
	# --cuda-graph-scope mamba attn moe_router \
		# --ckpt-fully-parallel-load \
		# --async-save \
			# --use-persistent-ckpt-worker \

SEQ_LEN=262144
TRAIN_SAMPLES=3000000
LR_WARMUP_SAMPLES=15000
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES-LR_WARMUP_SAMPLES))
LOG_INTERVAL=10
GBS=64
LR=5e-5
MIN_LR=2e-5

TOKENIZER_MODEL_PATH="/scratch/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/geshen/nano_sft/nano-v3-sft-tokenizer"
BASE_MODEL_PATH="/lustre/fsw/portfolios/llmservice/users/abukharin/sft/results/checkpoints/super-repeated-mtp-reinit-embeddings"
BLEND_PATH="/lustre/fsw/portfolios/llmservice/users/soumyes/sft-runs/blends/super_revised_notool.json"


OPTIONS=" \
    --sft \
    --sft-tokenizer-prompt-format identity \
    --distributed-timeout-minutes 20 \
    --num-dataset-builder-threads 32 \
    --tokenizer-type SFTTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL_PATH} \
	\
	--recompute-granularity selective \
	--recompute-modules moe \
	\
	--tensor-model-parallel-size 2 \
	--expert-model-parallel-size 64 \
	--expert-tensor-parallel-size 1 \
	--pipeline-model-parallel-size 1 \
	--hybrid-override-pattern 'MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME' \
	--mtp-hybrid-override-pattern \"*E\" \
	--mtp-use-repeated-layer \
	\
	--pretrained-checkpoint ${BASE_MODEL_PATH} \
	--save-interval 200 \
	--save-retain-interval 200 \
	--lr $LR \
	--min-lr $MIN_LR \
	--lr-decay-style constant \
	--train-samples ${TRAIN_SAMPLES} \
	--lr-warmup-samples ${LR_WARMUP_SAMPLES} \
	--lr-decay-samples ${LR_DECAY_SAMPLES} \
	--seq-length ${SEQ_LEN} \
	--max-position-embeddings ${SEQ_LEN} \
	--log-interval ${LOG_INTERVAL} \
	--micro-batch-size 1 \
	--global-batch-size ${GBS} \
	--overlap-grad-reduce \
	--overlap-param-gather \
	\
	--mtp-spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
	--mtp-num-layers 2 \
	--calculate-per-token-loss \
	--mtp-loss-scaling-factor 0.3 \
	\
	--cuda-graph-scope mamba attn moe_router \
	--te-rng-tracker \
	--high-priority-stream-groups ep \
	--manual-gc-interval 10 \
	--moe-shared-expert-compute-before-router \
	--ddp-num-buckets 10 \
	--manual-gc \
	\
	--moe-latent-size 1024 \
	--moe-permute-fusion \
	--cross-entropy-loss-fusion \
	--cross-entropy-fusion-impl native \
	--use-fused-weighted-squared-relu \
	\
	--moe-token-dispatcher-type alltoall \
	--moe-router-score-function sigmoid \
	--moe-grouped-gemm \
	--num-experts 512 \
	--moe-router-topk 22 \
	--moe-aux-loss-coeff 1e-4 \
	--moe-router-topk-scaling-factor 5.0 \
	--moe-router-enable-expert-bias \
	--moe-router-dtype fp32 \
	--moe-router-load-balancing-type seq_aux_loss \
	--moe-shared-expert-intermediate-size 5376 \
	\
	--attention-backend flash \
	--num-workers 1 \
	--ckpt-format torch_dist \
	--ckpt-fully-parallel-save \
	--ckpt-fully-parallel-load \
	--ckpt-assume-constant-structure \
	--use-persistent-ckpt-worker \
        \
	--squared-relu \
        --no-mmap-bin-files \
	--exit-duration-in-mins 5750 \
	--no-create-attention-mask-in-dataloader \
        \
        --sequence-parallel \
	--use-distributed-optimizer \
	--override-opt_param-scheduler \
        \
	--mamba-num-heads 128 \
	--is-hybrid-model \
	--untie-embeddings-and-output-weights \
        --init-method-std 0.014 \
        --position-embedding-type none \
        --num-layers 88 \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --group-query-attention \
        --num-query-groups 2 \
	--ffn-hidden-size 2688 \
	--kv-channels 128 \
	--save /scratch/fsw/portfolios/llmservice/users/tpoon/repos_super/megatron-lm/temp/checkpoints/${NAME} \
        --load ${CHECKPOINT_DIR} \
	--per-split-data-args-path ${BLEND_PATH} \
	--data-cache-path ${DATACACHE_DIR} \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --disable-bias-linear \
        --normalization RMSNorm \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
	--log-params-norm \
	--log-num-zeros-in-grad \
	--log-throughput \
	--log-progress \
	--log-energy \
	--log-memory-interval 200 \
	--logging-level 20 \
	--log-straggler \
	--disable-straggler-on-startup \
	--straggler-minmax-count 16 \
	--check-weight-hash-across-dp-replicas-interval 20000 \
	--ddp-pad-buckets-for-high-nccl-busbw \
	--timing-log-option minmax \
        --eval-interval 1000 \
        --eval-iters 14 \
        --bf16 \
        --use-mcore-models \
        --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
	--dist-ckpt-strictness log_unexpected \
	--ckpt-convert-format torch \
	--ckpt-convert-save /scratch/fsw/portfolios/llmservice/users/tpoon/checkpoints/super-v3-posttrained-tp2-ep64-mcore \
        --tensorboard-dir ${TENSORBOARD_DIR}"

RUN_CMD="python -u ${MEGATRON_LM_DIR}/pretrain_mamba.py ${OPTIONS}"

srun -l \
     --container-image=${IMAGE} \
     --container-mounts="/scratch:/scratch,/lustre:/lustre" \
     --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
     sh -c "${RUN_CMD}"
