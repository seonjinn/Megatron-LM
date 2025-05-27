#!/bin/bash

#SBATCH -A llmservice_fm_vision
#SBATCH -p batch_block1
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --job-name=math500_llama_3p1_radio




USER=$SLURM_JOB_USER

export HF_DATASETS_CACHE="/lustre/fsw/portfolios/llmservice/users/${USER}/hf_cache"

#MODEL_NAME="pretrain_llama_nemotron_reasoning_8b_commercial_0521"
#MODEL_NAME="v1316_reasoning_20.5Mtext+SFT"
#MODEL_NAME="v1316_reasoning_llm_baseline"
MODEL_NAME="v1316_reasoning_2.5Mtext+SFT"
TP=1

WORKSPACE="/lustre/fsw/portfolios/llmservice/users/${USER}/workspace"
SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
#MODEL_PATH="${OUTPUT_BASE}/${MODEL_NAME}"
MODEL_PATH=/lustre/fs1/portfolios/llmservice/users/ksapra/checkpoints/tp1/${MODEL_NAME}
#MODEL_PATH=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/ksapra/checkpoints/{MODEL_NAME}
echo $MODEL_PATH
sleep 5

TASK="Math500"

OUTPUT_DIR="${WORKSPACE}/benchmarks/${TASK}/${MODEL_NAME}"
echo $OUTPUT_DIR
mkdir -p ${OUTPUT_DIR}
rm ${OUTPUT_DIR}/*
OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_NAME}"

OPTIONS=" \
    --input-image-path ignored \
    --model-path ${MODEL_PATH}/ \
    --gt-path ignored \
    --output-path ${OUTPUT_PATH} \
    --task ${TASK} \
    --tensor-model-parallel-size ${TP} \
"

# Ensure required evaluation dependencies are present.
#python -m pip install --upgrade pip --quiet
# Install Math500 evaluator dependencies (latex parser & sympy). The
# latex2sympy2 wheel already depends on a compatible antlr4 runtime (4.9.3),
# so we omit a separate antlr4 pin to avoid version conflicts.
pip install --quiet sympy==1.13.1 latex2sympy2==1.9.1

run_cmd=" \
    cd ${SOURCE}; \
    # If this script is executed under an allocation that still has multiple
    # tasks per node (e.g. in interactive mode), ensure we only invoke the
    # workload once per node (local ID 0).
    if [[ -z \"${SLURM_LOCALID}\" || \"${SLURM_LOCALID}\" == 0 ]]; then \
        examples/multimodal/eagle/text_generation_llama_3p1_8b_radio.sh ${OPTIONS}; \
    else \
        echo \"Skipping duplicate task with SLURM_LOCALID=${SLURM_LOCALID}\"; \
    fi; \
    python examples/multimodal/evaluation/evaluate_math500.py --input-path ${OUTPUT_PATH} | tee ${OUTPUT_PATH}-${TASK}-score.txt"

# Auto-detect batch or interactive mode.
which srun
BATCH=$((1-$?))

if [[ $BATCH -eq 0 ]]; then
    cd ${SOURCE}
    . examples/multimodal/v2/text_generation_llama_3p1_8b_radio.sh ${OPTIONS}
    python examples/multimodal/evaluation/evaluate_math500.py --input-path ${OUTPUT_PATH}/${MODEL_NAME} | tee ${OUTPUT_PATH}-${TASK}-score.txt
else
    DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

    srun -l --verbose \
    --container-image /lustre/fsw/portfolios/llmservice/users/matthieul/docker/megatron-dev-img-05142025-pytorch-dev-te-cd37379-energon-710-mamba-fix-vlmeval.sqsh \
    --container-mounts "/lustre" \
    --output=${OUTPUT_DIR}/logs/%x_%j_$DATETIME.log \
    sh -c "${run_cmd}"
    set +x
fi
