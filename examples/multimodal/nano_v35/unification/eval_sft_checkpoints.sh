#!/bin/bash

# Submit the requested MCore, non-reasoning VLM suite for saved SFT checkpoints.
set -euo pipefail

EXPERIMENT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${EXPERIMENT_DIR}/experiment_config.sh"

PROJECT_ROOT=${NANO_V35_PROJECT_ROOT}
MEGATRON_SRC_DEFAULT=${PROJECT_ROOT}/megatron-lm
VLMEVALKIT_SRC=${PROJECT_ROOT}/VLMEvalKitMcore
MODEL_NAME=${SFT_MODEL_NAME}
MODEL_SIZE=30_3b
MODEL_TYPE=hybrid
EVAL_MODE=mcore
EVAL_MODE_REASONING=""
USER_NAME=${SLURM_JOB_USER:-${USER}}

ADD_CONVERSION=${ADD_CONVERSION:-true}
SKIP_EXISTING_RESULTS=${SKIP_EXISTING_RESULTS:-false}
DRY_RUN=${DRY_RUN:-false}
BENCHMARKS_DIR_SUFFIX=${BENCHMARKS_DIR_SUFFIX:-reasoning-off}
GROUP_BENCHMARKS=false

MCORE_CONTAINER_IMAGE=${MCORE_CONTAINER_IMAGE:-/lustre/fsw/portfolios/llmservice/users/${USER_NAME}/docker/containers/pytorch25.11-moe-avlm-editable-energon-super-eval.sqsh}
MCORE_PYTHON_VENV=${MCORE_PYTHON_VENV:-}
if [[ "${MCORE_DISABLE_TRITON_OVERLAY:-false}" == true ]]; then
    MCORE_TRITON_PYTHONPATH=""
else
    MCORE_TRITON_PYTHONPATH=${MCORE_TRITON_PYTHONPATH:-/lustre/fsw/portfolios/llmservice/users/${USER_NAME}/docker/overlays/triton35_site/usr/local/lib/python3.12/dist-packages}
fi
MCORE_CONVERSION_TYPE=CONVERT_TO_TP_1
MCORE_TP1_DIR=tp_1
SLURM_ACCOUNT=${SLURM_ACCOUNT:-nemotron_omni_vision}
PARTITIONS_OVERRIDE=${PARTITIONS_OVERRIDE:-batch_block1}
SLURM_NODELIST=${EVAL_SLURM_NODELIST:-}
if [[ "${EVAL_ALLOW_POOL0:-false}" != true && -z "$SLURM_NODELIST" ]]; then
    pool1_nodes=$(sinfo -N -h -p batch_block1 -o '%N' | awk '/^pool1-/' | sort -u | paste -sd,)
    SLURM_NODELIST=$(scontrol show hostlistsorted "$pool1_nodes")
fi

# OCRBenchV2 produces both the requested EN and CN scores in one evaluation.
# The three ScreenSpot names are aggregate datasets over their registered splits.
BENCHMARK_GROUPS_NORMAL=(
    "AI2D_TEST"
    "CV-Bench-2D"
    "ChartQA_TEST"
    "DocVQA_VAL"
    "InfoVQA_VAL"
    "MMLongBench_DOC"
    "MMMU_DEV_VAL"
    "MathVista_MINI"
    "OCRBench"
    "OCRBenchV2"
    "RefCOCO"
    "ScreenSpot"
    "ScreenSpot_Pro"
    "ScreenSpot_v2"
    "TextVQA_VAL"
    "TreeBench"
    "Video-MME"
)
BENCHMARK_GROUPS_REASONING=()

# Optionally select a comma-separated subset without editing this file.
if [[ -n "${BENCHMARKS_OVERRIDE:-}" ]]; then
    IFS=',' read -ra BENCHMARK_GROUPS_NORMAL <<< "$BENCHMARKS_OVERRIDE"
fi

DEFAULT_ITERATIONS=(3284 5000 6435)
if [[ $# -gt 0 ]]; then
    ITERATIONS=("$@")
else
    ITERATIONS=("${DEFAULT_ITERATIONS[@]}")
fi

requires_openai_key=false
for benchmark in "${BENCHMARK_GROUPS_NORMAL[@]}"; do
    case "$benchmark" in
        MathVista_MINI|MMLongBench_DOC*) requires_openai_key=true ;;
    esac
done

if [[ -z "${OPENAI_API_KEY:-}" && "$requires_openai_key" == true ]]; then
    if [[ "$DRY_RUN" == true ]]; then
        OPENAI_API_KEY=dry-run
    else
        echo "ERROR: OPENAI_API_KEY is required for standard MathVista and MMLongBench scoring." >&2
        echo "Export it before launching this suite." >&2
        exit 1
    fi
fi
OPENAI_API_KEY=${OPENAI_API_KEY:-not-required}

cd "${VLMEVALKIT_SRC}"
source eval/lib.sh

flatten_groups_to_benchmarks
init_partitions
init_model_paths
init_derived_paths
init_eval_args

validate_model_name
validate_openai_key
validate_benchmarks_configured
validate_inference_paths
validate_checkpoint_dir

mkdir -p "${MODEL_CKPT_DIR_TP_1}"

for iteration in "${ITERATIONS[@]}"; do
    folder_name=$(format_iteration "$iteration")
    if [[ ! -d "${MODEL_CKPT_DIR}/${folder_name}" ]]; then
        echo "ERROR: Checkpoint does not exist: ${MODEL_CKPT_DIR}/${folder_name}" >&2
        exit 1
    fi
    submit_jobs_for_iteration "$folder_name"
done
