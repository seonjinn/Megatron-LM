#!/bin/bash

# Shared paths and run names for the base-Nano-v3.5 VLM experiment.
EXPERIMENT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CODE_DIR=$(cd -- "${EXPERIMENT_DIR}/../../../.." && pwd)
USER_NAME=${SLURM_JOB_USER:-${USER:-$(whoami)}}

NANO_V35_PROJECT_ROOT=${NANO_V35_PROJECT_ROOT:-"$(dirname -- "${CODE_DIR}")"}
NANO_V35_RESOURCES=${NANO_V35_RESOURCES:-"${NANO_V35_PROJECT_ROOT}/resources"}
WORKSPACE=${WORKSPACE:-"/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/${USER_NAME}"}

CONTAINER_IMAGE=${CONTAINER_IMAGE:-"/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/containers/pytorch25.11-moe-avlm-editable-energon-super-triton35.sqsh"}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-"/lustre"}

NVDATASET_CACHE_DIR=${NVDATASET_CACHE_DIR:-"/home/svc-dss/cache/nemotron"}
if [[ -n "${NVDATASET_CACHE_DIR}" ]]; then
    export NVDATASET_CACHE_DIR
    case ",${CONTAINER_MOUNTS}," in
        *,"${NVDATASET_CACHE_DIR}:${NVDATASET_CACHE_DIR}",*) ;;
        *) CONTAINER_MOUNTS+=",${NVDATASET_CACHE_DIR}:${NVDATASET_CACHE_DIR}" ;;
    esac
fi

# MSC is not required by the local or DSS recipes. Export it only when a caller
# supplies one.
if [[ -n "${MSC_CONFIG:-}" ]]; then
    export MSC_CONFIG
else
    unset MSC_CONFIG || true
fi

EXPERIMENT_ID=${EXPERIMENT_ID:-"nano_v35_base3p5_combined_omni_16k_vlm_only"}

DERIVED_VLM_TOKENIZER_DIR=${DERIVED_VLM_TOKENIZER_DIR:-"${NANO_V35_RESOURCES}/tokenizer/nano_v35_sft_v10_closethink_unmask_orig6k_vlm"}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-"${DERIVED_VLM_TOKENIZER_DIR}"}

BASE_HF_CKPT_DIR=${BASE_HF_CKPT_DIR:-"/lustre/fsw/portfolios/llmservice/users/venkats/training_actual_0603/nano_n3_post/checkpoints/nano-3.5-sft-v10-closethink-unmask-orig6k-from-midtrain-100B-lc-lr2e-5/eval/iter_0006000/hf"}
BASE_MCORE_CKPT_DIR=${BASE_MCORE_CKPT_DIR:-"${WORKSPACE}/checkpoints/${EXPERIMENT_ID}/base_llm_mcore_tp2_ep32"}
BASE_VLM_CKPT_DIR=${BASE_VLM_CKPT_DIR:-"${WORKSPACE}/checkpoints/${EXPERIMENT_ID}/base_vlm_radio_v4_tp2_ep32"}
VISION_CKPT_DIR=${VISION_CKPT_DIR:-${RADIO_CKPT_DIR:-"/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/checkpoints/c-radio-v4-h-rc2-tp2"}}

OUTPUT_BASE=${OUTPUT_BASE:-"${WORKSPACE}/workspace/output"}
VISION_PRETRAIN_MODEL_NAME=${VISION_PRETRAIN_MODEL_NAME:-"${EXPERIMENT_ID}_vision_adapter_pretrain"}
VISION_PRETRAIN_CKPT_DIR=${VISION_PRETRAIN_CKPT_DIR:-"${OUTPUT_BASE}/${VISION_PRETRAIN_MODEL_NAME}/checkpoints"}
SFT_MODEL_NAME=${SFT_MODEL_NAME:-"${EXPERIMENT_ID}_sft"}
SFT_DATA_RECIPE=${SFT_DATA_RECIPE:-"${CODE_DIR}/examples/multimodal/v3_baseline/sft_combined_omni_16k_vlm_only_webbrowse_dss.yaml"}

export CODE_DIR EXPERIMENT_DIR EXPERIMENT_ID USER_NAME
export NANO_V35_PROJECT_ROOT NANO_V35_RESOURCES WORKSPACE
export CONTAINER_IMAGE CONTAINER_MOUNTS NVDATASET_CACHE_DIR
export DERIVED_VLM_TOKENIZER_DIR TOKENIZER_MODEL
export BASE_HF_CKPT_DIR BASE_MCORE_CKPT_DIR BASE_VLM_CKPT_DIR VISION_CKPT_DIR
export OUTPUT_BASE
export VISION_PRETRAIN_MODEL_NAME VISION_PRETRAIN_CKPT_DIR
export SFT_MODEL_NAME SFT_DATA_RECIPE
