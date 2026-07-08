#!/bin/bash

# Shared paths and run names for the base-Nano-v3.5 VLM experiment.
EXPERIMENT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OCI_NRT_DIR=$(cd -- "${EXPERIMENT_DIR}/../.." && pwd)
source "${OCI_NRT_DIR}/cluster_config.sh"

EXPERIMENT_ID=${EXPERIMENT_ID:-"nano_v35_base3p5_combined_omni_16k_vlm_only"}

BASE_HF_CKPT_DIR=${BASE_HF_CKPT_DIR:-"${NANO_V35_RESOURCES}/checkpoints/base_nano3p5_hf"}
BASE_MCORE_CKPT_DIR=${BASE_MCORE_CKPT_DIR:-"${WORKSPACE}/checkpoints/${EXPERIMENT_ID}/base_llm_mcore_tp2_ep32"}
BASE_VLM_CKPT_DIR=${BASE_VLM_CKPT_DIR:-"${WORKSPACE}/checkpoints/${EXPERIMENT_ID}/base_vlm_radio_v4_tp2_ep32"}

VISION_PRETRAIN_MODEL_NAME=${VISION_PRETRAIN_MODEL_NAME:-"${EXPERIMENT_ID}_vision_adapter_pretrain"}
VISION_PRETRAIN_CKPT_DIR=${VISION_PRETRAIN_CKPT_DIR:-"${WORKSPACE}/output/${VISION_PRETRAIN_MODEL_NAME}/checkpoints"}
SFT_MODEL_NAME=${SFT_MODEL_NAME:-"${EXPERIMENT_ID}_sft"}
SFT_DATA_RECIPE=${SFT_DATA_RECIPE:-"${EXPERIMENT_DIR}/sft_combined_omni_16k_vlm_only.yaml"}

export EXPERIMENT_DIR OCI_NRT_DIR EXPERIMENT_ID
export BASE_HF_CKPT_DIR BASE_MCORE_CKPT_DIR BASE_VLM_CKPT_DIR
export VISION_PRETRAIN_MODEL_NAME VISION_PRETRAIN_CKPT_DIR
export SFT_MODEL_NAME SFT_DATA_RECIPE
