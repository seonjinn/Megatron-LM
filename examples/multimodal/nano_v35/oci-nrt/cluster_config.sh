#!/bin/bash

# Destination-cluster defaults shared by the Nano v3.5 VLM bootstrap and training jobs.
USER_NAME=${SLURM_JOB_USER:-${USER:-$(whoami)}}
OCI_NRT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=$(cd -- "${OCI_NRT_DIR}/../../../../.." && pwd)

NANO_V35_PROJECT_ROOT=${NANO_V35_PROJECT_ROOT:-"${DEFAULT_PROJECT_ROOT}"}
NANO_V35_RESOURCES=${NANO_V35_RESOURCES:-"${NANO_V35_PROJECT_ROOT}/resources"}
WORKSPACE=${WORKSPACE:-"/lustre/fsw/portfolios/llmservice/users/${USER_NAME}/workspace"}

CONTAINER_IMAGE=${CONTAINER_IMAGE:-"$(dirname -- "${NANO_V35_PROJECT_ROOT}")/docker/containers/pytorch25.11-moe-avlm-editable-energon-super-triton35.sqsh"}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-"/lustre"}

TOKENIZER_MODEL=${TOKENIZER_MODEL:-"${NANO_V35_RESOURCES}/tokenizer/nemotron_3_nano_30b_a3b_tokenizer"}
HF_CKPT_DIR=${HF_CKPT_DIR:-"${NANO_V35_RESOURCES}/checkpoints/nano_v35_checkpoint_hf"}
VISION_CKPT_DIR=${VISION_CKPT_DIR:-${RADIO_CKPT_DIR:-"${NANO_V35_RESOURCES}/checkpoints/c-radio-v4-h-rc2-tp2"}}

MCORE_CKPT_DIR=${MCORE_CKPT_DIR:-"${WORKSPACE}/checkpoints/nano_v35_llm_mcore_tp2_ep32_mtpfix"}
LM_MCORE_DIR=${LM_MCORE_DIR:-"${MCORE_CKPT_DIR}/torch"}
OUTPUT_CKPT_DIR=${OUTPUT_CKPT_DIR:-"${WORKSPACE}/checkpoints/nano_v35_vlm/nano_v35_moe_tp2_ep32_radio_v4_mtpfix"}

# MSC is not required for these local data recipes. Export it only when a caller supplies one.
if [[ -n "${MSC_CONFIG:-}" ]]; then
    export MSC_CONFIG
else
    unset MSC_CONFIG || true
fi
