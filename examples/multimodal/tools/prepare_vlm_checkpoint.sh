#!/bin/bash

MEGATRON_SRC=/lustre/fsw/portfolios/llmservice/users/${USER}/megatron-lm
export PYTHONPATH=${PYTHONPATH}:${MEGATRON_SRC}

# Ensure that the command line arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <checkpoint_dir> <tp> <iteration>"
    exit 1
fi

# Remove trailing slash from CHECKPOINT_DIR if present
CHECKPOINT_DIR=${1%/}
TP=$2
ITER=$3

# Check if the checkpoint directory exists
CHECKPOINT_TORCH_DIR="${CHECKPOINT_DIR}/torch"
if [ ! -d "${CHECKPOINT_TORCH_DIR}" ]; then
    echo "Checkpoint directory ${CHECKPOINT_TORCH_DIR} does not exist"
    exit 1
fi

# Check if the tp is valid
if [ "$TP" -ne 4 ] && [ "$TP" -ne 8 ]; then
    echo "Invalid tp: $TP"
    exit 1
fi

# Patch embeddings
python examples/multimodal/tools/prepare_llm.py --input-dir $CHECKPOINT_TORCH_DIR --output-dir ${CHECKPOINT_DIR}_patched --tp $TP --iter $ITER

# Combine with vision backbone
python examples/multimodal/tools/replace_llm_backbone.py --input-dir ${CHECKPOINT_DIR}_patched --output-dir ${CHECKPOINT_DIR}_vlm --tp $TP --iter $ITER

# Write the iteration to a file
echo $ITER > ${CHECKPOINT_DIR}_vlm/latest_checkpointed_iteration.txt