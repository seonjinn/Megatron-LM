#!/bin/bash

# H100
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/matthieul/docker/megatron-dev-img-05142025-pytorch-dev-te-cd37379-energon-710-mamba-fix-vlmeval.sqsh"
# For audio/video development:
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/containers/megatron-dev-img-05142025-pytorch-dev-te-cd37379-editable-energon-mamba-fix-vlmeval-av.sqsh"

if [[ $1 == "moe" ]]; then
    CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/containers/nm6_hybrid_moe_yash_07_17_vlm.sqsh"
    CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/containers/pytorch25.06-moe-avlm.sqsh"
fi

# Set partitions based on hostname
if [[ $(hostname) == *"draco-oci"* ]]; then
    PARTITIONS="interactive,batch_singlenode,backfill_singlenode,backfill_block1,backfill_block3,backfill_block4,batch_block1,batch_block2,batch_block3,batch_block4"
elif [[ $(hostname) == *"cw-dfw"* ]]; then
    PARTITIONS="interactive,batch"
elif [[ $(hostname) == *"oci-nrt"* ]]; then
    PARTITIONS="interactive,batch_block1,backfill,batch_singlenode"
else
    PARTITIONS="interactive"
fi

echo "Using container image: ${CONTAINER_IMAGE}"

srun -p ${PARTITIONS} -A llmservice_fm_vision -N 1 --pty \
    --container-image ${CONTAINER_IMAGE} \
    --container-mounts "/lustre,/home" \
    --gpus 8 \
    --exclusive \
    --job-name "llmservice_fm_vision-megatron-dev:interactive" \
    -t 4:00:00 \
    bash -l