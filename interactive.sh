#!/bin/bash

# H100
CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/containers/megatron-dev-img-05142025-pytorch-dev-te-cd37379-energon-develop-08471f7-mamba-vlmeval.sqsh"

# Set partitions based on hostname
if [[ $(hostname) == *"oci-iad"* ]]; then
    PARTITIONS="interactive,batch_singlenode,backfill_singlenode,backfill_block1,backfill_block3,backfill_block4,batch_block1,batch_block2,batch_block3,batch_block4"
elif [[ $(hostname) == *"cw-dfw"* ]]; then
    PARTITIONS="interactive,batch"
elif [[ $(hostname) == *"oci-nrt"* ]]; then
    PARTITIONS="interactive,batch_block1,backfill,batch_singlenode"
else
    PARTITIONS="interactive"
fi

srun -p ${PARTITIONS} -A llmservice_fm_vision -N 1 --pty \
    --container-image ${CONTAINER_IMAGE} \
    --container-mounts "/lustre" \
    --gpus 8 \
    --exclusive \
    --job-name "llmservice_fm_vision-megatron-dev:interactive" \
    -t 4:00:00 \
    bash -l