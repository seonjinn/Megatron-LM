#!/bin/bash

CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/containers/pytorch25.06-moe-avlm-editable-energon.sqsh"

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
