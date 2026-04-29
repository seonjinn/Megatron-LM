#!/bin/bash

CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/containers/pytorch25.06-moe-avlm-editable-energon.sqsh"

ACCOUNT="llmservice_fm_vision"

# Set partitions based on hostname
if [[ $(hostname) == *"draco-oci"* ]]; then
    PARTITIONS="interactive,batch_singlenode,backfill_singlenode,backfill_block1,backfill_block3,backfill_block4,batch_block1,batch_block2,batch_block3,batch_block4"
elif [[ $(hostname) == *"cw-dfw"* ]]; then
    PARTITIONS="interactive,batch"
elif [[ $(hostname) == *"oci-nrt"* ]]; then
    PARTITIONS="interactive,batch_block1,backfill,batch_singlenode"
elif [[ $(hostname) == *"lbd-lax"* ]]; then
    PARTITIONS="interactive"
    ACCOUNT="llmservice_nemotron_super"
elif [[ $(hostname) == *"nsc-svg"* ]]; then
    PARTITIONS="batch"
    CONTAINER_IMAGE="/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/containers/pytorch25.06-moe-avlm-editable-energon-super.sqsh/pytorch25.06-moe-avlm-editable-energon-super.sqsh"
else
    PARTITIONS="interactive"
fi

MOUNTS="/lustre,/home"
[[ -d /scratch ]] && MOUNTS="/scratch,/lustre,/home"

echo "Using container image: ${CONTAINER_IMAGE}"
echo "Using partitions: ${PARTITIONS}"
echo "Using account: ${ACCOUNT}"
echo "Using mounts: ${MOUNTS}"

srun -p ${PARTITIONS} -A ${ACCOUNT} -N 1 --pty \
    --container-image ${CONTAINER_IMAGE} \
    --container-mounts ${MOUNTS} \
    --gpus 8 \
    --exclusive \
    --job-name "${ACCOUNT}-megatron-dev:interactive" \
    -t 4:00:00 \
    bash -l
