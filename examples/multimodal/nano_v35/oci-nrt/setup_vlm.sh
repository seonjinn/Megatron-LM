#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/cluster_config.sh"

REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)
VISION_PRETRAIN_DATA="${REPO_ROOT}/examples/multimodal/v3_baseline/pretrain_vision_adaptor_recipe.yaml"
VLM_SFT_DATA="${REPO_ROOT}/examples/multimodal/v3_baseline/1377_video_text.yaml"

python "${SCRIPT_DIR}/validate_vlm_resources.py" \
    --tokenizer "${TOKENIZER_MODEL}" \
    --hf-checkpoint "${HF_CKPT_DIR}" \
    --radio-checkpoint "${VISION_CKPT_DIR}" \
    --container "${CONTAINER_IMAGE}"

required_paths=(
    "${VISION_PRETRAIN_DATA}"
    "${VLM_SFT_DATA}"
)

failed=0
for path in "${required_paths[@]}"; do
    if [[ ! -r "${path}" ]]; then
        echo "Missing or unreadable: ${path}" >&2
        failed=1
    fi
done
if [[ "${failed}" -ne 0 ]]; then
    exit 1
fi

echo "Nano v3.5 VLM resources are ready."
echo "  container: ${CONTAINER_IMAGE}"
echo "  tokenizer: ${TOKENIZER_MODEL}"
echo "  HF checkpoint: ${HF_CKPT_DIR}"
echo "  RADIO checkpoint: ${VISION_CKPT_DIR}"
echo "  vision pretrain recipe: ${VISION_PRETRAIN_DATA}"
echo "  VLM SFT recipe: ${VLM_SFT_DATA}"

if [[ ! -f "${LM_MCORE_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "  pending: HF-to-MCore conversion -> ${LM_MCORE_DIR}"
fi
if [[ ! -f "${OUTPUT_CKPT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "  pending: LLM+RADIO merge -> ${OUTPUT_CKPT_DIR}"
fi
