#!/bin/bash

#SBATCH -A nemotron_n4_post
#SBATCH -p batch
#SBATCH --qos=interactive
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=4
#SBATCH --mem=64G
#SBATCH --job-name=nano35-validate-container

set -euo pipefail

IMAGE_KIND=${IMAGE_KIND:-training}
OUTPUT_DIR=${OUTPUT_DIR:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/${USER}/containers}
case "${IMAGE_KIND}" in
    training)
        IMAGE_PATH=${IMAGE_PATH:-${OUTPUT_DIR}/pytorch25.11-moe-avlm-editable-energon-super-triton35.sqsh}
        ;;
    eval)
        IMAGE_PATH=${IMAGE_PATH:-${OUTPUT_DIR}/pytorch25.11-moe-avlm-editable-energon-super-eval.sqsh}
        ;;
    *)
        echo "IMAGE_KIND must be training or eval, got: ${IMAGE_KIND}" >&2
        exit 1
        ;;
esac

test -s "${IMAGE_PATH}"

srun --kill-on-bad-exit=1 \
    --container-image="${IMAGE_PATH}" \
    --container-mounts=/lustre \
    --no-container-entrypoint \
    python -c 'import importlib.metadata as m, platform, torch, transformers, triton, transformer_engine.pytorch, mamba_ssm, causal_conv1d, grouped_gemm, flash_attn, megatron.energon; assert platform.machine() == "aarch64"; assert torch.version.cuda == "13.0"; assert tuple(map(int, triton.__version__.split(".")[:2])) >= (3, 5); assert int(transformers.__version__.split(".", 1)[0]) < 5; assert torch.cuda.is_available(); print({"arch": platform.machine(), "torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__, "transformers": transformers.__version__, "te": m.version("transformer-engine"), "mamba": m.version("mamba-ssm"), "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0), "cuda_sum": (torch.ones(8, device="cuda") + 1).sum().item()})'

if [[ "${IMAGE_KIND}" == eval ]]; then
    srun --kill-on-bad-exit=1 \
        --container-image="${IMAGE_PATH}" \
        --container-mounts=/lustre \
        --no-container-entrypoint \
        python -c 'import av, cv2, decord, fitz, openai, qwen_vl_utils, transformers; print({"transformers": transformers.__version__, "opencv": cv2.__version__, "decord": decord.__version__})'
fi
