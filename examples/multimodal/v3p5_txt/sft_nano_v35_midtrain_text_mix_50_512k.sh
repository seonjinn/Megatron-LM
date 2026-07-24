#!/usr/bin/env bash
set -euo pipefail

REMOTE_WORKTREE="${REMOTE_WORKTREE:-/lustre/fsw/portfolios/llmservice/users/guyueh/sft-code/megatron-lm-nano-3.5-omni}"
TRAIN_SCRIPT="${REMOTE_WORKTREE}/examples/multimodal/v3p5_txt/sft_nano_v35_text_512k.sh"

MODEL_NAME="${MODEL_NAME:-vlm2-branch-nano-3.5-sft-saffron-narwhal-mix50-from-midtrain-100B-lc-lr2e-5}"
DATA_TRAIN="${DATA_TRAIN:-/lustre/fsw/portfolios/llmservice/users/guyueh/sft-data/omni_sft_pipeline/mix_50_offline_packed/openai_messages.offline_packed.yaml}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/guyueh/checkpoints/omni_sft_pipeline/midtrain-100B-lc-llava-wrapper-mtp2-tp8-ep8-mix50-20260707-100107}"
ALLOW_LLM_ONLY_CHECKPOINT="${ALLOW_LLM_ONLY_CHECKPOINT:-0}"

# Keep the launch fully iteration-driven, even if the submit shell has sample
# schedule variables set from a previous run.
while IFS='=' read -r name _; do
    if [[ "${name}" == *_SAMPLES ]]; then
        unset "${name}"
    fi
done < <(env)

cd "${REMOTE_WORKTREE}"

sbatch \
    --nodes=16 \
    --job-name="${MODEL_NAME}" \
    --export=ALL,MODEL_NAME="${MODEL_NAME}",DATA_TRAIN="${DATA_TRAIN}",BASE_MODEL_PATH="${BASE_MODEL_PATH}",ALLOW_LLM_ONLY_CHECKPOINT="${ALLOW_LLM_ONLY_CHECKPOINT}",ENABLE_ONLINE_PACKING=0,GBS=32,TRAIN_ITERS=6000,LR_WARMUP_ITERS=150,LR_DECAY_ITERS=5850 \
    "${TRAIN_SCRIPT}"
