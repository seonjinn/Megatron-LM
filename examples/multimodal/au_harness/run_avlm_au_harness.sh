#!/bin/bash
# =============================================================================
# AVLM + AU-Harness Evaluation Script
# =============================================================================
# Starts an AVLM server and runs AU-Harness evaluation benchmarks.
# Supports base64-encoded audio in OpenAI multimodal format.
#
# Usage:
#   bash run_avlm_au_harness.sh
# =============================================================================

# =============================================================================
# Paths Configuration
# =============================================================================

MEGATRON_PATH=/lustre/fsw/portfolios/convai/users/mmkrtchyan/projects/speechLM/megatron-lm
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH}

# Nithin's checkpoint for longform
export MODEL_PATH=/lustre/fsw/portfolios/llmservice/users/nkoluguri/workspace/output/stage2-nm_5p5_h_9b_cradio_parakeet-nodes16-seq32768-bz1024-tp4-vlm2-branch-longform/checkpoints/tp_1
export MODEL_CONFIG_YAML="/lustre/fsw/portfolios/llmservice/users/nkoluguri/workspace/output/stage2-nm_5p5_h_9b_cradio_parakeet-nodes16-seq32768-bz1024-tp4-vlm2-branch-longform/config.yaml"

# AU-Harness directory
AU_HARNESS_DIR="/lustre/fsw/portfolios/convai/users/mmkrtchyan/AU-Harness"

# =============================================================================
# Server Configuration
# =============================================================================

NUM_GPUS=1
NUM_NODES=1
PORT=8000
INFERENCE_MAX_REQUESTS=1
MICRO_BATCH_SIZE=1

# =============================================================================
# Start AVLM Server
# =============================================================================

echo "Starting AVLM server..."
echo "  MODEL_PATH: ${MODEL_PATH}"
echo "  PORT: ${PORT}"

SERVER_ARGS="--port ${PORT} --inference-max-requests ${INFERENCE_MAX_REQUESTS} --model-config ${MODEL_CONFIG_YAML}"
SCRIPT=${MEGATRON_PATH}/examples/multimodal/run_avlm_text_generation_server.py

export MKL_NUM_THREADS=1
cd ${MEGATRON_PATH}

python ${SCRIPT} \
    --load ${MODEL_PATH} \
    --tensor-model-parallel-size ${NUM_GPUS} \
    --pipeline-model-parallel-size ${NUM_NODES} \
    --use-checkpoint-args \
    --max-tokens-to-oom 12000000 \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --sound-pad-to-clip-duration \
    ${SERVER_ARGS} &

SERVER_PID=$!
echo "Server started with PID: ${SERVER_PID}"

echo "Waiting for server to initialize..."
sleep 30

# =============================================================================
# Install AU-Harness Dependencies
# =============================================================================

echo "Installing AU-Harness dependencies..."
deactivate 2>/dev/null || true
export HF_AUDIO_DECODER="soundfile"
/usr/bin/python3 -m pip install --no-cache-dir -r ${AU_HARNESS_DIR}/requirements_flexible.txt "datasets<4.0" soundfile librosa || \
python3 -m pip install --no-cache-dir -r ${AU_HARNESS_DIR}/requirements_flexible.txt "datasets<4.0" soundfile librosa

# =============================================================================
# Run AU-Harness Evaluation
# =============================================================================

echo "Starting AU-Harness evaluation..."
echo "  AU_HARNESS_DIR: ${AU_HARNESS_DIR}"

cd "${AU_HARNESS_DIR}"
/usr/bin/python3 evaluate.py --config run_configs/megatron.yaml || python3 evaluate.py --config run_configs/megatron.yaml

echo ""
echo "AU-Harness evaluation complete!"
echo "Results saved in: ${AU_HARNESS_DIR}/run_logs/"

# =============================================================================
# Cleanup
# =============================================================================

echo "Stopping server..."
kill ${SERVER_PID} 2>/dev/null || true
echo "Done!"
