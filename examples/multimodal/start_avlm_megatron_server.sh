#! /bin/bash

MEGATRON_PATH=/lustre/fsw/portfolios/convai/users/mmkrtchyan/projects/speechLM/megatron-lm
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH}

# you can extract and save the model config from a distributed checkpoint by running:
# python examples/multimodal/tools/create_yaml_inference_config.py --ckpt_path "/path/to/checkpoint/iter_0000000/mp_rank_00/model_optim_rng.pt"  --output_config "/path/to/output/config.yaml"

export MODEL_PATH=/lustre/fsw/portfolios/convai/users/mmkrtchyan/avlm_checkpoints/draco-mcore-nm_5p5_h_9b-cradio-parakeet-nemo-stage2-alm-nodes16-seq16384-bz2048-tp4-vlm2-branch-1009-tp1
export MODEL_CONFIG_YAML="/lustre/fsw/portfolios/convai/users/mmkrtchyan/avlm_checkpoints/draco-mcore-nm_5p5_h_9b-cradio-parakeet-nemo-stage2-alm-nodes16-seq16384-bz2048-tp4-vlm2-branch-1009-tp1/config.yaml"


NUM_GPUS=1
NUM_NODES=1
SERVER_ARGS="--port 8000 --inference-max-requests 1 --model-config ${MODEL_CONFIG_YAML}"

SCRIPT=$MEGATRON_PATH/examples/multimodal/run_avlm_text_generation_server.py

export CUDA_DEVICE_MAX_CONNECTIONS=1 && \
cd ${MEGATRON_PATH} && \
python ${SCRIPT} \
    --load ${MODEL_PATH} \
    --tensor-model-parallel-size ${NUM_GPUS} \
    --pipeline-model-parallel-size ${NUM_NODES} \
    --use-checkpoint-args \
    --sound-pad-to-clip-duration \
    --max-tokens-to-oom 12000000 \
    --micro-batch-size 1 \
    ${SERVER_ARGS}


# Command to test the server's chat completion API with audio:
# groundtruth: "And wylder chuckled angrily, and the smali change in his pocket tinkled fiercely, as his eye glanced on the graceful captai n, who was entertaining the ladies, no doubt, very agreeably in the distance."
# AUDIO_FILE="/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/LibriTTS/test-clean/5683/32865/5683_32865_000052_000000.wav"
# curl -X POST http://localhost:8000/chat/completions -H "Content-Type: application/json" -d '{"messages": [{"role": "system", "content": "You are a helpful assistant. /no_think"}, {"role": "user", "content": "Transcribe the audio file into English text.", "audio": {"path": "'${AUDIO_FILE}'"}}], "max_tokens": 256}'

# Test the chat completion API with only text:
# curl -X POST http://localhost:8000/chat/completions -H "Content-Type: application/json" -d '{"messages": [{"role": "system", "content": "You are a helpful assistant. /no_think"}, {"role": "user", "content": "What is the capital of France?"}], "max_tokens": 256}'
