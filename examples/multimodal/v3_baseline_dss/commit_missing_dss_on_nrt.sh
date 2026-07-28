#!/usr/bin/env bash

set -euo pipefail

# Historical helper for the five post-mapping uploads recorded in
# DSS_OPERATIONS_LOG.csv. All five @v0 uploads completed on 2026-07-23.
# Check DSS before rerunning: dataset commits are not a cache refresh command.

export NVDATASET_GROUPID=omni_vision
export EDT_TEMP_UPLOAD=/lustre/fsw/portfolios/llmservice/users/matthieul/edt_temp_upload

NRT_CLUSTER=oci-nrt-cs-001

CAPRL_JSONL=/lustre/fsw/portfolios/llmservice/users/arushig/av_data_gen/qwen35/reasoning_video_0623_cleaned/caprl_video_178k_dense_temporal_captions.jsonl
INTERNVID_JSONL=/lustre/fsw/portfolios/llmservice/users/arushig/av_data_gen/qwen35/reasoning_video_0623_cleaned/internvid_dense_temporal_captions.jsonl
HDVILA_JSONL=/lustre/fsw/portfolios/llmservice/users/arushig/av_data_gen/qwen35/reasoning_video_0623_cleaned/hdvila_hopchain_qa_520k.jsonl

LLAVA_MEDIA=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/vids/LLaVA-Video-178K
INTERNVID_MEDIA=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_audio/users/arushig/datasets/video_data/internvid

JSONL_PATHS=(
  "$CAPRL_JSONL"
  "$INTERNVID_JSONL"
  "$HDVILA_JSONL"
)

MEDIA_PATHS=(
  "$LLAVA_MEDIA"
  "$INTERNVID_MEDIA"
)

for command_name in edt energon; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Error: required command '$command_name' is not available." >&2
    exit 1
  fi
done

for jsonl_path in "${JSONL_PATHS[@]}"; do
  if [[ ! -f "$jsonl_path" ]]; then
    echo "Error: JSONL does not exist: $jsonl_path" >&2
    exit 1
  fi
done

for media_path in "${MEDIA_PATHS[@]}"; do
  if [[ ! -d "$media_path" ]]; then
    echo "Error: media directory does not exist: $media_path" >&2
    exit 1
  fi
done

mkdir -p "$EDT_TEMP_UPLOAD"

for jsonl_path in "${JSONL_PATHS[@]}"; do
  if [[ ! -f "${jsonl_path}.idx" ]]; then
    echo "Preparing Energon index for $jsonl_path"
    energon prepare "$jsonl_path"
  else
    echo "Using existing Energon index: ${jsonl_path}.idx"
  fi
done

edt commit \
  "$CAPRL_JSONL" \
  nano_v14_49k_caprl_video_178k_dense_temporal_captions@v0 \
  --src-cluster-name "$NRT_CLUSTER" \
  -y

edt commit \
  "$INTERNVID_JSONL" \
  nano_v14_49k_internvid_dense_temporal_captions@v0 \
  --src-cluster-name "$NRT_CLUSTER" \
  -y

edt commit \
  "$HDVILA_JSONL" \
  nano_v14_49k_hdvila_hopchain_qa_520k@v0 \
  --src-cluster-name "$NRT_CLUSTER" \
  -y

edt commit \
  "$LLAVA_MEDIA" \
  nano_v14_49k_LLaVA-Video-178K@v0 \
  --src-cluster-name "$NRT_CLUSTER" \
  -y

edt commit \
  "$INTERNVID_MEDIA" \
  nano_v14_49k_internvid@v0 \
  --src-cluster-name "$NRT_CLUSTER" \
  -y

echo "All five DSS upload jobs were submitted."
echo "HDVILA media was not submitted because nano_v14_49k_hdvila@v0 already exists."
