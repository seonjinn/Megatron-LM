#!/bin/bash

# Add the VLMEvalKit dependencies to the completed ARM training image.
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
unset PIP_CONSTRAINT

BUILD_DIR=$(mktemp -d /tmp/nano-v35-eval-build.XXXXXX)
trap 'rm -rf "${BUILD_DIR}"' EXIT

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "Expected an aarch64 build node, found $(uname -m)" >&2
    exit 1
fi

# PyPI does not publish Decord wheels for Linux aarch64. Build its CPU decoder
# against Ubuntu's FFmpeg libraries; this is sufficient for VLMEvalKit's video
# dataset readers and avoids relying on the host's NVDEC driver libraries.
apt-get update
apt-get install -y --no-install-recommends \
    cmake \
    libavcodec-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev

python -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
    accelerate \
    'datasets[audio]==3.6.0' \
    pysubs2 \
    moviepy \
    dotenv \
    einops \
    google-genai \
    gradio \
    huggingface_hub \
    imageio \
    ipdb \
    json_repair \
    math-verify \
    matplotlib \
    nltk \
    numpy \
    omegaconf \
    openai \
    'opencv-python>=4.7.0.72' \
    openpyxl \
    pandas \
    pillow \
    portalocker \
    protobuf \
    pylatexenc==2.10 \
    python-dotenv \
    qwen_vl_utils \
    requests \
    rich \
    scikit-learn \
    sentencepiece \
    setuptools \
    sty \
    sympy \
    tabulate \
    tiktoken \
    timeout-decorator \
    timm \
    tqdm \
    'transformers<5.0.0' \
    typing_extensions \
    validators \
    xlsxwriter \
    distance \
    apted \
    lxml \
    zss \
    Levenshtein \
    editdistance \
    jieba \
    Polygon3 \
    PyMuPDF

git clone --recursive https://github.com/dmlc/decord.git "${BUILD_DIR}/decord"
git -C "${BUILD_DIR}/decord" checkout d2e56190286ae394032a8141885f76d5372bd44b
# Decord's pinned source predates FFmpeg 5/6: AVBSFContext moved to its own
# public header and av_find_best_stream now returns a const AVCodec pointer.
sed -i '/#include <libavcodec\/avcodec.h>/a #include <libavcodec/bsf.h>' \
    "${BUILD_DIR}/decord/src/video/ffmpeg/ffmpeg_common.h"
sed -i 's/    AVCodec \*dec;/    const AVCodec *dec;/' \
    "${BUILD_DIR}/decord/src/video/video_reader.cc"
cmake \
    -S "${BUILD_DIR}/decord" \
    -B "${BUILD_DIR}/decord/build" \
    -DUSE_CUDA=0 \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}/decord/build" --parallel "${MAX_JOBS:-$(nproc)}"
python -m pip install --no-cache-dir --no-build-isolation --no-deps \
    "${BUILD_DIR}/decord/python"

if [[ "${INSTALL_VOICEBENCH:-0}" -eq 1 ]]; then
    python -m pip install --no-cache-dir --no-deps \
        absl-py \
        immutabledict \
        langdetect \
        loguru==0.7.2 \
        qa_metrics==0.2.17 \
        contractions \
        textsearch \
        anyascii \
        pyahocorasick
    GIT_LFS_SKIP_SMUDGE=1 python -m pip install --no-cache-dir --no-deps \
        'VoiceBench @ git+ssh://git@gitlab-master.nvidia.com:12051/yifanp/VoiceBench.git@vlmeval'
fi

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python -m nltk.downloader -d /usr/local/share/nltk_data punkt_tab

python - <<'PY'
import platform

import av
import cv2
import decord
import fitz
import mamba_ssm
import openai
import qwen_vl_utils
import torch
import transformer_engine.pytorch
import transformers
import triton

assert platform.machine() == "aarch64", platform.machine()
assert torch.version.cuda == "13.0", (torch.__version__, torch.version.cuda)
assert tuple(map(int, triton.__version__.split(".")[:2])) >= (3, 5), triton.__version__
assert torch.cuda.is_available()

print("architecture:", platform.machine())
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("triton:", triton.__version__)
print("transformers:", transformers.__version__)
print("opencv:", cv2.__version__)
print("decord:", decord.__version__)
print("gpu:", torch.cuda.get_device_name(0))
PY

rm -rf /root/.cache/pip /tmp/pip-* /tmp/tmp*
apt-get clean
rm -rf /var/lib/apt/lists/*
