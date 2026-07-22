#!/bin/bash

# Install the Nano v3.5 VLM training stack into a writable Pyxis container.
# This mirrors examples/multimodal/super/Dockerfile.super, but is executable
# through Slurm on clusters where Docker/BuildKit is not exposed on login nodes.
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=${MAX_JOBS:-16}

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "Expected an aarch64 build node, found $(uname -m)" >&2
    exit 1
fi

rm -rf /opt/megatron-lm
apt-get update
apt-get install -y --no-install-recommends \
    autojump \
    bash \
    bash-builtins \
    bmon \
    build-essential \
    curl \
    default-jre \
    gdb \
    gettext \
    git \
    git-lfs \
    htop \
    libfabric-dev \
    net-tools \
    python-is-python3 \
    python3-dev \
    python3-pip \
    rsync \
    software-properties-common \
    strace \
    sudo \
    tmux \
    unzip \
    vim \
    wget \
    zip \
    zsh

wget -q https://github.com/mikefarah/yq/releases/download/v4.27.5/yq_linux_arm64 -O /usr/bin/yq
chmod +x /usr/bin/yq
/usr/bin/yq --version

unset PIP_CONSTRAINT
# Ubuntu's python3-blinker package has no pip RECORD. Install a pip-owned copy
# first so Flask can upgrade it without trying to uninstall the Debian package.
python -m pip install --no-cache-dir --ignore-installed 'blinker>=1.9.0'
python -m pip install --no-cache-dir \
    accelerate \
    albumentations \
    black==24.4.2 \
    blobfile \
    boto3 \
    braceexpand \
    click \
    coverage \
    darker \
    datasets \
    debugpy \
    dm-tree \
    einops \
    einops-exts \
    fairscale \
    fire \
    flake8==7.1.0 \
    flask \
    flask-restful \
    ftfy \
    isort==5.13.2 \
    librosa \
    mistral-common \
    modelcards \
    mypy \
    nltk \
    nvidia-pytriton \
    packaging \
    py-spy \
    pydantic \
    pylint==3.2.6 \
    pynvml \
    pytest \
    pytest-cov \
    pytest-random-order \
    pytest_asyncio \
    pytest_mock \
    sentencepiece \
    setuptools==69.5.1 \
    tiktoken \
    timm \
    tokenizers \
    torch_tb_profiler \
    tqdm \
    'transformers<5.0.0' \
    wandb \
    webdataset \
    wrapt \
    yapf \
    zarr \
    "tensorstore>=0.1.82"

# Internal packages are kept separate so public packages continue to resolve
# from the normal PyPI index.
python -m pip install --no-cache-dir \
    --index-url https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple \
    one-logger one-logger-utils

TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0" \
    python -m pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/fanshiqing/grouped_gemm@v1.1.4

NVTE_FRAMEWORK=pytorch \
    python -m pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/NVIDIA/TransformerEngine.git@v2.13

work_dir=$(mktemp -d /tmp/nano-v35-container-build.XXXXXX)
git clone --depth 1 https://github.com/Dao-AILab/causal-conv1d.git "${work_dir}/causal-conv1d"
CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    python -m pip install --no-cache-dir --no-build-isolation "${work_dir}/causal-conv1d"

git clone --branch v2.3.0 --depth 1 https://github.com/state-spaces/mamba.git "${work_dir}/mamba"
MAMBA_FORCE_BUILD=TRUE \
    python -m pip install --no-cache-dir --no-build-isolation --no-deps "${work_dir}/mamba"

python -m pip install --no-cache-dir flash-attn
python -m pip install --no-cache-dir git+https://github.com/openai/CLIP.git
python -m pip install --no-cache-dir --no-deps mmf 'open_clip_torch' 'open-flamingo[eval]'
python -m pip install --no-cache-dir 'git+https://github.com/NVIDIA/Megatron-Energon.git@bef8be243#egg=megatron-energon[av_decode]'
python -m pip install --no-cache-dir --upgrade 'nemo_toolkit[asr]'
python -m pip uninstall --yes multi-storage-client || true
python -m pip install --no-cache-dir 'multi-storage-client>=0.34.0'
python -m pip install --no-cache-dir boto3==1.43.0 botocore==1.43.0
# Reassert the Megatron/VLMEval-compatible major version after optional packages
# have had a chance to resolve their own transformer dependencies.
python -m pip install --no-cache-dir 'transformers<5.0.0'

python - <<'PY'
import importlib.metadata as metadata
import platform

import causal_conv1d
import flash_attn
import grouped_gemm
import mamba_ssm
import megatron.energon
import torch
import transformer_engine.pytorch
import transformers
import triton

assert platform.machine() == "aarch64", platform.machine()
assert torch.version.cuda == "13.0", (torch.__version__, torch.version.cuda)
assert tuple(map(int, triton.__version__.split(".")[:2])) >= (3, 5), triton.__version__
assert int(transformers.__version__.split(".", 1)[0]) < 5, transformers.__version__
assert torch.cuda.is_available()

print("architecture:", platform.machine())
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("triton:", triton.__version__)
print("transformer-engine:", metadata.version("transformer-engine"))
print("mamba-ssm:", metadata.version("mamba-ssm"))
print("causal-conv1d:", metadata.version("causal-conv1d"))
print("flash-attn:", metadata.version("flash-attn"))
print("megatron-energon:", metadata.version("megatron-energon"))
print("gpu:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))
print("cuda smoke:", (torch.ones(8, device="cuda") + 1).sum().item())
PY

rm -rf "${work_dir}" /root/.cache/pip /tmp/pip-* /tmp/tmp*
apt-get clean
rm -rf /var/lib/apt/lists/*
