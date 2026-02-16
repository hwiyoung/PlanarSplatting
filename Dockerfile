# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FORCE_CUDA=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    python-is-python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/PlanarSplatting

COPY . .

RUN git submodule update --init --recursive

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt && \
    pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable" && \
    pip install --no-build-isolation submodules/diff-rect-rasterization && \
    pip install --no-build-isolation submodules/quaternion-utils && \
    pip install -e submodules/vggt/ && \
    pip install "transformers>=4.45.0,<5.0.0" supervision pycocotools

CMD ["bash"]
