FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev \
    build-essential git cmake ninja-build \
    curl ca-certificates libgl1 tzdata \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

RUN pip install git+https://github.com/nerfstudio-project/gsplat.git
