FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    ninja-build \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install OpenGL and X11 libraries separately to handle potential conflicts
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install additional GUI libraries if needed
RUN apt-get update && apt-get install -y \
    libglu1-mesa \
    libxi6 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libxdamage1 \
    libxtst6 \
    libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Create conda environment with Python 3.11
RUN conda create -n anycam python=3.11 -y
ENV CONDA_DEFAULT_ENV=anycam
ENV PATH=/opt/conda/envs/anycam/bin:$PATH

# Activate environment and install PyTorch with CUDA 12.4 support
RUN /bin/bash -c "source activate anycam && \
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124"

# Install CUDA toolkit for compilation
RUN /bin/bash -c "source activate anycam && \
    conda install -c nvidia cuda-toolkit -y"

# Set working directory
WORKDIR /workspace

# Clone the anycam repository
# RUN git clone --recursive https://github.com/Brummi/anycam.git
RUN git clone --recursive https://github.com/bmahlbrand/anycam.git

# Change to anycam directory
WORKDIR /workspace/anycam

# Copy and install requirements
RUN /bin/bash -c "source activate anycam && \
    pip install -r requirements.txt && pip install pillow matplotlib numpy"

# Make download script executable and download pretrained models
RUN chmod +x download_checkpoints.sh && \
    /bin/bash -c "source activate anycam && \
    ./download_checkpoints.sh anycam_seq8"

# Copy test script and make it executable
COPY test_installation.sh /workspace/
COPY process_video.sh /workspace/
ARG INCUBATOR_VER=unknown

COPY create_comparison_video.py /workspace/
COPY create_comparison_video.sh /workspace/
COPY create_simple_video.py /workspace/
COPY create_simple_video.sh /workspace/
RUN chmod +x /workspace/test_installation.sh /workspace/process_video.sh /workspace/create_comparison_video.sh /workspace/create_simple_video.sh

# Create a startup script that activates the environment
RUN echo '#!/bin/bash\nsource activate anycam\nexec "$@"' > /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]

# Expose port for rerun.io web viewer (optional)
EXPOSE 9090

