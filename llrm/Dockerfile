# Use NVIDIA CUDA development image with Ubuntu 22.04
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba \
    && mv bin/micromamba /usr/local/bin/ \
    && rm -rf bin

# Initialize micromamba
RUN micromamba shell init -s bash -p /opt/conda

# Set working directory
WORKDIR /workspace/llrm

# Copy environment file
COPY environment.yml .

# Create conda environment using micromamba
RUN micromamba create -f environment.yml -y && \
    micromamba clean --all --yes

# Activate environment and set up shell
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate llrm" >> ~/.bashrc

# Copy project files
COPY . .

# Install additional CUDA packages and gsplat
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate llrm && \
    export CUDA_HOME=/opt/conda && \
    export PATH=$CUDA_HOME/bin:$PATH && \
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH && \
    pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121 && \
    FORCE_CUDA=1 pip install git+https://github.com/nerfstudio-project/gsplat --no-build-isolation

# Set up entrypoint to activate conda environment
RUN echo '#!/bin/bash\nsource /opt/conda/etc/profile.d/conda.sh\nconda activate llrm\nexec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Expose any ports if needed (adjust as necessary)
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["python", "main.py", "--config", "configs/dl3dv_i512_32input_8target.yaml"]
