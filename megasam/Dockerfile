FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu22.04

ENV CUDA_HOME=/usr/local/cuda \
    PATH=$CUDA_HOME/bin:$PATH \
    LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

SHELL ["/bin/bash", "-c"]

# Get rid of annoying tz popup
ENV TZ=America/New_York
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y tzdata
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

RUN echo $CUDA_HOME
RUN ls $CUDA_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    git \
    sudo \
    cmake \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    vim \
    nano

# Install Miniforge as root in /root's home directory
ENV CONDA_DIR=/root/miniforge3
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p ${CONDA_DIR} && \
    rm Miniforge3-$(uname)-$(uname -m).sh

# Set environment path for root
ENV PATH="${CONDA_DIR}/bin:$PATH"

# Initialize conda for bash and disable auto activation of base
RUN conda init bash && \
    echo "conda config --set auto_activate_base false" >> ~/.bashrc

WORKDIR /mega-sam

# Clone the MegaSAM repository with submodules
RUN git clone --recursive https://github.com/mega-sam/mega-sam.git .

# Create the conda environment from environment.yml
RUN conda env create -f environment.yml

# Set up shell to use the conda environment
SHELL ["conda", "run", "-n", "mega_sam", "/bin/bash", "-c"]

# Install xformers for UniDepth model
# Download the specific xformers version mentioned in the README
RUN wget https://anaconda.org/xformers/xformers/0.0.22.post7/download/linux-64/xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2
RUN conda install xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2

# Compile the extensions for the camera tracking module
RUN cd base && python setup.py install

# Create directories for checkpoints
RUN mkdir -p checkpoints
RUN mkdir -p Depth-Anything/checkpoints
RUN mkdir -p cvd_opt

# Add conda environment activation to .bashrc
RUN echo "conda activate mega_sam" >> ~/.bashrc

# Set environment variables for debugging
ENV TORCH_USE_CUDA_DSA=1
ENV CUDA_LAUNCH_BLOCKING=1

# Set PYTHONPATH to include UniDepth
ENV PYTHONPATH="${PYTHONPATH}:/mega-sam/UniDepth"

# Download pretrained checkpoints (optional - can be done at runtime)
# Uncomment these lines if you want to download checkpoints during build
# RUN wget -O Depth-Anything/checkpoints/depth_anything_vitl14.pth \
#     https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth

# Set working directory
WORKDIR /mega-sam

# Default command
CMD ["/bin/bash"]