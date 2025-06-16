# ------------------------------------------------------------------------------
# 1) BASE IMAGE: CUDA 12.0 • Ubuntu 22.04
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# install “software-properties-common” so we can add the toolchain PPA
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    # install the newer libstdc++6 from the PPA
    apt-get install -y --no-install-recommends libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# 2) SETUP MINICONDA INSTALL LOCATION
# ------------------------------------------------------------------------------
ENV CONDA_DIR=/root/miniconda3
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# ------------------------------------------------------------------------------
# 3) INSTALL APT DEPENDENCIES + MINICONDA
# ------------------------------------------------------------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        unzip sudo git g++ libglm-dev libglew-dev libglfw3-dev \
        libgles2-mesa-dev zlib1g-dev wget cmake vim libxi6 libgconf-2-4 \
        libxkbcommon-x11-0 \
        libxrender1 libxext6 libx11-6 libsm6 libxrender-dev && \
        \

    # Download & install Miniconda (quietly, into /root/miniconda3)
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" && \
    rm /tmp/miniconda.sh && \
    \

    apt-get clean && rm -rf /var/lib/apt/lists/* 


# ------------------------------------------------------------------------------
# 5) CLONE THE Infinigen REPOSITORY
# ------------------------------------------------------------------------------
RUN mkdir -p /opt && \
    git clone https://github.com/princeton-vl/infinigen.git /opt/infinigen

# Set working directory
WORKDIR /opt/infinigen

# ------------------------------------------------------------------------------
# 6) INSTALL Infinigen INTO THE “infinigen” ENV VIA ITS PIP
#    → We directly call the env’s pip binary so that packages go into that env.
# ------------------------------------------------------------------------------

RUN conda init bash && \
    . ~/.bashrc && \
    conda create --name infinigen python=3.11 -y && \
    conda activate infinigen && \
    # pip install -e ".[dev]"
    conda run -n infinigen  \
    pip install "numpy<2.0" Cython landlab==2.6.0 && \
    conda run -n infinigen python setup.py install && \
    pip install jinja2

# ------------------------------------------------------------------------------
# 7) AT RUNTIME, MAKE THE “infinigen” ENV THE DEFAULT
#    by prepending its bin to PATH. Now `python`, `conda`, etc. will
#    automatically be the ones from the infinigen env.
# ------------------------------------------------------------------------------
ENV PATH="${CONDA_DIR}/envs/infinigen/bin:${PATH}"

# Use bash‑login so that any future “conda” calls (e.g. `conda list`) work
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["bash"]
