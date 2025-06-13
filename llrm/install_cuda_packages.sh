#!/bin/bash

# Install packages that need special CUDA handling
echo "Installing xformers with compatible CUDA version..."
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

echo "Installing gsplat with proper environment..."
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Try installing gsplat with specific flags
FORCE_CUDA=1 pip install git+https://github.com/nerfstudio-project/gsplat --no-build-isolation

echo "Installation complete!"
