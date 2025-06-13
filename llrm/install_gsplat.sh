#!/bin/bash

# Post-installation script for gsplat
# Run this after creating the conda environment

echo "Installing gsplat with proper CUDA environment..."

# Ensure CUDA environment variables are set
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set compilation flags to avoid CUDA compatibility issues
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
export FORCE_CUDA=1

echo "CUDA_HOME: $CUDA_HOME"
echo "Installing gsplat..."

# Try installing with no build isolation first
pip install git+https://github.com/nerfstudio-project/gsplat --no-build-isolation

if [ $? -eq 0 ]; then
    echo "✅ gsplat installed successfully!"
else
    echo "❌ gsplat installation failed. Trying alternative approach..."
    
    # Try building from source with specific flags
    pip install --upgrade pip setuptools wheel
    pip install git+https://github.com/nerfstudio-project/gsplat --no-cache-dir --verbose
    
    if [ $? -eq 0 ]; then
        echo "✅ gsplat installed successfully with alternative approach!"
    else
        echo "❌ gsplat installation failed. You may need to install it manually."
        echo "Try running: FORCE_CUDA=1 pip install git+https://github.com/nerfstudio-project/gsplat --no-build-isolation"
    fi
fi
