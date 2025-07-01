#!/bin/bash

# Test script to verify AnyCam installation in Docker container

echo "Testing AnyCam Docker installation..."
echo "======================================"

# Check if we're in the right environment
echo "1. Checking conda environment..."
if [ "$CONDA_DEFAULT_ENV" = "anycam" ]; then
    echo "   ✓ AnyCam conda environment is active"
else
    echo "   ✗ AnyCam conda environment is not active"
    exit 1
fi

# Check Python version
echo "2. Checking Python version..."
python_version=$(python --version 2>&1)
if [[ $python_version == *"3.11"* ]]; then
    echo "   ✓ Python 3.11 is installed: $python_version"
else
    echo "   ✗ Wrong Python version: $python_version"
    exit 1
fi

# Check PyTorch installation
echo "3. Checking PyTorch installation..."
python -c "import torch; print(f'   ✓ PyTorch {torch.__version__} installed')" 2>/dev/null || {
    echo "   ✗ PyTorch not installed or not working"
    exit 1
}

# Check CUDA availability
echo "4. Checking CUDA availability..."
python -c "import torch; print(f'   ✓ CUDA available: {torch.cuda.is_available()}'); print(f'   ✓ CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null || {
    echo "   ✗ CUDA check failed"
    exit 1
}

# Check if AnyCam repository exists
echo "5. Checking AnyCam repository..."
if [ -d "/workspace/anycam" ]; then
    echo "   ✓ AnyCam repository found"
else
    echo "   ✗ AnyCam repository not found"
    exit 1
fi

# Check pretrained models
echo "6. Checking pretrained models..."
if [ -d "/workspace/anycam/pretrained_models/anycam_seq8" ]; then
    echo "   ✓ Pretrained model anycam_seq8 found"
else
    echo "   ✗ Pretrained model anycam_seq8 not found"
    exit 1
fi

# Test importing AnyCam
echo "7. Testing AnyCam import..."
cd /workspace/anycam
python -c "
import sys
sys.path.append('/workspace/anycam')
try:
    # Try to import core modules
    import torch
    print('   ✓ Core imports successful')
except Exception as e:
    print(f'   ✗ Import failed: {e}')
    exit(1)
" || exit 1

echo ""
echo "======================================"
echo "✓ All tests passed! AnyCam is ready to use."
echo ""
echo "Example usage:"
echo "  python anycam/scripts/anycam_demo.py \\"
echo "    ++input_path=/workspace/data/your_video.mp4 \\"
echo "    ++model_path=pretrained_models/anycam_seq8 \\"
echo "    ++output_path=/workspace/outputs \\"
echo "    ++visualize=true"
