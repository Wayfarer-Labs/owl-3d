#!/bin/bash

# LLRM Docker Build and Run Script

set -e

echo "Building LLRM Docker image..."
docker build -t llrm:latest .

echo "Build complete!"
echo ""
echo "Usage options:"
echo "1. Run with default config:"
echo "   docker run --gpus all -it --rm -v \$(pwd)/output:/workspace/llrm/output llrm:latest"
echo ""
echo "2. Run with custom config:"
echo "   docker run --gpus all -it --rm -v \$(pwd)/output:/workspace/llrm/output llrm:latest python main.py --config configs/your_config.yaml"
echo ""
echo "3. Run in interactive mode:"
echo "   docker run --gpus all -it --rm -v \$(pwd):/workspace/llrm llrm:latest bash"
echo ""
echo "4. Run with custom data volume:"
echo "   docker run --gpus all -it --rm -v \$(pwd):/workspace/llrm -v /path/to/data:/data llrm:latest"
