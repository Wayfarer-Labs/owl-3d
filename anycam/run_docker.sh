#!/bin/bash

# Check if GPU support is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Running without GPU support."
    GPU_FLAGS=""
else
    GPU_FLAGS="--gpus all"
fi

# Create data directory if it doesn't exist
mkdir -p data

# Run the AnyCam Docker container
echo "Starting AnyCam Docker container..."
docker run $GPU_FLAGS -it --rm \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    -p 9090:9090 \
    anycam:latest

echo "Container exited."
