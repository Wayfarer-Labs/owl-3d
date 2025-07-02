#!/bin/bash

# Build the AnyCam Docker image
echo "Building AnyCam Docker image..."
docker build -t anycam:latest . --build-arg INCUBATOR_VER=$(date +%Y%m%d-%H%M%S)

echo "Build complete! You can now run the container with:"
echo "  ./run_docker.sh"
echo ""
echo "Or run it manually with:"
echo "  docker run --gpus all -it --rm -v \$(pwd)/data:/workspace/data anycam:latest"
