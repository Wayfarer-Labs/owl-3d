#!/usr/bin/env bash
# Host script to run create_side_by_side inside AnyCam Docker container
# Usage: run_side_by_side.sh INPUT_OUTPUT_DIR BASE_NAME OUTPUT_VIDEO [FPS] [COLORMAP]

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 INPUT_OUTPUT_DIR BASE_NAME OUTPUT_VIDEO [FPS] [COLORMAP]"
  exit 1
fi

HOST_DIR="$1"   # Local path containing the outputs directory
BASE_NAME="$2"
OUTPUT_VIDEO="$3"
FPS="${4:-10.0}"
COLORMAP="${5:-JET}"

# Convert relative path to absolute for Docker mount
if [[ "$HOST_DIR" != /* ]]; then
  HOST_DIR="$(pwd)/$HOST_DIR"
fi

# Docker image name (update to your built image tag)
IMAGE="docker.io/library/anycam:latest"

# Mount the host outputs dir into container at /workspace/anycam/outputs
docker run --gpus all --rm \
  -v "${HOST_DIR}:/workspace/anycam/outputs" \
  "${IMAGE}" \
  /workspace/anycam/create_side_by_side.sh \
    outputs "$BASE_NAME" "$OUTPUT_VIDEO" "$FPS" "$COLORMAP"
