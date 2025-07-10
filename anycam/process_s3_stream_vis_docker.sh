#!/usr/bin/env bash
# process_s3_stream_vis_docker.sh
#
# Docker wrapper to run test_s3_stream_vis.py inside the AnyCam environment
# Streams frames from an S3 video, processes with AnyCam + TSDF, and outputs a PLY.
#
# Usage:
#   ./process_s3_stream_vis_docker.sh --bucket BUCKET --key KEY [OPTIONS]
#
# Options:
#   --bucket        S3 bucket name (required)
#   --key           S3 object key for video (required)
#   --num_frames N  Number of frames to stream (default: 50)
#   --output FILE   Local output PLY file path (default: output.ply)
#   --container IMG Docker image name (default: anycam-hub:latest)
#   --help          Show this help message

set -e

# Defaults
CONTAINER_NAME="docker.io/library/anycam:latest"
BUCKET=""
KEY=""
NUM_FRAMES=50
OUTPUT="output.ply"

# Help
show_help() {
  cat << EOF
Usage: $0 --bucket BUCKET --key KEY [OPTIONS]

Required:
  --bucket        S3 bucket name
  --key           S3 object key for video (e.g., path/to/video.mp4)

Options:
  --num_frames N  Number of frames to stream (default: $NUM_FRAMES)
  --output FILE   Output PLY file path (default: $OUTPUT)
  --container IMG Docker image (default: $CONTAINER_NAME)
  --help          Show this help
EOF
  exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --bucket)
      BUCKET="$2"; shift 2;;
    --key)
      KEY="$2"; shift 2;;
    --num_frames)
      NUM_FRAMES="$2"; shift 2;;
    --output)
      OUTPUT="$2"; shift 2;;
    --container)
      CONTAINER_NAME="$2"; shift 2;;
    --help)
      show_help;;
    *)
      echo "Unknown option: $1"; show_help;;
  esac
done

# Validate required args
if [[ -z "$BUCKET" || -z "$KEY" ]]; then
  echo "Error: --bucket and --key are required." >&2
  show_help
fi

# Check Docker
if ! command -v docker &> /dev/null; then
  echo "Error: docker not found." >&2; exit 1; fi

# Build Docker command
# Mount current workspace and pass AWS env vars
DOCKER_CMD=(docker run --rm \
  --env-file .env
  --gpus all \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "$CONTAINER_NAME" \
  python /workspace/test_s3_stream_vis.py \
  --bucket "$BUCKET" \
  --key "$KEY" \
  --num_frames "$NUM_FRAMES" \
  --output "$OUTPUT"
)

echo "Running in Docker image: $CONTAINER_NAME"
echo " -> ${DOCKER_CMD[*]}"

# Execute
"${DOCKER_CMD[@]}"

echo "Done. Output saved to $OUTPUT"
