#!/usr/bin/env bash

# streaming_anycam_s3_docker.sh
#
# Run the streaming_anycam_s3.py inside a Docker container
#
# Usage:
#   ./streaming_anycam_s3_docker.sh --bucket BUCKET [--prefix PREFIX] [--ext EXT] [--frame-batch-size N] [--container NAME]
#
# Example:
#   ./streaming_anycam_s3_docker.sh --bucket cod-yt-playlist-spmem-tensors --prefix raw_videos/ --ext mp4 --frame-batch-size 32

set -e

# Default values
CONTAINER_NAME="docker.io/library/anycam:latest"
BUCKET=""
PREFIX=""
EXT="mp4"
FRAME_BATCH_SIZE=50

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}
print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}
print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
Streaming AnyCam S3 Runner

Usage: $0 --bucket BUCKET [OPTIONS]

Options:
  --bucket BUCKET          S3 bucket name (required)
  --prefix PREFIX          S3 prefix/folder (default: empty)
  --ext EXT                Video extension filter (default: mp4)
  --frame-batch-size N     Number of frames per batch (default: 50)
  --container NAME         Docker container name (default: anycam:latest)
  --help                   Show this help message
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --bucket)
                BUCKET="$2"; shift 2;;
            --prefix)
                PREFIX="$2"; shift 2;;
            --ext)
                EXT="$2"; shift 2;;
            --frame-batch-size)
                FRAME_BATCH_SIZE="$2"; shift 2;;
            --container)
                CONTAINER_NAME="$2"; shift 2;;
            --help)
                show_help; exit 0;;
            *)
                print_error "Unknown option: $1"; show_help; exit 1;;
        esac
    done

    if [[ -z "$BUCKET" ]]; then
        print_error "--bucket is required"
        show_help
        exit 1
    fi
}

validate_inputs() {
    print_info "Bucket: $BUCKET"
    print_info "Prefix: $PREFIX"
    print_info "Extension: $EXT"
    print_info "Frame batch size: $FRAME_BATCH_SIZE"
    print_info "Container: $CONTAINER_NAME"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    if ! docker image inspect "$CONTAINER_NAME" &> /dev/null; then
        print_error "Docker image '$CONTAINER_NAME' not found"
        exit 1
    fi
    print_success "Docker environment validated"
}

check_gpu() {
    # Initialize GPU flags array
    GPU_FLAGS=()
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        GPU_FLAGS=(--gpus all)
        print_success "NVIDIA GPU detected - enabling GPU support"
    else
        print_warning "No NVIDIA GPU support detected - running on CPU"
    fi
}

run_docker() {
    local script_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

    print_info "Starting Docker container..."
    # Build docker command
    local docker_cmd=(
        docker run --rm
        --env-file .env
        -v "$script_dir:/workspace/scripts:ro"
        -w /workspace
    )
    # Append GPU flags if available
    if [ ${#GPU_FLAGS[@]} -gt 0 ]; then
        docker_cmd+=("${GPU_FLAGS[@]}")
    fi
    # Mount host outputs directory and pass OUTPUT_DIR to container
    docker_cmd+=(
        -v "${PWD}/outputs:/workspace/anycam/outputs"
        -e "OUTPUT_DIR=/workspace/anycam/outputs"
        "$CONTAINER_NAME"
        /bin/bash -c "python /workspace/scripts/streaming_anycam_s3.py \
            --bucket '$BUCKET' \
            --prefix '$PREFIX' \
            --ext '$EXT' \
            --frame-batch-size $FRAME_BATCH_SIZE"
    )

    print_info "Running command: ${docker_cmd[*]}"
    echo
    "${docker_cmd[@]}"

    if [[ $? -eq 0 ]]; then
        print_success "Streaming processing completed successfully!"
    else
        print_error "Streaming processing failed!"
        exit 1
    fi
}

main() {
    parse_args "$@"
    validate_inputs
    check_docker
    check_gpu
    run_docker
}

main "$@"
