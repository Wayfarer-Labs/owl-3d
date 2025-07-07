#!/bin/bash

# process_video_hub_docker.sh
# 
# Run the AnyCam hub processor inside Docker container
#
# Usage:
#   ./process_video_hub_docker.sh INPUT_VIDEO OUTPUT_DIR [OPTIONS]
#
# Example:
#   ./process_video_hub_docker.sh /path/to/video.mp4 ./outputs --ba_refinement --max_frames 100

set -e  # Exit on any error

# Default values
CONTAINER_NAME="docker.io/library/anycam:latest"
INPUT_VIDEO=""
OUTPUT_DIR=""
EXTRA_ARGS=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
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

# Help function
show_help() {
    cat << EOF
AnyCam Hub Processor Docker Runner

Usage: $0 INPUT_VIDEO OUTPUT_DIR [OPTIONS]

Arguments:
  INPUT_VIDEO         Path to input video file
  OUTPUT_DIR          Directory for output files

Options:
  --ba_refinement     Enable bundle adjustment refinement
  --max_frames N      Process only first N frames
  --resize_height N   Resize frames to height N (maintains aspect ratio)
    --batch_size N      Number of frames to process per video (batch size)
  --container NAME    Docker container name (default: anycam-hub:latest)
  --help              Show this help message

Examples:
  # Basic processing
  $0 video.mp4 ./outputs

  # With bundle adjustment and frame limit
  $0 video.mp4 ./outputs --ba_refinement --max_frames 50

  # Resize frames and use custom container
  $0 video.mp4 ./outputs --resize_height 480 --container my-anycam:v1

EOF
}

# Parse command line arguments
parse_args() {
    if [ $# -lt 2 ]; then
        print_error "Missing required arguments"
        show_help
        exit 1
    fi

    INPUT_VIDEO="$1"
    OUTPUT_DIR="$2"
    shift 2

    # Parse optional arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --ba_refinement)
                EXTRA_ARGS="$EXTRA_ARGS --ba_refinement"
                shift
                ;;
            --max_frames)
                if [[ -n $2 && $2 =~ ^[0-9]+$ ]]; then
                    EXTRA_ARGS="$EXTRA_ARGS --max_frames $2"
                    shift 2
                else
                    print_error "Invalid value for --max_frames: $2"
                    exit 1
                fi
                ;;
            --resize_height)
                if [[ -n $2 && $2 =~ ^[0-9]+$ ]]; then
                    EXTRA_ARGS="$EXTRA_ARGS --resize_height $2"
                    shift 2
                else
                    print_error "Invalid value for --resize_height: $2"
                    exit 1
                fi
                ;;
            --batch_size)
                if [[ -n $2 && $2 =~ ^[0-9]+$ ]]; then
                    EXTRA_ARGS="$EXTRA_ARGS --batch_size $2"
                    shift 2
                else
                    print_error "Invalid value for --batch_size: $2"
                    exit 1
                fi
                ;;
            --container)
                if [[ -n $2 ]]; then
                    CONTAINER_NAME="$2"
                    shift 2
                else
                    print_error "Missing value for --container"
                    exit 1
                fi
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate inputs
validate_inputs() {
    # Check if input video exists
    if [[ ! -f "$INPUT_VIDEO" ]]; then
        print_error "Input video file does not exist: $INPUT_VIDEO"
        exit 1
    fi

    # Get absolute paths
    INPUT_VIDEO=$(realpath "$INPUT_VIDEO")
    OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"

    print_info "Input video: $INPUT_VIDEO"
    print_info "Output directory: $OUTPUT_DIR"
    print_info "Container: $CONTAINER_NAME"
    
    if [[ -n "$EXTRA_ARGS" ]]; then
        print_info "Extra arguments: $EXTRA_ARGS"
    fi
}

# Check Docker availability
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi

    # Check if the Docker image exists
    if ! docker image inspect "$CONTAINER_NAME" &> /dev/null; then
        print_error "Docker image '$CONTAINER_NAME' not found"
        print_info "Please build the Docker image first or use a different container name"
        exit 1
    fi

    print_success "Docker environment validated"
}

# Check for GPU support
check_gpu() {
    GPU_FLAGS=""
    
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            GPU_FLAGS="--gpus all"
            print_success "NVIDIA GPU detected - enabling GPU support"
        else
            print_warning "nvidia-smi found but GPU not accessible"
        fi
    else
        print_warning "No NVIDIA GPU support detected - running on CPU"
    fi
}

# Run the Docker container
run_docker() {
    local input_dir=$(dirname "$INPUT_VIDEO")
    local input_filename=$(basename "$INPUT_VIDEO")
    local script_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
    # Derive base name without extension for side-by-side
    local base_name="${input_filename%.*}"
    local container_input="/workspace/data/$input_filename"
    local container_output="/workspace/outputs"
    local container_script="/workspace/scripts/anycam_hub_processor.py"
    
    print_info "Starting Docker container..."
    print_info "Script directory: $script_dir"
    
    # Build the Docker command
    local docker_cmd=(
        docker run --rm
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        -v "$input_dir:/workspace/data:ro"
        -v "$OUTPUT_DIR:/workspace/outputs"
        -v "$script_dir:/workspace/scripts:ro"
        -w /workspace
    )
    
    # Add GPU support if available
    if [[ -n "$GPU_FLAGS" ]]; then
        docker_cmd+=($GPU_FLAGS)
    fi
    
    # Add container name and run processor, then generate side-by-side inside container
    docker_cmd+=(
        "$CONTAINER_NAME"
        /bin/bash -c "python '$container_script' --input_video '$container_input' --output_dir '$container_output' $EXTRA_ARGS && \
                      bash /workspace/scripts/create_side_by_side.sh '$container_output' '$base_name' '$container_output/side_by_side.mp4'"
    )
    
    print_info "Running command: ${docker_cmd[*]}"
    echo
    
    # Execute the Docker command
    "${docker_cmd[@]}"
    
    if [[ $? -eq 0 ]]; then
        print_success "Processing completed successfully!"
        print_info "Results saved to: $OUTPUT_DIR"
        
        # List output files
        echo
        print_info "Generated files:"
        ls -la "$OUTPUT_DIR"
    else
        print_error "Processing failed!"
        exit 1
    fi
}

# Main execution
main() {
    echo "========================================"
    echo "  AnyCam Hub Processor Docker Runner"
    echo "========================================"
    echo
    
    parse_args "$@"
    validate_inputs
    check_docker
    check_gpu
    
    echo
    print_info "Starting video processing..."
    run_docker
    
    echo
    print_success "All done!"
}

# Run main function with all arguments
main "$@"
