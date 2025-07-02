#!/bin/bash

# AnyCam Docker Video Processing Script
# Run AnyCam video processing from outside the container

set -e  # Exit on any error

# Default values
INPUT_PATH=""
OUTPUT_PATH="$(pwd)/outputs"
MODEL_PATH="pretrained_models/anycam_seq8"
VISUALIZE="true"
BA_REFINEMENT="true"
EXPORT_COLMAP="false"
FPS=""
RERUN_MODE="spawn"
CONTAINER_NAME="anycam:latest"

# Function to display help
show_help() {
    echo "AnyCam Docker Video Processing Script"
    echo "===================================="
    echo ""
    echo "Usage: $0 [OPTIONS] <input_video>"
    echo ""
    echo "Arguments:"
    echo "  <input_video>     Path to input video file (required)"
    echo ""
    echo "Options:"
    echo "  -o, --output DIR        Output directory (default: ./outputs)"
    echo "  -m, --model PATH        Model path (default: pretrained_models/anycam_seq8)"
    echo "  -v, --visualize BOOL    Enable visualization (default: true)"
    echo "  -r, --refinement BOOL   Enable bundle adjustment refinement (default: true)"
    echo "  -c, --colmap           Export to COLMAP format"
    echo "  -f, --fps FPS          Subsample video to specified FPS"
    echo "  --rerun-mode MODE      Rerun mode: spawn|connect (default: spawn)"
    echo "  --container NAME       Docker container name (default: anycam:latest)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  $0 ./data/video.mp4"
    echo ""
    echo "  # Fast processing without refinement"
    echo "  $0 -r false ./data/video.mp4"
    echo ""
    echo "  # Export to COLMAP"
    echo "  $0 -c ./data/video.mp4"
    echo ""
    echo "  # Subsample high framerate video"
    echo "  $0 -f 10 ./data/video.mp4"
    echo ""
    echo "  # Custom output directory"
    echo "  $0 -o ./my_results ./data/video.mp4"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -v|--visualize)
            VISUALIZE="$2"
            shift 2
            ;;
        -r|--refinement)
            BA_REFINEMENT="$2"
            shift 2
            ;;
        -c|--colmap)
            EXPORT_COLMAP="true"
            shift
            ;;
        -f|--fps)
            FPS="$2"
            shift 2
            ;;
        --rerun-mode)
            RERUN_MODE="$2"
            shift 2
            ;;
        --container)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            if [ -z "$INPUT_PATH" ]; then
                INPUT_PATH="$1"
            else
                echo "Multiple input files specified. Only one is supported."
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if input path is provided
if [ -z "$INPUT_PATH" ]; then
    echo "Error: Input video path is required."
    echo ""
    show_help
    exit 1
fi

# Convert to absolute paths
INPUT_PATH=$(realpath "$INPUT_PATH")
OUTPUT_PATH=$(realpath "$OUTPUT_PATH")

# Check if input file exists
if [ ! -f "$INPUT_PATH" ]; then
    echo "Error: Input file does not exist: $INPUT_PATH"
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if the container image exists
if ! docker image inspect "$CONTAINER_NAME" &> /dev/null; then
    echo "Error: Docker image '$CONTAINER_NAME' not found."
    echo "Please build it first with: ./build_docker.sh"
    exit 1
fi

# Check if GPU support is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Running without GPU support."
    GPU_FLAGS=""
else
    GPU_FLAGS="--gpus all"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Get container paths
CONTAINER_INPUT="/workspace/data/$(basename "$INPUT_PATH")"
CONTAINER_OUTPUT="/workspace/outputs"

# Build the AnyCam command
ANYCAM_CMD="cd /workspace/anycam && python anycam/scripts/anycam_demo.py"
ANYCAM_CMD="$ANYCAM_CMD ++input_path=$CONTAINER_INPUT"
ANYCAM_CMD="$ANYCAM_CMD ++model_path=$MODEL_PATH"
ANYCAM_CMD="$ANYCAM_CMD ++output_path=$CONTAINER_OUTPUT"
ANYCAM_CMD="$ANYCAM_CMD ++visualize=$VISUALIZE"
ANYCAM_CMD="$ANYCAM_CMD ++ba_refinement=$BA_REFINEMENT"

if [ "$EXPORT_COLMAP" = "true" ]; then
    ANYCAM_CMD="$ANYCAM_CMD ++export_colmap=true"
fi

if [ -n "$FPS" ]; then
    ANYCAM_CMD="$ANYCAM_CMD ++fps=$FPS"
fi

if [ "$RERUN_MODE" = "connect" ]; then
    ANYCAM_CMD="$ANYCAM_CMD ++rerun_mode=connect"
fi

# Display configuration
echo "AnyCam Docker Video Processing"
echo "=============================="
echo "Input:        $INPUT_PATH"
echo "Output:       $OUTPUT_PATH"
echo "Model:        $MODEL_PATH"
echo "Visualize:    $VISUALIZE"
echo "Refinement:   $BA_REFINEMENT"
echo "Export COLMAP: $EXPORT_COLMAP"
if [ -n "$FPS" ]; then
    echo "FPS:          $FPS"
fi
echo "Rerun mode:   $RERUN_MODE"
echo "Container:    $CONTAINER_NAME"
echo ""

# Ask for confirmation
read -p "Proceed with processing? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo "Starting Docker container and processing..."
echo ""

# Run the Docker container with the AnyCam command
docker run $GPU_FLAGS -it --rm \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$(dirname "$INPUT_PATH"):/workspace/data" \
    -v "$OUTPUT_PATH:/workspace/outputs" \
    -p 9090:9090 \
    "$CONTAINER_NAME" \
    bash -c "$ANYCAM_CMD"

echo ""
echo "Processing complete!"
echo "Results saved to: $OUTPUT_PATH"
