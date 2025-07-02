#!/bin/bash

# AnyCam Video Processing Script
# This script provides an easy interface to process videos with AnyCam

set -e  # Exit on any error

# Default values
INPUT_PATH=""
OUTPUT_PATH="/workspace/outputs"
MODEL_PATH="pretrained_models/anycam_seq8"
VISUALIZE="true"
BA_REFINEMENT="true"
EXPORT_COLMAP="false"
FPS=""
RERUN_MODE="spawn"

# Function to display help
show_help() {
    echo "AnyCam Video Processing Script"
    echo "=============================="
    echo ""
    echo "Usage: $0 [OPTIONS] <input_video>"
    echo ""
    echo "Arguments:"
    echo "  <input_video>     Path to input video file (required)"
    echo ""
    echo "Options:"
    echo "  -o, --output DIR        Output directory (default: /workspace/outputs)"
    echo "  -m, --model PATH        Model path (default: pretrained_models/anycam_seq8)"
    echo "  -v, --visualize BOOL    Enable visualization (default: true)"
    echo "  -r, --refinement BOOL   Enable bundle adjustment refinement (default: true)"
    echo "  -c, --colmap           Export to COLMAP format"
    echo "  -f, --fps FPS          Subsample video to specified FPS"
    echo "  --rerun-mode MODE      Rerun mode: spawn|connect (default: spawn)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  $0 /workspace/data/video.mp4"
    echo ""
    echo "  # Fast processing without refinement"
    echo "  $0 -r false /workspace/data/video.mp4"
    echo ""
    echo "  # Export to COLMAP"
    echo "  $0 -c /workspace/data/video.mp4"
    echo ""
    echo "  # Subsample high framerate video"
    echo "  $0 -f 10 /workspace/data/video.mp4"
    echo ""
    echo "  # Remote server setup (connect to existing rerun viewer)"
    echo "  $0 --rerun-mode connect /workspace/data/video.mp4"
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

# Check if input file exists
if [ ! -f "$INPUT_PATH" ]; then
    echo "Error: Input file does not exist: $INPUT_PATH"
    exit 1
fi

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    echo "Available models:"
    ls -la pretrained_models/ 2>/dev/null || echo "  No pretrained models found"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Build the command
CMD="python anycam/scripts/anycam_demo.py"
CMD="$CMD ++input_path=$INPUT_PATH"
CMD="$CMD ++model_path=$MODEL_PATH"
CMD="$CMD ++output_path=$OUTPUT_PATH"
CMD="$CMD ++visualize=$VISUALIZE"
CMD="$CMD ++ba_refinement=$BA_REFINEMENT"

if [ "$EXPORT_COLMAP" = "true" ]; then
    CMD="$CMD ++export_colmap=true"
fi

if [ -n "$FPS" ]; then
    CMD="$CMD ++fps=$FPS"
fi

if [ "$RERUN_MODE" = "connect" ]; then
    CMD="$CMD ++rerun_mode=connect"
fi

# Display configuration
echo "AnyCam Video Processing"
echo "======================"
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
echo ""

# Ask for confirmation
read -p "Proceed with processing? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo "Starting processing..."
echo "Command: $CMD"
echo ""

# Change to anycam directory
cd /workspace/anycam

# Run the command
eval $CMD

echo ""
echo "Processing complete!"
echo "Results saved to: $OUTPUT_PATH"
