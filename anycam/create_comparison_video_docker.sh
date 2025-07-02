#!/bin/bash

# AnyCam Docker Video Comparison Script
# Creates side-by-side videos from outside the container

set -e

# Default values
INPUT_VIDEO=""
OUTPUT_DIR=""
OUTPUT_VIDEO="comparison_video.mp4"
FPS=20
COLORMAP="viridis"
LIST_FILES=false
CONTAINER_NAME="anycam:latest"

show_help() {
    echo "AnyCam Docker Video Comparison Tool"
    echo "==================================="
    echo ""
    echo "Usage: $0 [OPTIONS] <input_video_or_frames> <anycam_output_dir>"
    echo ""
    echo "Arguments:"
    echo "  <input_video_or_frames>  Video file (.mp4, .avi, etc.) or directory with frame images"
    echo "  <anycam_output_dir>      AnyCam output directory containing depth maps"
    echo ""
    echo "Options:"
    echo "  -o, --output FILE     Output video filename (default: comparison_video.mp4)"
    echo "  --fps FPS            Output video FPS (default: 30)"
    echo "  --colormap NAME      Depth colormap: viridis, plasma, inferno, magma, jet"
    echo "  --list-files         List found files and exit"
    echo "  --container NAME     Docker container name (default: anycam:latest)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Basic usage with video file"
    echo "  $0 data/video.mp4 outputs/"
    echo ""
    echo "  # Basic usage with frames directory"  
    echo "  $0 data/frames/ outputs/"
    echo ""
    echo "  # Custom output and colormap"
    echo "  $0 -o my_comparison.mp4 --colormap plasma data/video.mp4 outputs/"
    echo ""
    echo "  # Debug file matching"
    echo "  $0 --list-files data/video.mp4 outputs/"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -o|--output)
            OUTPUT_VIDEO="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --colormap)
            COLORMAP="$2"
            shift 2
            ;;
        --list-files)
            LIST_FILES=true
            shift
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
            if [ -z "$INPUT_VIDEO" ]; then
                INPUT_VIDEO="$1"
            elif [ -z "$OUTPUT_DIR" ]; then
                OUTPUT_DIR="$1"
            else
                echo "Too many arguments"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# Check required arguments
if [ -z "$INPUT_VIDEO" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Both input video and output directory are required"
    show_help
    exit 1
fi

# Convert to absolute paths
INPUT_VIDEO=$(realpath "$INPUT_VIDEO")
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
OUTPUT_VIDEO=$(realpath "$OUTPUT_VIDEO")

# Check if files exist
if [ ! -e "$INPUT_VIDEO" ]; then
    echo "Error: Input does not exist: $INPUT_VIDEO"
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory does not exist: $OUTPUT_DIR"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

if ! docker image inspect "$CONTAINER_NAME" &> /dev/null; then
    echo "Error: Docker image '$CONTAINER_NAME' not found"
    echo "Please build it first with: ./build_docker.sh"
    exit 1
fi

echo "AnyCam Docker Video Comparison"
echo "============================="
echo "Input:        $INPUT_VIDEO"
echo "Output Dir:   $OUTPUT_DIR"
echo "Output Video: $OUTPUT_VIDEO"
echo "FPS:          $FPS"
echo "Colormap:     $COLORMAP"
echo "Container:    $CONTAINER_NAME"
echo ""

if [ "$LIST_FILES" = true ]; then
    echo "Listing files..."
else
    echo "Creating comparison video..."
fi

# Run Docker container
if [ -f "$INPUT_VIDEO" ]; then
    # If input is a file, mount it with proper filename
    docker run --rm \
        -v "$INPUT_VIDEO:/workspace/$(basename "$INPUT_VIDEO")" \
        -v "$OUTPUT_DIR:/workspace/output_dir" \
        -v "$(dirname "$OUTPUT_VIDEO"):/workspace/output" \
        -v "$(pwd)/create_simple_video.py:/workspace/create_simple_video.py" \
        "$CONTAINER_NAME" \
        bash -c "cd /workspace && python3 /workspace/create_simple_video.py '/workspace/$(basename "$INPUT_VIDEO")' /workspace/output_dir -o '/workspace/output/$(basename "$OUTPUT_VIDEO")' --fps $FPS --colormap $COLORMAP$([ "$LIST_FILES" = true ] && echo " --list-files" || echo "")"
else
    # If input is a directory, mount it normally
    docker run --rm \
        -v "$INPUT_VIDEO:/workspace/input_frames" \
        -v "$OUTPUT_DIR:/workspace/output_dir" \
        -v "$(dirname "$OUTPUT_VIDEO"):/workspace/output" \
        -v "$(pwd)/create_simple_video.py:/workspace/create_simple_video.py" \
        "$CONTAINER_NAME" \
        bash -c "cd /workspace && python3 /workspace/create_simple_video.py /workspace/input_frames /workspace/output_dir -o '/workspace/output/$(basename "$OUTPUT_VIDEO")' --fps $FPS --colormap $COLORMAP$([ "$LIST_FILES" = true ] && echo " --list-files" || echo "")"
fi

# Check if the video was created and copy it to the host
if [ "$LIST_FILES" = false ]; then
    if [ -f "$OUTPUT_VIDEO" ]; then
        echo ""
        echo "✅ Comparison video created successfully!"
        echo "📹 Output: $OUTPUT_VIDEO"
        echo ""
        echo "The video shows:"
        echo "  • Left side: Original frames"
        echo "  • Right side: Depth maps (colormap: $COLORMAP)"
    else
        echo ""
        echo "⚠️  Video file not created: $OUTPUT_VIDEO"
        echo "Check the output directory for any generated files"
    fi
fi
