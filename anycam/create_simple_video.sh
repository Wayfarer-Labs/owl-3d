#!/bin/bash

# Simple Video Creator - No Docker, No OpenCV
# Creates side-by-side videos using FFmpeg directly

set -e

# Default values
INPUT_VIDEO=""
OUTPUT_DIR=""
OUTPUT_VIDEO="comparison_video.mp4"
FPS=30
COLORMAP="viridis"
LIST_FILES=false

show_help() {
    echo "Simple Video Creator (No OpenCV)"
    echo "================================"
    echo ""
    echo "Usage: $0 [OPTIONS] <input_video_or_frames> <depth_output_dir>"
    echo ""
    echo "Arguments:"
    echo "  <input_video_or_frames>  Video file (.mp4, .avi, etc.) or directory with frame images"
    echo "  <depth_output_dir>       Directory containing depth maps"
    echo ""
    echo "Options:"
    echo "  -o, --output FILE     Output video filename (default: comparison_video.mp4)"
    echo "  --fps FPS            Output video FPS (default: 30)"
    echo "  --colormap NAME      Depth colormap: viridis, plasma, inferno, magma, jet"
    echo "  --list-files         List found files and exit"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Basic usage with video file"
    echo "  $0 video.mp4 outputs/"
    echo ""
    echo "  # Basic usage with frames directory"
    echo "  $0 frames/ outputs/"
    echo ""
    echo "  # Custom output and colormap"
    echo "  $0 -o my_comparison.mp4 --colormap plasma video.mp4 outputs/"
    echo ""
    echo "  # Debug file matching"
    echo "  $0 --list-files frames/ outputs/"
    echo ""
    echo "Requirements:"
    echo "  - Python 3 with PIL, matplotlib, numpy"
    echo "  - FFmpeg"
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
    echo "Error: Both input video/frames and depth output directory are required"
    show_help
    exit 1
fi

# Convert to absolute paths
INPUT_VIDEO=$(realpath "$INPUT_VIDEO")
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
OUTPUT_VIDEO=$(realpath "$OUTPUT_VIDEO")

# Check if input exists (file or directory)
if [ ! -e "$INPUT_VIDEO" ]; then
    echo "Error: Input does not exist: $INPUT_VIDEO"
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory does not exist: $OUTPUT_DIR"
    exit 1
fi

# Check dependencies
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "Error: FFmpeg is not installed"
    exit 1
fi

# Check Python dependencies
python3 -c "import PIL, matplotlib, numpy" 2>/dev/null || {
    echo "Error: Missing Python dependencies"
    echo "Install with: pip install pillow matplotlib numpy"
    exit 1
}

# Build command
CMD="python3 $(dirname "$0")/create_simple_video.py"
CMD="$CMD '$INPUT_VIDEO'"
CMD="$CMD '$OUTPUT_DIR'"
CMD="$CMD -o '$OUTPUT_VIDEO'"
CMD="$CMD --fps $FPS"
CMD="$CMD --colormap $COLORMAP"

if [ "$LIST_FILES" = true ]; then
    CMD="$CMD --list-files"
fi

echo "Simple Video Creator"
echo "==================="
echo "Input:        $INPUT_VIDEO"
echo "Output Dir:   $OUTPUT_DIR"
echo "Output Video: $OUTPUT_VIDEO"
echo "FPS:          $FPS"
echo "Colormap:     $COLORMAP"
echo ""

if [ "$LIST_FILES" = true ]; then
    echo "Listing files..."
else
    echo "Creating comparison video..."
fi

# Run the command
eval $CMD

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
        echo "❌ Video creation failed"
        exit 1
    fi
fi
