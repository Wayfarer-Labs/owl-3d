#!/bin/bash

# AnyCam Video Comparison Script
# Creates side-by-side videos showing original frames and depth maps

# Default values
INPUT_VIDEO=""
OUTPUT_DIR=""
OUTPUT_VIDEO="comparison_video.mp4"
FPS=30
COLORMAP="viridis"
LIST_FILES=false

show_help() {
    echo "AnyCam Video Comparison Tool"
    echo "============================"
    echo ""
    echo "Usage: $0 [OPTIONS] <input_video> <anycam_output_dir>"
    echo ""
    echo "Arguments:"
    echo "  <input_video>         Original video file or frame directory"
    echo "  <anycam_output_dir>   AnyCam output directory containing depth maps"
    echo ""
    echo "Options:"
    echo "  -o, --output FILE     Output video filename (default: comparison_video.mp4)"
    echo "  --fps FPS            Output video FPS (default: 30)"
    echo "  --colormap NAME      Depth colormap: viridis, plasma, inferno, magma, jet (default: viridis)"
    echo "  --list-files         List found files and exit"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  $0 data/video.mp4 outputs/"
    echo ""
    echo "  # Custom output filename"
    echo "  $0 -o my_comparison.mp4 data/video.mp4 outputs/"
    echo ""
    echo "  # Different colormap"
    echo "  $0 --colormap plasma data/video.mp4 outputs/"
    echo ""
    echo "  # List files to debug"
    echo "  $0 --list-files data/video.mp4 outputs/"
    echo ""
    echo "Available colormaps:"
    echo "  - viridis (blue to green to yellow)"
    echo "  - plasma (purple to pink to yellow)"
    echo "  - inferno (black to red to yellow)"
    echo "  - magma (black to purple to white)"
    echo "  - jet (blue to red, classic)"
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
    echo "Error: Both input video and output directory are required"
    echo ""
    show_help
    exit 1
fi

# Check if files exist
if [ ! -e "$INPUT_VIDEO" ]; then
    echo "Error: Input video/directory does not exist: $INPUT_VIDEO"
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory does not exist: $OUTPUT_DIR"
    exit 1
fi

# Build Python command
PYTHON_CMD="python3 create_comparison_video.py"
PYTHON_CMD="$PYTHON_CMD \"$INPUT_VIDEO\" \"$OUTPUT_DIR\""
PYTHON_CMD="$PYTHON_CMD -o \"$OUTPUT_VIDEO\""
PYTHON_CMD="$PYTHON_CMD --fps $FPS"
PYTHON_CMD="$PYTHON_CMD --colormap $COLORMAP"

if [ "$LIST_FILES" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --list-files"
fi

echo "AnyCam Video Comparison"
echo "======================"
echo "Input:        $INPUT_VIDEO"
echo "Output Dir:   $OUTPUT_DIR"
echo "Output Video: $OUTPUT_VIDEO"
echo "FPS:          $FPS"
echo "Colormap:     $COLORMAP"
echo ""

if [ "$LIST_FILES" = true ]; then
    echo "Listing files..."
    eval $PYTHON_CMD
    exit 0
fi

echo "Creating comparison video..."
eval $PYTHON_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Comparison video created successfully!"
    echo "📹 Output: $OUTPUT_VIDEO"
    echo ""
    echo "You can now view the side-by-side comparison of:"
    echo "  • Left side: Original frames"
    echo "  • Right side: Depth maps"
else
    echo ""
    echo "❌ Failed to create comparison video"
    echo ""
    echo "Troubleshooting tips:"
    echo "  1. Check if depth files exist: $0 --list-files \"$INPUT_VIDEO\" \"$OUTPUT_DIR\""
    echo "  2. Verify AnyCam output structure"
    echo "  3. Ensure Python dependencies are installed: opencv-python matplotlib numpy tqdm"
fi
