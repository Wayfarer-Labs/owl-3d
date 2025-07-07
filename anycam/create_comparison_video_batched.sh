#!/bin/bash

# AnyCam Docker Video Comparison Script for Batched Outputs
# Creates side-by-side videos from batched AnyCam outputs

set -e

# Default values
INPUT_VIDEO=""
BATCH_OUTPUT_DIR=""
OUTPUT_VIDEO="comparison_video_batched.mp4"
COLORMAP="viridis"
LIST_FILES=false
CONTAINER_NAME="anycam:latest"
COMBINE_BATCHES=true

show_help() {
    echo "AnyCam Docker Video Comparison Tool for Batched Outputs"
    echo "======================================================="
    echo ""
    echo "Usage: $0 [OPTIONS] <input_video> <batch_output_dir>"
    echo ""
    echo "Arguments:"
    echo "  <input_video>        Original video file (.mp4, .avi, etc.)"
    echo "  <batch_output_dir>   Directory containing batch_XXX subdirectories from batched processing"
    echo ""
    echo "Options:"
    echo "  -o, --output FILE     Output video filename (default: comparison_video_batched.mp4)"
    echo "  --colormap NAME      Depth colormap: viridis, plasma, inferno, magma, jet"
    echo "  --list-files         List found batch files and exit"
    echo "  --no-combine         Don't combine batches, create separate videos for each batch"
    echo "  --container NAME     Docker container name (default: anycam:latest)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  $0 data/video.mp4 outputs/batches/"
    echo ""
    echo "  # Custom output and colormap"
    echo "  $0 -o my_comparison.mp4 --colormap plasma data/video.mp4 outputs/batches/"
    echo ""
    echo "  # Create separate videos for each batch"
    echo "  $0 --no-combine data/video.mp4 outputs/batches/"
    echo ""
    echo "  # Debug file matching"
    echo "  $0 --list-files data/video.mp4 outputs/batches/"
    echo ""
    echo "Expected directory structure:"
    echo "  batch_output_dir/"
    echo "    ├── batch_001/"
    echo "    │   ├── depth_maps/"
    echo "    │   └── other_outputs/"
    echo "    ├── batch_002/"
    echo "    │   ├── depth_maps/"
    echo "    │   └── other_outputs/"
    echo "    └── ..."
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
        --colormap)
            COLORMAP="$2"
            shift 2
            ;;
        --list-files)
            LIST_FILES=true
            shift
            ;;
        --no-combine)
            COMBINE_BATCHES=false
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
            elif [ -z "$BATCH_OUTPUT_DIR" ]; then
                BATCH_OUTPUT_DIR="$1"
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
if [ -z "$INPUT_VIDEO" ] || [ -z "$BATCH_OUTPUT_DIR" ]; then
    echo "Error: Both input video and batch output directory are required"
    show_help
    exit 1
fi

# Convert to absolute paths
INPUT_VIDEO=$(realpath "$INPUT_VIDEO")
BATCH_OUTPUT_DIR=$(realpath "$BATCH_OUTPUT_DIR")
OUTPUT_VIDEO=$(realpath "$OUTPUT_VIDEO")

# Check if files exist
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Input video does not exist: $INPUT_VIDEO"
    exit 1
fi

if [ ! -d "$BATCH_OUTPUT_DIR" ]; then
    echo "Error: Batch output directory does not exist: $BATCH_OUTPUT_DIR"
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

# Find all batch directories
echo "Scanning for batch directories..."
batch_dirs=($(find "$BATCH_OUTPUT_DIR" -type d -name "batch_*" | sort))

if [ ${#batch_dirs[@]} -eq 0 ]; then
    echo "Error: No batch directories found in $BATCH_OUTPUT_DIR"
    echo "Expected directories named batch_001, batch_002, etc."
    exit 1
fi

echo "Found ${#batch_dirs[@]} batch directories:"
for dir in "${batch_dirs[@]}"; do
    echo "  $(basename "$dir")"
done
echo ""

echo "AnyCam Docker Video Comparison for Batched Outputs"
echo "=================================================="
echo "Input Video:     $INPUT_VIDEO"
echo "Batch Dir:       $BATCH_OUTPUT_DIR"
echo "Output Video:    $OUTPUT_VIDEO"
echo "Colormap:        $COLORMAP"
echo "Container:       $CONTAINER_NAME"
echo "Combine Batches: $COMBINE_BATCHES"
echo ""

if [ "$LIST_FILES" = true ]; then
    echo "Listing batch contents..."
    for batch_dir in "${batch_dirs[@]}"; do
        echo ""
        echo "=== $(basename "$batch_dir") ==="
        find "$batch_dir" -name "*.npy" -o -name "*.png" -o -name "*.jpg" | head -10
    done
    exit 0
fi

# Function to extract frames from original video for a specific batch
extract_batch_frames() {
    local input_video="$1"
    local start_frame="$2"
    local frame_count="$3"
    local output_dir="$4"
    local batch_id="$5"
    
    echo "Extracting frames for batch $batch_id: frames $start_frame to $((start_frame + frame_count - 1))" >&2
    
    # Get video frame rate
    local fps=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$input_video" 2>/dev/null)
    
    if [ -z "$fps" ]; then
        echo "Error: Could not get frame rate from video" >&2
        return 1
    fi
    
    if [[ "$fps" == *"/"* ]]; then
        local fps_decimal=$(awk "BEGIN {printf \"%.6f\", $fps}")
    else
        local fps_decimal="$fps"
    fi
    
    # Calculate start time and duration
    local start_time=$(awk "BEGIN {printf \"%.6f\", $start_frame / $fps_decimal}")
    local duration=$(awk "BEGIN {printf \"%.6f\", $frame_count / $fps_decimal}")
    
    echo "Debug: Extracting batch frames from time $start_time for duration $duration seconds" >&2
    
    # Create batch-specific video
    local batch_video="$output_dir/batch_${batch_id}_frames.mp4"
    
    ffmpeg -ss "$start_time" -i "$input_video" -t "$duration" -c copy "$batch_video" -y >&2
    
    if [ -f "$batch_video" ]; then
        echo "$batch_video"
        return 0
    else
        echo "Error: Failed to create batch video" >&2
        return 1
    fi
}

# Function to count actual depth frames in a batch directory
count_depth_frames() {
    local batch_dir="$1"
    local depth_count=0
    
    echo "Debug: Counting depth frames in $batch_dir" >&2
    
    # Check for single depth file first (depths.npy, depth.npy, etc.)
    local single_depth_patterns=("depths.npy" "depth.npy" "all_depths.npy" "depths/depths.npy" "depths/depth.npy")
    
    for pattern in "${single_depth_patterns[@]}"; do
        local full_path="$batch_dir/$pattern"
        echo "Debug: Checking for $full_path" >&2
        if [ -f "$full_path" ]; then
            echo "Debug: Found single depth file: $full_path" >&2
            # Use Docker container with numpy to get frame count
            depth_count=$(docker run --rm \
                -v "$batch_dir:/workspace/batch_dir" \
                "$CONTAINER_NAME" \
                python3 -c "
import numpy as np
import os
import sys
try:
    file_path = '/workspace/batch_dir/$pattern'
    if not os.path.exists(file_path):
        print('0')
        sys.exit()
    depths = np.load(file_path)
    if len(depths.shape) >= 3:
        print(depths.shape[0])  # First dimension is number of frames
    else:
        print(1)  # Single frame
except Exception as e:
    print('0')
" 2>/dev/null)
            
            echo "Debug: Docker returned: '$depth_count'" >&2
            # Clean the result
            depth_count=$(echo "$depth_count" | grep -o '^[0-9]\+$' || echo "0")
            echo "Debug: Cleaned depth_count: '$depth_count'" >&2
            
            if [ -n "$depth_count" ] && [ "$depth_count" -gt 0 ] 2>/dev/null; then
                echo "$depth_count"
                return 0
            fi
        fi
    done
    
    echo "Debug: No single depth file found, checking individual files..." >&2
    
    # Fall back to counting individual depth files
    local individual_patterns=("$batch_dir/depth_*.npy" "$batch_dir/depths/depth_*.npy" "$batch_dir/*_depth.npy" "$batch_dir/depth_*.png" "$batch_dir/depths/depth_*.png" "$batch_dir/*_depth.png")
    
    for pattern in "${individual_patterns[@]}"; do
        echo "Debug: Checking pattern: $pattern" >&2
        local files=($(ls $pattern 2>/dev/null))
        if [ ${#files[@]} -gt 0 ]; then
            echo "Debug: Found ${#files[@]} individual files matching $pattern" >&2
            echo "${#files[@]}"
            return 0
        fi
    done
    
    echo "Debug: No depth files found at all" >&2
    echo "0"
}

# Function to create comparison video for a single batch
create_batch_comparison() {
    local batch_dir="$1"
    local batch_name=$(basename "$batch_dir")
    local output_file="$2"
    local batch_id="${batch_name#batch_}"  # Extract number from batch_XXX
    
    echo "Creating comparison for $batch_name..."
    
    # Count actual depth frames in this batch
    local frame_count=$(count_depth_frames "$batch_dir")
    
    # Validate frame_count is a positive integer
    if ! [[ "$frame_count" =~ ^[0-9]+$ ]] || [ "$frame_count" -eq 0 ]; then
        echo "Error: No depth frames found in $batch_name (got: '$frame_count')"
        return 1
    fi
    
    echo "Found $frame_count depth frames in $batch_name"
    
    # Extract batch number and calculate frame range
    # Use actual frame count instead of fixed batch size
    local batch_num=$((10#$batch_id))  # Force base 10 to handle leading zeros
    local start_frame=$(( (batch_num - 1) * 4 ))  # Assuming step size 4, adjust if needed
    
    # Create temporary directory for batch frames
    local temp_dir=$(mktemp -d)
    local batch_video=$(extract_batch_frames "$INPUT_VIDEO" "$start_frame" "$frame_count" "$temp_dir" "$batch_id")
    
    if [ $? -ne 0 ] || [ ! -f "$batch_video" ]; then
        echo "Failed to extract frames for $batch_name"
        rm -rf "$temp_dir"
        return 1
    fi
    
    # Verify that extracted video has the expected number of frames
    local extracted_frame_count=$(ffprobe -v quiet -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$batch_video" 2>/dev/null)
    if [ -n "$extracted_frame_count" ] && [ "$extracted_frame_count" != "$frame_count" ]; then
        echo "Warning: Expected $frame_count frames but extracted $extracted_frame_count frames for $batch_name"
    fi
    
    # Run Docker container for this batch
    docker run --rm \
        -v "$batch_video:/workspace/$(basename "$batch_video")" \
        -v "$batch_dir:/workspace/batch_output" \
        -v "$(dirname "$output_file"):/workspace/output" \
        -v "$(pwd)/create_simple_video.py:/workspace/create_simple_video.py" \
        "$CONTAINER_NAME" \
        bash -c "cd /workspace && python3 /workspace/create_simple_video.py '/workspace/$(basename "$batch_video")' /workspace/batch_output -o '/workspace/output/$(basename "$output_file")' --colormap $COLORMAP"
    
    # Clean up temporary files
    rm -rf "$temp_dir"
}

# Function to combine all batch videos into one
combine_batch_videos() {
    local temp_dir=$(mktemp -d)
    local video_list="$temp_dir/video_list.txt"
    local batch_videos=()
    
    echo "Creating individual batch videos..."
    
    # Create a video for each batch
    for i in "${!batch_dirs[@]}"; do
        local batch_dir="${batch_dirs[$i]}"
        local batch_name=$(basename "$batch_dir")
        local batch_video="$temp_dir/${batch_name}_comparison.mp4"
        
        echo "Processing $batch_name ($(($i + 1))/${#batch_dirs[@]})..."
        create_batch_comparison "$batch_dir" "$batch_video"
        
        if [ -f "$batch_video" ]; then
            batch_videos+=("$batch_video")
            echo "file '$batch_video'" >> "$video_list"
        else
            echo "Warning: Failed to create video for $batch_name"
        fi
    done
    
    if [ ${#batch_videos[@]} -eq 0 ]; then
        echo "Error: No batch videos were created successfully"
        rm -rf "$temp_dir"
        exit 1
    fi
    
    echo ""
    echo "Combining ${#batch_videos[@]} batch videos..."
    
    # Use ffmpeg to concatenate all batch videos
    if command -v ffmpeg &> /dev/null; then
        ffmpeg -f concat -safe 0 -i "$video_list" -c copy "$OUTPUT_VIDEO" -y
        
        if [ -f "$OUTPUT_VIDEO" ]; then
            echo ""
            echo "✅ Combined comparison video created successfully!"
            echo "📹 Output: $OUTPUT_VIDEO"
            echo ""
            echo "The video shows:"
            echo "  • Left side: Original frames from all batches"
            echo "  • Right side: Depth maps from all batches (colormap: $COLORMAP)"
            echo "  • Contains: ${#batch_videos[@]} batches combined"
        else
            echo "⚠️  Failed to create combined video"
        fi
    else
        echo "Warning: ffmpeg not found. Creating videos for individual batches only."
        for i in "${!batch_videos[@]}"; do
            local batch_name=$(basename "${batch_dirs[$i]}")
            local individual_output="${OUTPUT_VIDEO%.*}_${batch_name}.mp4"
            cp "${batch_videos[$i]}" "$individual_output"
            echo "📹 Individual batch video: $individual_output"
        done
    fi
    
    # Clean up
    rm -rf "$temp_dir"
}

# Main processing
if [ "$COMBINE_BATCHES" = true ]; then
    combine_batch_videos
else
    # Create separate videos for each batch
    echo "Creating separate comparison videos for each batch..."
    
    for batch_dir in "${batch_dirs[@]}"; do
        local batch_name=$(basename "$batch_dir")
        local batch_output="${OUTPUT_VIDEO%.*}_${batch_name}.mp4"
        
        create_batch_comparison "$batch_dir" "$batch_output"
        
        if [ -f "$batch_output" ]; then
            echo "✅ Created: $batch_output"
        else
            echo "⚠️  Failed: $batch_output"
        fi
    done
    
    echo ""
    echo "✅ Individual batch comparison videos created!"
fi

echo ""
echo "Processing complete!"
