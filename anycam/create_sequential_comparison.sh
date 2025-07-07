#!/bin/bash

# Sequential AnyCam Batched Video Comparison Script
# Creates a continuous side-by-side video from all batched AnyCam outputs

set -e

# Default values
INPUT_VIDEO=""
BATCH_OUTPUT_DIR=""
OUTPUT_VIDEO="sequential_comparison.mp4"
COLORMAP="viridis"
LIST_FILES=false
BATCH_SIZE=5
BATCH_OVERLAP=1

show_help() {
    echo "Sequential AnyCam Batched Video Comparison Tool"
    echo "==============================================="
    echo ""
    echo "Creates a continuous video where:"
    echo "  • Left side: Original video frames"
    echo "  • Right side: Depth maps from all batches played sequentially"
    echo ""
    echo "Usage: $0 [OPTIONS] <input_video> <batch_output_dir>"
    echo ""
    echo "Arguments:"
    echo "  <input_video>        Original video file (.mp4, .avi, etc.)"
    echo "  <batch_output_dir>   Directory containing batch_XXX subdirectories"
    echo ""
    echo "Options:"
    echo "  -o, --output FILE     Output video filename (default: sequential_comparison.mp4)"
    echo "  --colormap NAME      Depth colormap: viridis, plasma, inferno, magma, jet"
    echo "  --list-files         List found batch files and exit"
    echo "  --batch-size N       Batch size used in processing (default: 5)"
    echo "  --batch-overlap N    Batch overlap used in processing (default: 1)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  $0 data/video.mp4 outputs/batches/"
    echo ""
    echo "  # Custom parameters matching your batch processing"
    echo "  $0 --batch-size 10 --batch-overlap 2 data/video.mp4 outputs/batches/"
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
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --batch-overlap)
            BATCH_OVERLAP="$2"
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

# Find all batch directories
echo "Scanning for batch directories..."
batch_dirs=($(find "$BATCH_OUTPUT_DIR" -type d -name "batch_*" | sort))

if [ ${#batch_dirs[@]} -eq 0 ]; then
    echo "Error: No batch directories found in $BATCH_OUTPUT_DIR"
    echo "Expected directories named batch_001, batch_002, etc."
    exit 1
fi

echo "Found ${#batch_dirs[@]} batch directories"

echo ""
echo "Sequential AnyCam Video Comparison"
echo "=================================="
echo "Input Video:     $INPUT_VIDEO"
echo "Batch Dir:       $BATCH_OUTPUT_DIR"
echo "Output Video:    $OUTPUT_VIDEO"
echo "Colormap:        $COLORMAP"
echo "Batch Size:      $BATCH_SIZE"
echo "Batch Overlap:   $BATCH_OVERLAP"
echo "Found Batches:   ${#batch_dirs[@]}"
echo ""

if [ "$LIST_FILES" = true ]; then
    echo "Listing batch contents..."
    for batch_dir in "${batch_dirs[@]}"; do
        echo ""
        echo "=== $(basename "$batch_dir") ==="
        ls -la "$batch_dir/"
    done
    exit 0
fi

# Create temporary Python script for video processing
temp_script=$(mktemp --suffix=.py)
cat > "$temp_script" << 'EOF'
import cv2
import numpy as np
import os
import sys
import argparse
import glob
import matplotlib.pyplot as plt

def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total_frames, width, height

def load_frame_at_index(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def load_depth_from_batch(batch_dir, frame_index):
    depth_file = os.path.join(batch_dir, "depths.npy")
    if os.path.exists(depth_file):
        try:
            depths = np.load(depth_file)
            if frame_index < depths.shape[0]:
                return depths[frame_index]
        except Exception as e:
            print(f"Error loading depth from {depth_file}: {e}")
    return None

def apply_colormap(depth, colormap):
    if depth.max() > depth.min():
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    else:
        depth_norm = np.zeros_like(depth)
    
    cmap = plt.get_cmap(colormap)
    colored = cmap(depth_norm)
    return (colored[:, :, :3] * 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help='Path to original video')
    parser.add_argument('batch_dir', help='Directory containing batch outputs')
    parser.add_argument('-o', '--output', required=True, help='Output video path')
    parser.add_argument('--colormap', default='viridis', help='Colormap for depth')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size')
    parser.add_argument('--batch-overlap', type=int, default=1, help='Batch overlap')
    
    args = parser.parse_args()
    
    # Get video properties
    fps, total_frames, width, height = get_video_properties(args.video_path)
    print(f"Video: {fps} fps, {total_frames} frames, {width}x{height}")
    
    # Find batch directories
    batch_dirs = sorted(glob.glob(os.path.join(args.batch_dir, "batch_*")))
    print(f"Found {len(batch_dirs)} batches")
    
    if not batch_dirs:
        print("No batch directories found!")
        return
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width * 2, height))
    
    step_size = args.batch_size - args.batch_overlap
    written_frames = 0
    
    # Process each batch sequentially
    for batch_idx, batch_dir in enumerate(batch_dirs):
        batch_name = os.path.basename(batch_dir)
        print(f"Processing {batch_name} ({batch_idx + 1}/{len(batch_dirs)})...")
        
        # Calculate the frame range for this batch in the original video
        batch_start_frame = batch_idx * step_size
        
        # Process each frame in this batch
        for frame_in_batch in range(args.batch_size):
            global_frame = batch_start_frame + frame_in_batch
            
            # Skip if we've gone beyond the video
            if global_frame >= total_frames:
                print(f"Reached end of video at frame {global_frame}")
                break
                
            # Load original frame
            orig_frame = load_frame_at_index(args.video_path, global_frame)
            if orig_frame is None:
                print(f"Could not load frame {global_frame}")
                continue
                
            # Load depth from current batch
            depth = load_depth_from_batch(batch_dir, frame_in_batch)
            
            if depth is not None:
                # Resize depth to match frame
                depth_resized = cv2.resize(depth, (width, height))
                depth_colored = apply_colormap(depth_resized, args.colormap)
                depth_colored_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
            else:
                # Create black depth image if no depth available
                depth_colored_bgr = np.zeros((height, width, 3), dtype=np.uint8)
                print(f"No depth data for batch {batch_name}, frame {frame_in_batch}")
            
            # Combine side by side: original on left, depth on right
            combined = np.hstack([orig_frame, depth_colored_bgr])
            out.write(combined)
            written_frames += 1
    
    out.release()
    print(f"Video saved to: {args.output}")
    print(f"Total frames written: {written_frames}")

if __name__ == "__main__":
    main()
EOF

echo "Creating sequential comparison video..."

# Run the Python script
python3 "$temp_script" "$INPUT_VIDEO" "$BATCH_OUTPUT_DIR" \
    -o "$OUTPUT_VIDEO" \
    --colormap "$COLORMAP" \
    --batch-size "$BATCH_SIZE" \
    --batch-overlap "$BATCH_OVERLAP"

# Clean up
rm -f "$temp_script"

if [ -f "$OUTPUT_VIDEO" ]; then
    echo ""
    echo "✅ Sequential comparison video created successfully!"
    echo "📹 Output: $OUTPUT_VIDEO"
    echo ""
    echo "The video shows:"
    echo "  • Left side: Original frames playing sequentially"
    echo "  • Right side: Depth maps from all batches playing sequentially"
    echo "  • Colormap: $COLORMAP"
    echo "  • Batches: ${#batch_dirs[@]} processed sequentially"
else
    echo "⚠️  Failed to create video"
    exit 1
fi

echo ""
echo "Processing complete!"
