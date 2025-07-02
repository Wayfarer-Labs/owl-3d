#!/bin/bash

# Alternative comparison video creator using ffmpeg directly
# Usage: ./create_comparison_ffmpeg.sh <input_video_or_frames> <anycam_output_dir> [output_video.mp4] [fps]

set -e

INPUT="$1"
OUTPUT_DIR="$2"
OUTPUT_VIDEO="${3:-comparison_video.mp4}"
FPS="${4:-30}"

if [[ -z "$INPUT" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 <input_video_or_frames> <anycam_output_dir> [output_video.mp4] [fps]"
    echo ""
    echo "Examples:"
    echo "  $0 input.mp4 /path/to/anycam/output"
    echo "  $0 /path/to/frames /path/to/anycam/output comparison.mp4 25"
    exit 1
fi

# Check if input exists
if [[ ! -e "$INPUT" ]]; then
    echo "Error: Input '$INPUT' does not exist"
    exit 1
fi

# Check if output directory exists
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Error: Output directory '$OUTPUT_DIR' does not exist"
    exit 1
fi

echo "Creating comparison video using ffmpeg..."
echo "Input: $INPUT"
echo "AnyCam output: $OUTPUT_DIR"
echo "Output video: $OUTPUT_VIDEO"
echo "FPS: $FPS"

# Create temporary directory for processing
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "Using temporary directory: $TEMP_DIR"

# Step 1: Extract frames from input if it's a video
FRAMES_DIR="$TEMP_DIR/frames"
mkdir -p "$FRAMES_DIR"

if [[ -f "$INPUT" ]]; then
    # Input is a video file
    echo "Extracting frames from video..."
    ffmpeg -i "$INPUT" -q:v 2 "$FRAMES_DIR/frame_%04d.png" -y
else
    # Input is a directory with frames
    echo "Using existing frames directory..."
    cp "$INPUT"/*.{png,jpg,jpeg} "$FRAMES_DIR/" 2>/dev/null || true
    
    # Rename files to have consistent naming
    counter=1
    for file in "$FRAMES_DIR"/*; do
        if [[ -f "$file" ]]; then
            ext="${file##*.}"
            mv "$file" "$FRAMES_DIR/frame_$(printf "%04d" $counter).$ext"
            counter=$((counter + 1))
        fi
    done
fi

# Step 2: Process depth maps
DEPTH_DIR="$TEMP_DIR/depth"
mkdir -p "$DEPTH_DIR"

echo "Processing depth maps..."

# Use our Python script to generate side-by-side frames
python3 - << EOF
import os
import cv2
import numpy as np
import glob
from pathlib import Path

frames_dir = "$FRAMES_DIR"
output_dir = "$OUTPUT_DIR"
depth_dir = "$DEPTH_DIR"
fps = $FPS

# Find frame files
frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")) + 
                    glob.glob(os.path.join(frames_dir, "*.jpg")) + 
                    glob.glob(os.path.join(frames_dir, "*.jpeg")))

print(f"Found {len(frame_files)} frame files")

# Find depth files
depth_files = []
depth_npy_file = os.path.join(output_dir, "depths.npy")

if os.path.exists(depth_npy_file):
    print("Loading depths from single .npy file")
    depths = np.load(depth_npy_file)
    if len(depths.shape) == 4 and depths.shape[1] == 1:
        depths = depths.squeeze(1)  # Remove singleton dimension
    print(f"Loaded depths shape: {depths.shape}")
else:
    # Look for individual depth files
    depth_pattern = os.path.join(output_dir, "*.npy")
    depth_files = sorted(glob.glob(depth_pattern))
    print(f"Found {len(depth_files)} individual depth files")

# Process frames
for i, frame_file in enumerate(frame_files):
    try:
        # Load original frame
        frame = cv2.imread(frame_file)
        if frame is None:
            continue
            
        height, width = frame.shape[:2]
        
        # Load corresponding depth
        if os.path.exists(depth_npy_file):
            if i < len(depths):
                depth = depths[i]
            else:
                continue
        else:
            if i < len(depth_files):
                depth = np.load(depth_files[i])
            else:
                continue
        
        # Normalize depth for visualization
        if depth.max() > depth.min():
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
        else:
            depth_norm = depth
        
        # Apply colormap
        depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        
        # Resize depth to match frame
        depth_resized = cv2.resize(depth_colored, (width, height))
        
        # Create side-by-side
        combined = np.hstack([frame, depth_resized])
        
        # Add labels
        cv2.putText(combined, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Original", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Depth", (width + 10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save combined frame
        output_file = os.path.join(depth_dir, f"combined_{i:04d}.png")
        cv2.imwrite(output_file, combined)
        
    except Exception as e:
        print(f"Error processing frame {i}: {e}")
        continue

print("Frame processing complete")
EOF

# Step 3: Create video using ffmpeg
echo "Creating video with ffmpeg..."
ffmpeg -r "$FPS" -i "$DEPTH_DIR/combined_%04d.png" \
       -c:v libx264 -pix_fmt yuv420p \
       -crf 23 -preset medium \
       "$OUTPUT_VIDEO" -y

if [[ -f "$OUTPUT_VIDEO" ]]; then
    echo "✅ Video created successfully: $OUTPUT_VIDEO"
    echo "File size: $(du -h "$OUTPUT_VIDEO" | cut -f1)"
else
    echo "❌ Failed to create video"
    exit 1
fi
