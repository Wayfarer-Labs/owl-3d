#!/bin/bash

# Test script for video comparison functionality
# This creates test data and verifies the video creation works

set -e

echo "🧪 Testing video comparison functionality..."

# Create test directory
TEST_DIR="test_video_comparison"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

echo "📁 Created test directory: $TEST_DIR"

# Create test input video
echo "🎬 Creating test input video..."
ffmpeg -f lavfi -i testsrc=duration=2:size=640x480:rate=10 \
       -c:v libx264 -pix_fmt yuv420p \
       "$TEST_DIR/test_input.mp4" -y -loglevel quiet

echo "✅ Created test input video"

# Create mock AnyCam output
echo "🎨 Creating mock AnyCam depth output..."
ANYCAM_OUTPUT="$TEST_DIR/anycam_output"
mkdir -p "$ANYCAM_OUTPUT"

# Create test depth data using Python
python3 - << EOF
import numpy as np
import os

output_dir = "$ANYCAM_OUTPUT"
num_frames = 20

# Create individual depth files
for i in range(num_frames):
    depth = np.random.rand(480, 640) * 10  # Random depth values
    np.save(os.path.join(output_dir, f"depth_{i:04d}.npy"), depth)

# Also create a single combined file
all_depths = np.random.rand(num_frames, 480, 640) * 10
np.save(os.path.join(output_dir, "depths.npy"), all_depths)

print(f"Created {num_frames} individual depth files and combined depths.npy")
EOF

echo "✅ Created mock depth data"

# Test 1: Test the video creation diagnostics
echo ""
echo "🔬 Test 1: Running video creation diagnostics..."
python3 test_video_creation.py

# Test 2: Test OpenCV-based comparison video creation
echo ""
echo "🎥 Test 2: Testing OpenCV-based comparison video creation..."
python3 create_comparison_video.py \
    "$TEST_DIR/test_input.mp4" \
    "$ANYCAM_OUTPUT" \
    -o "$TEST_DIR/comparison_opencv.mp4" \
    --fps 10

if [[ -f "$TEST_DIR/comparison_opencv.mp4" && -s "$TEST_DIR/comparison_opencv.mp4" ]]; then
    echo "✅ OpenCV comparison video created successfully"
    echo "   File size: $(du -h "$TEST_DIR/comparison_opencv.mp4" | cut -f1)"
else
    echo "⚠️  OpenCV comparison video failed or is empty"
fi

# Test 3: Test ffmpeg-based comparison video creation
echo ""
echo "🎞️  Test 3: Testing ffmpeg-based comparison video creation..."
./create_comparison_ffmpeg.sh \
    "$TEST_DIR/test_input.mp4" \
    "$ANYCAM_OUTPUT" \
    "$TEST_DIR/comparison_ffmpeg.mp4" \
    10

if [[ -f "$TEST_DIR/comparison_ffmpeg.mp4" && -s "$TEST_DIR/comparison_ffmpeg.mp4" ]]; then
    echo "✅ ffmpeg comparison video created successfully"
    echo "   File size: $(du -h "$TEST_DIR/comparison_ffmpeg.mp4" | cut -f1)"
else
    echo "❌ ffmpeg comparison video failed"
fi

# Test 4: Test with combined depths.npy file
echo ""
echo "📦 Test 4: Testing with combined depths.npy file..."
# Remove individual files to force using combined file
rm -f "$ANYCAM_OUTPUT"/depth_*.npy

python3 create_comparison_video.py \
    "$TEST_DIR/test_input.mp4" \
    "$ANYCAM_OUTPUT" \
    -o "$TEST_DIR/comparison_combined.mp4" \
    --fps 10

if [[ -f "$TEST_DIR/comparison_combined.mp4" && -s "$TEST_DIR/comparison_combined.mp4" ]]; then
    echo "✅ Combined depths.npy test successful"
    echo "   File size: $(du -h "$TEST_DIR/comparison_combined.mp4" | cut -f1)"
else
    echo "⚠️  Combined depths.npy test failed or is empty"
fi

# Summary
echo ""
echo "📋 Test Summary:"
echo "=================="

if [[ -f "$TEST_DIR/comparison_opencv.mp4" && -s "$TEST_DIR/comparison_opencv.mp4" ]]; then
    echo "✅ OpenCV method: Working"
else
    echo "❌ OpenCV method: Failed"
fi

if [[ -f "$TEST_DIR/comparison_ffmpeg.mp4" && -s "$TEST_DIR/comparison_ffmpeg.mp4" ]]; then
    echo "✅ ffmpeg method: Working"
else
    echo "❌ ffmpeg method: Failed"
fi

if [[ -f "$TEST_DIR/comparison_combined.mp4" && -s "$TEST_DIR/comparison_combined.mp4" ]]; then
    echo "✅ Combined depths: Working"
else
    echo "❌ Combined depths: Failed"
fi

echo ""
echo "🗂️  Test files created in: $TEST_DIR"
echo "You can inspect the generated videos to verify they look correct."

# Clean up option
echo ""
read -p "Delete test files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$TEST_DIR"
    echo "🗑️  Test files deleted"
else
    echo "📁 Test files kept in: $TEST_DIR"
fi
