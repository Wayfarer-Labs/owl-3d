#!/usr/bin/env python3
"""
Simple Video Creator - No OpenCV dependency
Creates MP4 videos using FFmpeg directly
"""

import os
import numpy as np
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

def load_depth_image(depth_path, colormap='viridis'):
    """Load depth image and apply colormap using matplotlib."""
    
    # Check if depth_path is a tuple (file_path, frame_index) for single numpy file
    if isinstance(depth_path, tuple):
        file_path, frame_index = depth_path
        depths_array = np.load(file_path)
        
        # Extract the specific frame
        if len(depths_array.shape) == 4:  # (num_frames, height, width, channels) or (num_frames, channels, height, width)
            if depths_array.shape[1] == 1:  # (num_frames, 1, height, width)
                depth = depths_array[frame_index, 0, :, :]  # Remove channel dimension
            else:  # (num_frames, height, width, channels)
                depth = depths_array[frame_index, :, :, 0]  # Take first channel
        elif len(depths_array.shape) == 3:  # (num_frames, height, width)
            depth = depths_array[frame_index]
        else:
            raise ValueError(f"Unexpected depth array shape: {depths_array.shape}")
    else:
        # Handle individual depth files
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        else:
            # Use PIL instead of OpenCV
            img = Image.open(depth_path)
            depth = np.array(img)
    
    # Handle case where depth might be None or empty
    if depth is None:
        raise ValueError(f"Could not load depth data from {depth_path}")
    
    # Ensure depth is 2D
    if len(depth.shape) > 2:
        # If it's 3D, try to squeeze out dimensions of size 1
        if depth.shape[0] == 1:
            depth = depth[0]  # Remove first dimension if it's 1
        elif depth.shape[2] == 1:
            depth = depth[:, :, 0]  # Remove last dimension if it's 1
        elif depth.shape[2] > 1:
            depth = depth[:, :, 0]  # Take first channel if multiple channels
        else:
            depth = depth.squeeze()  # Remove all dimensions of size 1
    
    # Final check - ensure it's 2D
    if len(depth.shape) != 2:
        raise ValueError(f"Could not convert depth to 2D array. Final shape: {depth.shape}")
    
    # Normalize depth for visualization
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max == depth_min:
        # Handle case where all depth values are the same
        depth_norm = np.zeros_like(depth)
    else:
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    depth_colored = cmap(depth_norm)
    depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)
    
    return depth_colored

def extract_frames_from_video(video_path, temp_dir):
    """Extract frames from video using FFmpeg."""
    print(f"Extracting frames from video: {video_path}")
    
    # Create frames subdirectory
    frames_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Use FFmpeg to extract frames
    ffmpeg_cmd = [
        'ffmpeg', '-i', video_path,
        '-q:v', '2',  # High quality
        os.path.join(frames_dir, 'frame_%04d.png')
    ]
    
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        print("✅ Frames extracted successfully!")
        
        # Find extracted frames
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        frames = []
        for i, frame_file in enumerate(frame_files):
            frames.append((i, frame_file))
        
        return frames
        
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error extracting frames: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        return []
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg.")
        return []

def find_frame_files(input_video_path, temp_dir=None):
    """Find frame files from video or directory."""
    frames = []
    
    if os.path.isfile(input_video_path):
        # Check if it's a video file
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
        if any(input_video_path.lower().endswith(ext) for ext in video_extensions):
            if temp_dir is None:
                print("Error: temp_dir required for video extraction")
                return frames
            return extract_frames_from_video(input_video_path, temp_dir)
        else:
            print(f"Unknown file type: {input_video_path}")
            return frames
        
    elif os.path.isdir(input_video_path):
        # Load frames from directory
        print(f"Loading frames from directory: {input_video_path}")
        frame_files = sorted(glob.glob(os.path.join(input_video_path, "*.png")) + 
                           glob.glob(os.path.join(input_video_path, "*.jpg")) +
                           glob.glob(os.path.join(input_video_path, "*.jpeg")))
        
        for i, frame_file in enumerate(frame_files):
            frames.append((i, frame_file))
    
    return frames

def find_depth_files(depth_dir):
    """Find depth files in the output directory."""
    depth_files = {}
    
    # First, check for single numpy file containing all depths
    single_depth_patterns = [
        "depths.npy",
        "depth.npy", 
        "all_depths.npy",
        "depths/depths.npy",
        "depths/depth.npy"
    ]
    
    print("Searching for single depth file...")
    for pattern in single_depth_patterns:
        files = glob.glob(os.path.join(depth_dir, pattern))
        print(f"  Pattern {pattern}: {files}")
        
        if files:
            depth_file = files[0]
            print(f"Found single depth file: {depth_file}")
            try:
                # Load the numpy array
                depths_array = np.load(depth_file)
                print(f"Loaded depths array with shape: {depths_array.shape}")
                
                # Handle different possible shapes
                if len(depths_array.shape) == 4:  # (num_frames, height, width, channels)
                    num_frames = depths_array.shape[0]
                    for i in range(num_frames):
                        depth_files[i] = (depth_file, i)  # Store file path and frame index
                elif len(depths_array.shape) == 3:  # (num_frames, height, width)
                    num_frames = depths_array.shape[0]
                    for i in range(num_frames):
                        depth_files[i] = (depth_file, i)  # Store file path and frame index
                else:
                    print(f"Unexpected depth array shape: {depths_array.shape}")
                    continue
                    
                print(f"Extracted {num_frames} depth frames from single file")
                return depth_files
                
            except Exception as e:
                print(f"Error loading {depth_file}: {e}")
                continue
    
    print("No single depth file found, searching for individual depth files...")
    
    # Fall back to individual depth file patterns
    individual_patterns = [
        "depth_*.png", "depth_*.jpg", "depth_*.npy",
        "depths/depth_*.png", "depths/depth_*.jpg", "depths/depth_*.npy",
        "depths/*_depth.png", "depths/*_depth.jpg", "depths/*_depth.npy",
        "*_depth.png", "*_depth.jpg", "*_depth.npy"
    ]
    
    for pattern in individual_patterns:
        files = glob.glob(os.path.join(depth_dir, pattern))
        print(f"  Pattern {pattern}: found {len(files)} files")
        
        for file in files:
            # Extract frame number from filename
            basename = os.path.basename(file)
            # Try different naming conventions
            try:
                if 'depth_' in basename:
                    frame_num = int(basename.split('depth_')[1].split('.')[0])
                elif '_depth' in basename:
                    frame_num = int(basename.split('_depth')[0])
                else:
                    # Try to extract any number from filename
                    import re
                    numbers = re.findall(r'\d+', basename)
                    if numbers:
                        frame_num = int(numbers[-1])  # Take last number
                    else:
                        continue
                        
                depth_files[frame_num] = file  # Store just the file path
            except ValueError:
                continue
    
    print(f"Found {len(depth_files)} individual depth files")
    return depth_files

def create_simple_video(input_path, output_dir, output_video, fps=30, colormap='viridis'):
    """Create video using FFmpeg directly."""
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Find input frames (extract from video if needed)
        frames = find_frame_files(input_path, temp_dir)
        if not frames:
            print(f"No frames found in {input_path}")
            return False
        
        # Find depth files
        depth_files = find_depth_files(output_dir)
        if not depth_files:
            print(f"No depth files found in {output_dir}")
            return False
        
        print(f"Found {len(frames)} frames and {len(depth_files)} depth maps")
        
        # Get common frame indices
        frame_indices = {idx for idx, _ in frames}
        depth_indices = set(depth_files.keys())
        common_indices = sorted(frame_indices.intersection(depth_indices))
        
        if not common_indices:
            print("No matching frame and depth indices found!")
            return False
        
        print(f"Creating video with {len(common_indices)} matched frames")
        
        # Create frame lookup
        frame_dict = {idx: frame_path for idx, frame_path in frames}
        
        # Create subdirectory for side-by-side frames
        output_frames_dir = os.path.join(temp_dir, "output_frames")
        os.makedirs(output_frames_dir, exist_ok=True)
        
        print(f"Creating side-by-side frames in {output_frames_dir}")
        
        # Process each frame
        for i, idx in enumerate(tqdm(common_indices, desc="Processing frames")):
            try:
                # Load original frame
                original_frame_path = frame_dict[idx]
                original_img = Image.open(original_frame_path)
                original_array = np.array(original_img)
                
                # Load and process depth
                depth_path = depth_files[idx]
                depth_colored = load_depth_image(depth_path, colormap)
                
                # Convert depth to PIL Image and resize to match original
                depth_img = Image.fromarray(depth_colored)
                depth_img = depth_img.resize(original_img.size)
                
                # Create side-by-side image
                combined_width = original_img.width * 2
                combined_height = original_img.height
                combined_img = Image.new('RGB', (combined_width, combined_height))
                
                # Paste images side by side
                combined_img.paste(original_img, (0, 0))
                combined_img.paste(depth_img, (original_img.width, 0))
                
                # Save combined frame
                frame_filename = os.path.join(output_frames_dir, f"frame_{i:04d}.png")
                combined_img.save(frame_filename)
                
            except Exception as e:
                print(f"Error processing frame {idx}: {e}")
                continue
        
        # Use FFmpeg to create video
        print(f"Creating video with FFmpeg: {output_video}")
        ffmpeg_cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-framerate', str(fps),
            '-i', os.path.join(output_frames_dir, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',  # Good quality
            output_video
        ]
        
        try:
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            print("✅ Video created successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e}")
            print(f"FFmpeg stderr: {e.stderr}")
            return False
        except FileNotFoundError:
            print("Error: FFmpeg not found. Please install FFmpeg.")
            return False

def main():
    parser = argparse.ArgumentParser(description='Create simple comparison video without OpenCV')
    parser.add_argument('input_path', help='Input video file (.mp4, .avi, etc.) or frames directory')
    parser.add_argument('output_dir', help='Directory containing depth maps')
    parser.add_argument('-o', '--output', default='comparison_video.mp4', help='Output video path')
    parser.add_argument('--fps', type=int, default=30, help='Output FPS')
    parser.add_argument('--colormap', default='viridis', help='Depth colormap')
    parser.add_argument('--list-files', action='store_true', help='List found files and exit')
    
    args = parser.parse_args()
    
    if args.list_files:
        print("=== Input Frames ===")
        # For listing, we don't need to extract frames from video
        if os.path.isfile(args.input_path):
            print(f"  Input is video file: {args.input_path}")
            print("  (Frames will be extracted during processing)")
        else:
            frames = find_frame_files(args.input_path)
            for idx, path in frames[:10]:  # Show first 10
                print(f"  {idx}: {path}")
            if len(frames) > 10:
                print(f"  ... and {len(frames) - 10} more")
        
        print("\n=== Depth Files ===")
        depth_files = find_depth_files(args.output_dir)
        for idx in sorted(list(depth_files.keys())[:10]):  # Show first 10
            print(f"  {idx}: {depth_files[idx]}")
        if len(depth_files) > 10:
            print(f"  ... and {len(depth_files) - 10} more")
        return
    
    success = create_simple_video(
        args.input_path, 
        args.output_dir, 
        args.output, 
        args.fps, 
        args.colormap
    )
    
    if success:
        print(f"📹 Output: {args.output}")
    else:
        print("❌ Failed to create video")
        exit(1)

if __name__ == '__main__':
    main()
