#!/usr/bin/env python3
"""
AnyCam Side-by-Side Video Generator
Creates a comparison video showing original frames and depth maps side by side.
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def load_depth_image(depth_path, colormap='viridis'):
    """Load depth image and apply colormap."""
    
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
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    
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
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
    
    return depth_colored

def find_frame_files(input_video_path):
    """Extract frames from input video or find existing frames."""
    frames = []
    
    if os.path.isfile(input_video_path):
        # Extract frames from video
        print(f"Extracting frames from video: {input_video_path}")
        cap = cv2.VideoCapture(input_video_path)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append((frame_idx, frame))
            frame_idx += 1
            
        cap.release()
        
    elif os.path.isdir(input_video_path):
        # Load frames from directory
        print(f"Loading frames from directory: {input_video_path}")
        frame_files = sorted(glob.glob(os.path.join(input_video_path, "*.png")) + 
                           glob.glob(os.path.join(input_video_path, "*.jpg")) +
                           glob.glob(os.path.join(input_video_path, "*.jpeg")))
        
        for i, frame_file in enumerate(frame_files):
            frame = cv2.imread(frame_file)
            if frame is not None:
                frames.append((i, frame))
    
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

def create_side_by_side_video(input_path, output_dir, output_video, fps=30, colormap='viridis'):
    """Create side-by-side comparison video."""
    
    # Validate and fix output video path
    if not output_video or output_video.strip() == "":
        output_video = "comparison_video.mp4"
        print(f"Warning: Empty output path, using default: {output_video}")
    
    # Ensure output video has proper extension
    if not output_video.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        output_video = output_video + ".mp4"
        print(f"Warning: Added .mp4 extension: {output_video}")
    
    # Make output path absolute if it's not
    if not os.path.isabs(output_video):
        output_video = os.path.abspath(output_video)
    
    print(f"Output video path: '{output_video}'")
    
    # Find input frames
    frames = find_frame_files(input_path)
    if not frames:
        print(f"No frames found in {input_path}")
        return False
    
    # Find depth files
    depth_files = find_depth_files(output_dir)
    if not depth_files:
        print(f"No depth files found in {output_dir}")
        print("Looking for files matching patterns:")
        print("  - depth_*.png/jpg/npy")
        print("  - depths/depth_*.png/jpg/npy")
        print("  - *_depth.png/jpg/npy")
        return False
    
    print(f"Found {len(frames)} frames and {len(depth_files)} depth maps")
    
    # Get common frame indices
    frame_indices = {idx for idx, _ in frames}
    depth_indices = set(depth_files.keys())
    common_indices = sorted(frame_indices.intersection(depth_indices))
    
    if not common_indices:
        print("No matching frame and depth indices found!")
        print(f"Frame indices: {sorted(list(frame_indices))[:10]}...")
        print(f"Depth indices: {sorted(list(depth_indices))[:10]}...")
        return False
    
    print(f"Creating video with {len(common_indices)} matched frames")
    
    # Create frame lookup
    frame_dict = {idx: frame for idx, frame in frames}
    
    # Get video properties from first frame
    first_frame = frame_dict[common_indices[0]]
    height, width = first_frame.shape[:2]
    
    # Get depth properties from first depth frame to check size mismatch
    first_depth_path = depth_files[common_indices[0]]
    try:
        first_depth = load_depth_image(first_depth_path, colormap)
        depth_height, depth_width = first_depth.shape[:2]
        print(f"Frame size: {width}x{height}, Depth size: {depth_width}x{depth_height}")
    except Exception as e:
        print(f"Warning: Could not load first depth frame: {e}")
    
    # Ensure output directory exists
    output_dir_path = os.path.dirname(output_video)
    if output_dir_path and not os.path.exists(output_dir_path):
        print(f"Creating output directory: {output_dir_path}")
        os.makedirs(output_dir_path, exist_ok=True)
    
    # Try different codecs in order of preference
    codecs_to_try = [
        ('MP4V', cv2.VideoWriter_fourcc(*'MP4V'), '.mp4'),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG'), '.avi'),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID'), '.avi'),
    ]
    
    out = None
    successful_codec = None
    final_output_path = None
    
    for codec_name, fourcc, preferred_ext in codecs_to_try:
        # Adjust output path for codec if needed
        if codec_name in ['MJPG', 'XVID'] and output_video.endswith('.mp4'):
            test_output = output_video.replace('.mp4', '.avi')
        else:
            test_output = output_video
            
        print(f"Trying codec: {codec_name} with output: '{test_output}'")
        try:
            out = cv2.VideoWriter(test_output, fourcc, fps, (width * 2, height))
            
            if out and out.isOpened():
                successful_codec = codec_name
                final_output_path = test_output
                print(f"Successfully opened VideoWriter with codec: {codec_name}")
                break
            else:
                print(f"Failed to open VideoWriter with codec: {codec_name}")
                if out:
                    out.release()
                out = None
        except Exception as e:
            print(f"Exception with codec {codec_name}: {e}")
            if out:
                out.release()
            out = None
    
    if out is None or not out.isOpened():
        print("ERROR: Could not initialize VideoWriter with any codec")
        print("Trying ffmpeg fallback...")
        
        # Try ffmpeg fallback
        success = create_video_with_ffmpeg(
            common_indices, frame_dict, depth_files, 
            output_video, fps, colormap, width, height
        )
        
        if success:
            return True
        
        print("ffmpeg fallback also failed. Saving individual frames...")
        
        # Final fallback: save individual frames
        frames_dir = output_video.replace('.mp4', '_frames').replace('.avi', '_frames').replace('.mov', '_frames')
        print(f"Creating frames directory: {frames_dir}")
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, idx in enumerate(tqdm(common_indices, desc="Saving frames")):
            try:
                # Get original frame
                original_frame = frame_dict[idx]
                
                # Load and process depth
                depth_path = depth_files[idx]
                depth_colored = load_depth_image(depth_path, colormap)
                
                # Resize depth to match original frame
                depth_resized = cv2.resize(depth_colored, (width, height))
                
                # Create side-by-side frame
                combined_frame = np.hstack([original_frame, depth_resized])
                
                # Add frame number text
                cv2.putText(combined_frame, f"Frame {idx}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined_frame, "Original", (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(combined_frame, "Depth", (width + 10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save frame
                frame_filename = os.path.join(frames_dir, f"frame_{i:04d}.png")
                success = cv2.imwrite(frame_filename, combined_frame)
                if not success:
                    print(f"Failed to save frame {frame_filename}")
                    
            except Exception as e:
                print(f"Error processing frame {idx}: {e}")
                continue
        
        print(f"Frames saved to directory: {frames_dir}")
        print(f"You can create a video from these frames using:")
        print(f"  ffmpeg -r {fps} -i {frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {final_output_path or output_video}")
        return True
    
    print(f"Creating video: {final_output_path}")
    print(f"Resolution: {width * 2}x{height} at {fps} fps using {successful_codec}")
    
    frames_written = 0
    frames_failed = 0
    
    try:
        for idx in tqdm(common_indices, desc="Processing frames"):
            try:
                # Get original frame
                original_frame = frame_dict[idx]
                
                # Load and process depth
                depth_path = depth_files[idx]
                depth_colored = load_depth_image(depth_path, colormap)
                
                # Resize depth to match original frame
                depth_resized = cv2.resize(depth_colored, (width, height))
                
                # Create side-by-side frame
                combined_frame = np.hstack([original_frame, depth_resized])
                
                # Add frame number text
                cv2.putText(combined_frame, f"Frame {idx}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined_frame, "Original", (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(combined_frame, "Depth", (width + 10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Ensure frame is the right type and shape
                if combined_frame.dtype != np.uint8:
                    combined_frame = combined_frame.astype(np.uint8)
                
                if len(combined_frame.shape) != 3 or combined_frame.shape[2] != 3:
                    print(f"Warning: Frame {idx} has wrong shape: {combined_frame.shape}")
                    continue
                
                # Write frame
                success = out.write(combined_frame)
                if success:
                    frames_written += 1
                else:
                    frames_failed += 1
                    print(f"Warning: Failed to write frame {idx}")
                    
            except Exception as e:
                frames_failed += 1
                print(f"Error processing frame {idx}: {e}")
                continue
        
        print(f"Video writing complete. Frames written: {frames_written}, Failed: {frames_failed}")
        out.release()
        
        # Verify the file was created
        if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
            print(f"Video saved to: {final_output_path}")
            print(f"File size: {os.path.getsize(final_output_path)} bytes")
            
            # If we changed the extension, copy to original requested path
            if final_output_path != output_video:
                try:
                    import shutil
                    shutil.copy2(final_output_path, output_video)
                    print(f"Also copied to requested path: {output_video}")
                except Exception as e:
                    print(f"Could not copy to requested path: {e}")
                    
            return True
        else:
            print(f"ERROR: Video file was not created or is empty: {final_output_path}")
            print("Falling back to saving individual frames...")
            
            # Fallback: save individual frames
            frames_dir = output_video.replace('.mp4', '_frames').replace('.avi', '_frames').replace('.mov', '_frames')
            print(f"Creating frames directory: {frames_dir}")
            os.makedirs(frames_dir, exist_ok=True)
            
            for i, idx in enumerate(tqdm(common_indices, desc="Saving frames")):
                try:
                    # Get original frame
                    original_frame = frame_dict[idx]
                    
                    # Load and process depth
                    depth_path = depth_files[idx]
                    depth_colored = load_depth_image(depth_path, colormap)
                    
                    # Resize depth to match original frame
                    depth_resized = cv2.resize(depth_colored, (width, height))
                    
                    # Create side-by-side frame
                    combined_frame = np.hstack([original_frame, depth_resized])
                    
                    # Add frame number text
                    cv2.putText(combined_frame, f"Frame {idx}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(combined_frame, "Original", (10, height - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(combined_frame, "Depth", (width + 10, height - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save frame
                    frame_filename = os.path.join(frames_dir, f"frame_{i:04d}.png")
                    success = cv2.imwrite(frame_filename, combined_frame)
                    if not success:
                        print(f"Failed to save frame {frame_filename}")
                        
                except Exception as e:
                    print(f"Error processing frame {idx}: {e}")
                    continue
            
            print(f"Frames saved to directory: {frames_dir}")
            print(f"You can create a video from these frames using:")
            print(f"  ffmpeg -r {fps} -i {frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {final_output_path or output_video}")
            return True
            
    except Exception as e:
        print(f"ERROR during video creation: {e}")
        if out:
            out.release()
        return False

def create_video_with_ffmpeg(common_indices, frame_dict, depth_files, output_video, fps, colormap, width, height):
    """Create video using ffmpeg as fallback when OpenCV VideoWriter fails"""
    import tempfile
    import subprocess
    
    try:
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir}")
            
            # Save all combined frames as images
            for i, idx in enumerate(tqdm(common_indices, desc="Preparing frames for ffmpeg")):
                try:
                    # Get original frame
                    original_frame = frame_dict[idx]
                    
                    # Load and process depth
                    depth_path = depth_files[idx]
                    depth_colored = load_depth_image(depth_path, colormap)
                    
                    # Resize depth to match original frame
                    depth_resized = cv2.resize(depth_colored, (width, height))
                    
                    # Create side-by-side frame
                    combined_frame = np.hstack([original_frame, depth_resized])
                    
                    # Add frame number text
                    cv2.putText(combined_frame, f"Frame {idx}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(combined_frame, "Original", (10, height - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(combined_frame, "Depth", (width + 10, height - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save frame
                    frame_filename = os.path.join(temp_dir, f"frame_{i:04d}.png")
                    success = cv2.imwrite(frame_filename, combined_frame)
                    if not success:
                        print(f"Failed to save frame {frame_filename}")
                        return False
                        
                except Exception as e:
                    print(f"Error processing frame {idx}: {e}")
                    return False
            
            # Use ffmpeg to create video
            print("Creating video with ffmpeg...")
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite output file
                '-r', str(fps),  # framerate
                '-i', os.path.join(temp_dir, 'frame_%04d.png'),  # input pattern
                '-c:v', 'libx264',  # video codec
                '-pix_fmt', 'yuv420p',  # pixel format for compatibility
                '-crf', '23',  # quality (lower = better quality)
                '-preset', 'medium',  # encoding speed/quality tradeoff
                output_video
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                    print(f"✅ ffmpeg video created successfully: {output_video}")
                    print(f"File size: {os.path.getsize(output_video)} bytes")
                    return True
                else:
                    print(f"❌ ffmpeg succeeded but no output file created")
                    return False
            else:
                print(f"❌ ffmpeg failed with return code {result.returncode}")
                print(f"stderr: {result.stderr}")
                return False
                
    except Exception as e:
        print(f"❌ Exception in ffmpeg fallback: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Create side-by-side comparison video of frames and depth maps")
    parser.add_argument("input", help="Input video file or directory containing frames")
    parser.add_argument("output_dir", help="AnyCam output directory containing depth maps")
    parser.add_argument("-o", "--output", default="comparison_video.mp4", help="Output video filename")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS (default: 30)")
    parser.add_argument("--colormap", default="viridis", help="Colormap for depth visualization (default: viridis)")
    parser.add_argument("--list-files", action="store_true", help="List found files and exit")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}")
        return 1
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory does not exist: {args.output_dir}")
        return 1
    
    if args.list_files:
        print("=== INPUT FRAMES ===")
        frames = find_frame_files(args.input)
        print(f"Found {len(frames)} frames")
        for i, (idx, _) in enumerate(frames[:10]):
            print(f"  Frame {idx}")
        if len(frames) > 10:
            print(f"  ... and {len(frames) - 10} more")
        
        print("\n=== DEPTH FILES ===")
        depth_files = find_depth_files(args.output_dir)
        print(f"Found {len(depth_files)} depth files")
        for i, (idx, path) in enumerate(sorted(depth_files.items())[:10]):
            if isinstance(path, tuple):
                file_path, frame_idx = path
                print(f"  Frame {idx}: {os.path.basename(file_path)} (frame {frame_idx})")
            else:
                print(f"  Frame {idx}: {os.path.basename(path)}")
        if len(depth_files) > 10:
            print(f"  ... and {len(depth_files) - 10} more")
        
        return 0
    
    success = create_side_by_side_video(
        args.input, 
        args.output_dir, 
        args.output, 
        args.fps, 
        args.colormap
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
