#!/usr/bin/env python3
"""
video_batch_processor.py

Processes video files in batches through AnyCam Docker pipeline.

1. Takes an input video file and processes it in configurable batch sizes
2. For each batch:
   - Extracts RGB frames as tensor (batch_size, 3, H, W)
   - Processes through AnyCam within Docker to get depth/disparity
   - Returns output tensor (batch_size, 1, H, W)
3. Saves tensors for each batch

Usage:
    python video_batch_processor.py \
       --input_video PATH/TO/VIDEO.mp4 \
       --output_dir ./batch_outputs \
       --batch_size 97 \
       --container_name anycam:latest

The system works entirely within Docker and processes frames in batches
to avoid ffmpeg subsampling issues.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import json
import argparse
from typing import Tuple, List, Optional
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch import Tensor
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(
        description="Process video in batches through AnyCam Docker pipeline"
    )
    p.add_argument(
        "--input_video", required=True,
        help="Path to input video file"
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Output directory for batch tensors"
    )
    p.add_argument(
        "--batch_size", type=int, default=97,
        help="Number of frames per batch (default: 97)"
    )
    p.add_argument(
        "--container_name", default="anycam:latest",
        help="Docker container name (default: anycam:latest)"
    )
    p.add_argument(
        "--model_path", default="pretrained_models/anycam_seq8",
        help="Model path within container (default: pretrained_models/anycam_seq8)"
    )
    p.add_argument(
        "--ba_refinement", action="store_true",
        help="Enable bundle adjustment refinement"
    )
    p.add_argument(
        "--visualize", action="store_true",
        help="Enable visualization"
    )
    p.add_argument(
        "--overlap", type=int, default=0,
        help="Overlap between batches in frames (default: 0)"
    )
    return p.parse_args()


def get_video_info(video_path: str) -> Tuple[int, int, int, float]:
    """
    Get video information: frame count, width, height, fps
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    cap.release()
    return frame_count, width, height, fps


def extract_frames_batch(video_path: str, start_frame: int, num_frames: int) -> torch.Tensor:
    """
    Extract a batch of frames from video and return as tensor (n, 3, h, w)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could only read {i} frames of requested {num_frames}")
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and change from HWC to CHW
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)  # (3, H, W)
        frames.append(frame_tensor)
    
    cap.release()
    
    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted from video at start_frame={start_frame}")
    
    # Stack frames into batch tensor (n, 3, h, w)
    batch_tensor = torch.stack(frames, dim=0)
    return batch_tensor


def create_batch_video(frames_tensor: torch.Tensor, output_path: str, fps: float = 30.0):
    """
    Create a video file from frames tensor (n, 3, h, w) for AnyCam processing
    """
    n, c, h, w = frames_tensor.shape
    
    # Convert tensor to numpy and change from CHW to HWC
    frames_np = frames_tensor.permute(0, 2, 3, 1).numpy().astype(np.uint8)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for i in range(n):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()


def run_anycam_docker(
    input_video_path: str,
    output_dir: str,
    container_name: str,
    model_path: str,
    ba_refinement: bool = False,
    visualize: bool = False
) -> str:
    """
    Run AnyCam Docker processing on a video file and return output directory
    """
    # Create absolute paths
    input_video_abs = os.path.abspath(input_video_path)
    output_dir_abs = os.path.abspath(output_dir)
    
    # Ensure output directory exists
    os.makedirs(output_dir_abs, exist_ok=True)
    
    # Check if Docker image exists
    try:
        subprocess.run(
            ["docker", "image", "inspect", container_name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Docker image '{container_name}' not found. Please build it first.")
    
    # Check for GPU support
    gpu_flags = []
    try:
        subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        gpu_flags = ["--gpus", "all"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: nvidia-smi not found. Running without GPU support.")
    
    # Build AnyCam command
    container_input = f"/workspace/data/{os.path.basename(input_video_path)}"
    container_output = "/workspace/outputs"
    
    anycam_cmd = (
        f"cd /workspace/anycam && python anycam/scripts/anycam_demo.py "
        f"++input_path={container_input} "
        f"++model_path={model_path} "
        f"++output_path={container_output} "
        f"++visualize={str(visualize).lower()} "
        f"++ba_refinement={str(ba_refinement).lower()}"
    )
    
    # Run Docker container
    docker_cmd = [
        "docker", "run", "--rm",
        "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "-v", f"{os.path.dirname(input_video_abs)}:/workspace/data",
        "-v", f"{output_dir_abs}:/workspace/outputs",
        "-p", "9090:9090"
    ] + gpu_flags + [
        container_name,
        "bash", "-c", anycam_cmd
    ]
    
    print(f"Running Docker command: {' '.join(docker_cmd)}")
    
    try:
        result = subprocess.run(docker_cmd, check=True, capture_output=True, text=True)
        print("AnyCam processing completed successfully")
        return output_dir_abs
    except subprocess.CalledProcessError as e:
        print(f"Docker command failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise RuntimeError(f"AnyCam Docker processing failed: {e}")


def extract_depth_from_anycam_output(output_dir: str, num_frames: int) -> torch.Tensor:
    """
    Extract depth/disparity information from AnyCam output and return as tensor (n, 1, h, w)
    """
    # Look for depth or disparity outputs in the AnyCam output directory
    depth_files = []
    
    # Common patterns for depth outputs from AnyCam
    patterns = [
        "depth_*.png", "depth_*.jpg", "depth_*.npy",
        "disparity_*.png", "disparity_*.jpg", "disparity_*.npy",
        "*_depth.png", "*_depth.jpg", "*_depth.npy",
        "*_disparity.png", "*_disparity.jpg", "*_disparity.npy"
    ]
    
    import glob
    
    for pattern in patterns:
        files = glob.glob(os.path.join(output_dir, "**", pattern), recursive=True)
        depth_files.extend(files)
    
    if not depth_files:
        # If no specific depth files found, look for any output images
        image_patterns = ["*.png", "*.jpg", "*.jpeg"]
        for pattern in image_patterns:
            files = glob.glob(os.path.join(output_dir, "**", pattern), recursive=True)
            depth_files.extend(files)
    
    if not depth_files:
        raise RuntimeError(f"No depth/disparity outputs found in {output_dir}")
    
    # Sort files to ensure consistent ordering
    depth_files = sorted(depth_files)[:num_frames]  # Take only the number we need
    
    print(f"Found {len(depth_files)} depth/disparity files")
    
    depth_tensors = []
    for depth_file in depth_files:
        if depth_file.endswith('.npy'):
            # Load numpy array
            depth_data = np.load(depth_file)
        else:
            # Load image file
            depth_img = Image.open(depth_file)
            if depth_img.mode != 'L':
                depth_img = depth_img.convert('L')  # Convert to grayscale
            depth_data = np.array(depth_img)
        
        # Convert to tensor and ensure it's (1, H, W)
        if len(depth_data.shape) == 2:
            depth_tensor = torch.from_numpy(depth_data).unsqueeze(0)  # Add channel dimension
        elif len(depth_data.shape) == 3 and depth_data.shape[2] == 1:
            depth_tensor = torch.from_numpy(depth_data).permute(2, 0, 1)
        else:
            # Take first channel if multi-channel
            depth_tensor = torch.from_numpy(depth_data[:, :, 0]).unsqueeze(0)
        
        depth_tensors.append(depth_tensor)
    
    # Stack into batch tensor (n, 1, h, w)
    batch_depth_tensor = torch.stack(depth_tensors, dim=0)
    return batch_depth_tensor


def process_video_in_batches(
    input_video: str,
    output_dir: str,
    batch_size: int,
    container_name: str,
    model_path: str,
    ba_refinement: bool = False,
    visualize: bool = False,
    overlap: int = 0
) -> List[str]:
    """
    Process entire video in batches and save tensor outputs
    """
    # Get video information
    total_frames, width, height, fps = get_video_info(input_video)
    print(f"Video info: {total_frames} frames, {width}x{height}, {fps} fps")
    
    # Calculate batch parameters
    step_size = batch_size - overlap
    if step_size <= 0:
        raise ValueError("Batch size must be greater than overlap")
    
    num_batches = (total_frames + step_size - 1) // step_size
    print(f"Processing {num_batches} batches with step size {step_size}")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    batch_outputs = []
    
    current_frame = 0
    batch_id = 1
    
    with tempfile.TemporaryDirectory() as temp_dir:
        while current_frame < total_frames:
            # Calculate frames for this batch
            remaining_frames = total_frames - current_frame
            frames_to_process = min(batch_size, remaining_frames)
            
            print(f"\n=== Processing Batch {batch_id}/{num_batches} ===")
            print(f"Frames {current_frame} to {current_frame + frames_to_process - 1}")
            
            # Extract RGB frames as tensor
            rgb_tensor = extract_frames_batch(input_video, current_frame, frames_to_process)
            print(f"Extracted RGB tensor shape: {rgb_tensor.shape}")
            
            # Save RGB tensor
            rgb_output_path = os.path.join(output_dir, f"batch_{batch_id:03d}_rgb.pt")
            torch.save(rgb_tensor, rgb_output_path)
            
            # Create temporary video for AnyCam processing
            temp_video_path = os.path.join(temp_dir, f"batch_{batch_id}.mp4")
            create_batch_video(rgb_tensor, temp_video_path, fps)
            
            # Create temporary output directory for this batch
            temp_output_dir = os.path.join(temp_dir, f"output_{batch_id}")
            os.makedirs(temp_output_dir, exist_ok=True)
            
            # Run AnyCam processing
            try:
                run_anycam_docker(
                    temp_video_path,
                    temp_output_dir,
                    container_name,
                    model_path,
                    ba_refinement,
                    visualize
                )
                
                # Extract depth tensor from AnyCam output
                depth_tensor = extract_depth_from_anycam_output(temp_output_dir, frames_to_process)
                print(f"Extracted depth tensor shape: {depth_tensor.shape}")
                
                # Save depth tensor
                depth_output_path = os.path.join(output_dir, f"batch_{batch_id:03d}_depth.pt")
                torch.save(depth_tensor, depth_output_path)
                
                batch_info = {
                    'batch_id': batch_id,
                    'start_frame': current_frame,
                    'num_frames': frames_to_process,
                    'rgb_tensor_path': rgb_output_path,
                    'depth_tensor_path': depth_output_path,
                    'rgb_shape': rgb_tensor.shape,
                    'depth_shape': depth_tensor.shape
                }
                
                batch_outputs.append(batch_info)
                
                print(f"Batch {batch_id} completed successfully")
                
            except Exception as e:
                print(f"Error processing batch {batch_id}: {e}")
                print("Continuing to next batch...")
            
            # Move to next batch
            current_frame += step_size
            batch_id += 1
    
    # Save batch information
    batch_info_path = os.path.join(output_dir, "batch_info.json")
    with open(batch_info_path, 'w') as f:
        json.dump(batch_outputs, f, indent=2)
    
    print(f"\nProcessing complete! {len(batch_outputs)} batches processed successfully")
    print(f"Results saved in: {output_dir}")
    
    return [info['depth_tensor_path'] for info in batch_outputs]


def main():
    args = parse_args()
    
    # Validate inputs
    if not os.path.isfile(args.input_video):
        print(f"Error: Input video file does not exist: {args.input_video}")
        sys.exit(1)
    
    if args.batch_size <= 0:
        print("Error: Batch size must be positive")
        sys.exit(1)
    
    if args.overlap < 0 or args.overlap >= args.batch_size:
        print("Error: Overlap must be between 0 and batch_size-1")
        sys.exit(1)
    
    # Check if Docker is available
    try:
        subprocess.run(["docker", "--version"], check=True, stdout=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Docker is not installed or not in PATH")
        sys.exit(1)
    
    print("Video Batch Processor for AnyCam")
    print("================================")
    print(f"Input video: {args.input_video}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Overlap: {args.overlap}")
    print(f"Container: {args.container_name}")
    print(f"Model: {args.model_path}")
    print("")
    
    try:
        depth_tensor_paths = process_video_in_batches(
            args.input_video,
            args.output_dir,
            args.batch_size,
            args.container_name,
            args.model_path,
            args.ba_refinement,
            args.visualize,
            args.overlap
        )
        
        print(f"\nSuccess! Generated {len(depth_tensor_paths)} depth tensor files:")
        for path in depth_tensor_paths:
            print(f"  {path}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
