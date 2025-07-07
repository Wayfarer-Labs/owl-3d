#!/usr/bin/env python3
"""
anycam_hub_processor.py

A simplified AnyCam processor using the torch.hub interface.
Processes video files by extracting frames with ffmpeg and running through AnyCam.

Usage:
    python anycam_hub_processor.py \
       --input_video PATH/TO/VIDEO.mp4 \
       --output_dir ./outputs \
       --ba_refinement

The script:
1. Uses ffmpeg to extract frames from video as (H,W,3) arrays in [0,1] range
2. Processes frames through AnyCam using torch.hub interface
3. Saves trajectory, depths, uncertainties, and projection matrix
"""

import os
import sys
import subprocess
import tempfile
import argparse
import json
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process video through AnyCam using torch.hub interface"
    )
    parser.add_argument(
        "--input_video", nargs='*',
        help="One or more input video file paths"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--ba_refinement", action="store_true",
        help="Enable bundle adjustment refinement"
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Maximum number of frames to process (default: all)"
    )
    parser.add_argument(
        "--resize_height", type=int, default=None,
        help="Resize frames to this height (maintains aspect ratio)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Number of frames to process per video"
    )
    return parser.parse_args()


def get_video_info(video_path: str) -> tuple:
    """Get video information using ffprobe."""
    if not os.path.isfile(video_path):
        raise RuntimeError(f"Video file does not exist: {video_path}")
    
    try:
        # Get frame count
        frame_count_cmd = [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames", "-of", "csv=p=0",
            video_path
        ]
        result = subprocess.run(frame_count_cmd, capture_output=True, text=True, check=True)
        frame_count = int(result.stdout.strip())
        
        if frame_count <= 0:
            # Fallback: count frames by decoding
            frame_count_cmd = [
                "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-count_frames", "-show_entries", "stream=nb_read_frames",
                "-of", "csv=p=0", video_path
            ]
            result = subprocess.run(frame_count_cmd, capture_output=True, text=True, check=True)
            frame_count = int(result.stdout.strip())
        
        # Get width, height, and fps
        info_cmd = [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "csv=p=0", video_path
        ]
        result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
        width, height, fps_str = result.stdout.strip().split(',')
        
        width = int(width)
        height = int(height)
        
        # Convert fps fraction to decimal
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        
        return frame_count, width, height, fps
        
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"Failed to get video info for {video_path}: {e}")


def extract_frames_with_ffmpeg(video_path: str, max_frames: int = None, resize_height: int = None) -> List[np.ndarray]:
    """
    Extract frames from video using ffmpeg and return as list of (H,W,3) arrays in [0,1] range.
    """
    if not os.path.isfile(video_path):
        raise RuntimeError(f"Video file does not exist: {video_path}")
    
    # Get video info
    total_frames, width, height, fps = get_video_info(video_path)
    print(f"Video info: {total_frames} frames, {width}x{height}, {fps:.2f} fps")
    
    # Determine number of frames to extract
    frames_to_extract = min(total_frames, max_frames) if max_frames else total_frames
    print(f"Extracting {frames_to_extract} frames")
    
    frames = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Build ffmpeg command
        output_pattern = os.path.join(temp_dir, "frame_%06d.png")
        
        ffmpeg_cmd = [
            "ffmpeg", "-v", "quiet", "-i", video_path,
            "-frames:v", str(frames_to_extract),
            "-f", "image2"
        ]
        
        # Add resize filter if specified
        if resize_height:
            aspect_ratio = width / height
            resize_width = int(resize_height * aspect_ratio)
            # Ensure dimensions are even (required for some encoders)
            resize_width = resize_width + (resize_width % 2)
            resize_height = resize_height + (resize_height % 2)
            ffmpeg_cmd.extend(["-vf", f"scale={resize_width}:{resize_height}"])
        
        ffmpeg_cmd.append(output_pattern)
        
        try:
            subprocess.run(ffmpeg_cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract frames from video: {e}")
        
        # Load extracted frames
        print("Loading frames...")
        for i in tqdm(range(1, frames_to_extract + 1)):
            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            if not os.path.exists(frame_path):
                if i == 1:
                    raise RuntimeError("No frames were extracted from video")
                print(f"Warning: Could only extract {i-1} frames of requested {frames_to_extract}")
                break
            
            # Load frame using PIL and convert to RGB
            with Image.open(frame_path) as img:
                img_rgb = img.convert('RGB')
                # Convert to numpy array (H, W, 3) and normalize to [0, 1]
                frame_array = np.array(img_rgb, dtype=np.float32) / 255.0
                frames.append(frame_array)
        
        if len(frames) == 0:
            raise RuntimeError("No frames were successfully loaded")
        
        print(f"Successfully loaded {len(frames)} frames")
        return frames


def install_missing_dependencies():
    """Install missing dependencies for AnyCam."""
    import subprocess
    import sys
    
    print("Installing missing dependencies...")
    
    # List of packages that might be needed
    packages = [
        'opencv-python',
        'scipy',
        'scikit-image',
        'matplotlib',
        'easydict',
        'tensorboard',
        'timm',
        'kornia'
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install {package}: {e}")
    
    # Try to install unimatch specifically
    try:
        print("Installing unimatch...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            'git+https://github.com/autonomousvision/unimatch.git',
            '--quiet'
        ])
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to install unimatch: {e}")


def load_anycam_model():
    """Clone AnyCam repo with submodules and load model locally."""
    import os, subprocess
    print("Cloning or updating AnyCam repository with submodules...")
    cache_dir = torch.hub.get_dir()
    repo_dir = os.path.join(cache_dir, 'Brummi_anycam')
    if not os.path.isdir(repo_dir):
        subprocess.check_call([
            'git', 'clone', '--recursive',
            'https://github.com/Brummi/anycam.git', repo_dir
        ])
    else:
        subprocess.check_call([
            'git', 'submodule', 'update', '--init', '--recursive'
        ], cwd=repo_dir)

    print("Loading AnyCam model from local repository...")
    try:
        anycam = torch.hub.load(
            repo_dir,
            'AnyCam',
            source='local',
            version="1.0",
            training_variant="seq8",
            pretrained=True
        )
        print("AnyCam model loaded successfully from local repo")
        return anycam.cuda() if torch.cuda.is_available() else anycam.cpu()
    except Exception as e:
        raise RuntimeError(f"Failed to load AnyCam model from local repo: {e}")


def process_frames_with_anycam(anycam, frames: List[np.ndarray], ba_refinement: bool = False) -> Dict[str, Any]:
    """
    Process frames through AnyCam and return results.
    """
    print(f"Processing {len(frames)} frames through AnyCam...")
    print(f"Bundle adjustment refinement: {ba_refinement}")
    print(f"Frame shape: {frames[0].shape}")
    
    try:
        # Process frames through AnyCam
        results = anycam.process_video(frames, ba_refinement=ba_refinement)
        
        print("AnyCam processing completed successfully")
        print(f"Results keys: {list(results.keys())}")
        
        # Print result shapes/info
        if "trajectory" in results:
            print(f"Trajectory shape: {results['trajectory'].shape if hasattr(results['trajectory'], 'shape') else type(results['trajectory'])}")
        if "depths" in results:
            print(f"Depths shape: {results['depths'].shape if hasattr(results['depths'], 'shape') else type(results['depths'])}")
        if "uncertainties" in results:
            print(f"Uncertainties shape: {results['uncertainties'].shape if hasattr(results['uncertainties'], 'shape') else type(results['uncertainties'])}")
        if "projection_matrix" in results:
            print(f"Projection matrix shape: {results['projection_matrix'].shape if hasattr(results['projection_matrix'], 'shape') else type(results['projection_matrix'])}")
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"AnyCam processing failed: {e}")


def save_results(results: Dict[str, Any], output_dir: str, input_video_name: str, frames: Optional[List[np.ndarray]] = None):
    """
    Save AnyCam results to output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create base filename from input video
    base_name = os.path.splitext(os.path.basename(input_video_name))[0]
    
    saved_files = []
    
    # Save trajectory (camera poses)
    if "trajectory" in results:
        trajectory_path = os.path.join(output_dir, f"{base_name}_trajectory.pt")
        torch.save(results["trajectory"], trajectory_path)
        saved_files.append(trajectory_path)
        print(f"Saved trajectory: {trajectory_path}")
    
    # Save depths
    if "depths" in results:
        depths_path = os.path.join(output_dir, f"{base_name}_depths.pt")
        torch.save(results["depths"], depths_path)
        saved_files.append(depths_path)
        print(f"Saved depths: {depths_path}")
    
    # Save uncertainties
    if "uncertainties" in results:
        uncertainties_path = os.path.join(output_dir, f"{base_name}_uncertainties.pt")
        torch.save(results["uncertainties"], uncertainties_path)
        saved_files.append(uncertainties_path)
        print(f"Saved uncertainties: {uncertainties_path}")
    
    # Save projection matrix
    if "projection_matrix" in results:
        projection_path = os.path.join(output_dir, f"{base_name}_projection_matrix.pt")
        torch.save(results["projection_matrix"], projection_path)
        saved_files.append(projection_path)
        print(f"Saved projection matrix: {projection_path}")
    # Save video tensor if frames provided
    if frames is not None:
        # Convert list of HxWx3 arrays to tensor of shape (n,3,H,W)
        video_tensor = torch.stack([torch.from_numpy(f).permute(2,0,1) for f in frames])
        video_path = os.path.join(output_dir, f"{base_name}_video.pt")
        torch.save(video_tensor, video_path)
        saved_files.append(video_path)
        print(f"Saved video tensor: {video_path}")

        # Save RGB frames resized to depth resolution
        if "depths" in results:
            _, _, Hd, Wd = results["depths"].shape
            video_resized = torch.nn.functional.interpolate(video_tensor, size=(Hd, Wd), mode="bilinear", align_corners=False)
            resized_video_path = os.path.join(output_dir, f"{base_name}_video_resized.pt")
            torch.save(video_resized, resized_video_path)
            saved_files.append(resized_video_path)
            print(f"Saved resized video tensor: {resized_video_path}")
    
    # Save metadata
    metadata = {
        "input_video": input_video_name,
        "num_frames": len(results.get("depths", [])) if "depths" in results else None,
        "saved_files": saved_files,
        "results_keys": list(results.keys())
    }
    if frames is not None:
        metadata["video_tensor"] = video_path
    
    metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata: {metadata_path}")
    return saved_files


def main():
    args = parse_args()
    
    # Determine list of videos
    video_list = []
    if args.input_video:
        video_list = args.input_video
    else:
        print("Error: No input videos provided")
        sys.exit(1)
    # Validate each input video
    for vid in video_list:
        if not os.path.isfile(vid):
            print(f"Error: Input video file does not exist: {vid}")
            sys.exit(1)
    
    print("AnyCam Hub Processor")
    print("====================")
    print(f"Input videos: {', '.join(video_list)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Bundle adjustment: {args.ba_refinement}")
    print(f"Max frames: {args.max_frames or 'All'}")
    print(f"Resize height: {args.resize_height or 'Original'}")
    print()
    
    try:
        # Load AnyCam model once
        anycam = load_anycam_model()
        all_saved = []
        for vid in video_list:
            # Extract a batch of frames
            frames = extract_frames_with_ffmpeg(
                vid,
                max_frames=args.batch_size,
                resize_height=args.resize_height
            )
            # Process frames
            results = process_frames_with_anycam(anycam, frames, args.ba_refinement)
            # Save into subfolder per video
            base = os.path.splitext(os.path.basename(vid))[0]
            out_dir = os.path.join(args.output_dir, base)
            saved = save_results(results, out_dir, vid, frames)
            all_saved.extend(saved)
            print(f"\nFinished processing {vid}, saved {len(saved)} files to {out_dir}\n")
        print(f"\nAll videos processed successfully. Total output files: {len(all_saved)}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
