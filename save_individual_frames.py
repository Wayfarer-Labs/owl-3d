#!/usr/bin/env python3
"""
save_individual_frames.py

This script provides an alternative to `save_scene_tensor` that:
1. Processes input directory with the same structure as the main script
2. For each scene and each frame, it saves:
   - 0000.png: The frame image
   - 0000.ext.pt: The extrinsics parameters (pose) as a tensor
   - 0000.int.pt: The intrinsics parameters as a tensor
3. After processing all scenes, it groups them into N tarballs
   such that each tarball is no larger than the specified size.

Usage:
    python save_individual_frames.py \
       --input_root PATH/TO/DATASET_ROOT \
       --output_root ./processed_frames \
       --chunk_size_mb 100
"""

import os
import sys
import json
import argparse
from glob import glob
import shutil
import tarfile
from math import floor

from tqdm import tqdm
import torch
from PIL import Image

def parse_args():
    p = argparse.ArgumentParser(
        description="Save individual frames with their intrinsics and extrinsics"
    )
    p.add_argument(
        "--input_root", required=True,
        help="Root of your dataset. Must contain a subfolder 1K with one folder per scene_id."
    )
    p.add_argument(
        "--output_root", required=True,
        help="Where to write frame data"
    )
    p.add_argument(
        "--chunk_size_mb", type=float, default=100.0,
        help="Target maximum size (in MB) per tar file. Defaults to 100 MB."
    )
    return p.parse_args()

def load_poses_as_tensor(json_path):
    """
    Load transforms.json, which contains camera parameters and transform matrices.
    
    Returns a tuple of:
    - camera_intrinsics: torch.float32 tensor with [fx, fy, cx, cy]
    - poses_tensor: torch.float32 tensor with transform matrices for each frame
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Extract camera intrinsics
    fl_x = data.get("fl_x")
    fl_y = data.get("fl_y")
    cx = data.get("cx")
    cy = data.get("cy")
    w = data.get("w")  # Image width
    h = data.get("h")  # Image height
    
    if None in (fl_x, fl_y, cx, cy, w, h):
        raise RuntimeError(f"Missing camera intrinsics in {json_path}")
    
    # Create camera intrinsics tensor [fx, fy, cx, cy]
    camera_intrinsics = torch.tensor([fl_x, fl_y, cx, cy], dtype=torch.float32)
    
    # Extract the transform matrices from each frame
    transform_matrices = []
    for frame in data.get("frames", []):
        if "transform_matrix" in frame:
            transform_matrices.append(frame["transform_matrix"])
    
    # Convert to torch tensor (float32)
    poses_tensor = torch.tensor(transform_matrices, dtype=torch.float32)
    
    return camera_intrinsics, poses_tensor, (h, w)

def save_frames_individually(scene_folder, out_frames_folder):
    """
    For a single scene_folder (…/1K/<scene_hashid>/):
      - Create an output directory for this scene
      - For each image in "images_4/":
        - Copy the image to out_dir as "XXXX.png" (with zero padding)
        - Save the corresponding pose as "XXXX.ext.pt"
        - Save the intrinsics as "XXXX.int.pt"
    """
    scene_hash = os.path.basename(scene_folder.rstrip("/"))
    image_folder = os.path.join(scene_folder, "images_4")
    pose_file = os.path.join(scene_folder, "transforms.json")

    if not os.path.isdir(image_folder):
        raise RuntimeError(f"Expected an images_4/ subfolder in {scene_folder}")
    if not os.path.isfile(pose_file):
        raise RuntimeError(f"Expected a transforms.json file in {scene_folder}")

    # Create output directory for this scene
    scene_out_dir = os.path.join(out_frames_folder, scene_hash)
    os.makedirs(scene_out_dir, exist_ok=True)
    
    # Load camera intrinsics and extrinsics
    camera_intrinsics, poses_tensor, (json_H, json_W) = load_poses_as_tensor(pose_file)
    
    # Get image files, sorted
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for pat in patterns:
        image_files += glob(os.path.join(image_folder, pat))
    image_files = sorted(image_files)
    
    # Check if the number of poses matches the number of images
    if len(image_files) != poses_tensor.shape[0]:
        print(
            f"Warning: scene {scene_hash}: #images={len(image_files)} "
            f"but poses.shape[0]={poses_tensor.shape[0]}. Using the minimum number."
        )
    
    # Process each frame
    num_frames = min(len(image_files), poses_tensor.shape[0])
    for i in tqdm(range(num_frames), desc=f"Processing frames for {scene_hash}"):
        # Format frame number with leading zeros (e.g., 0000, 0001, etc.)
        frame_id = f"{i:04d}"
        
        # 1. Copy the image
        img_src = image_files[i]
        img_ext = os.path.splitext(img_src)[1]  # Get the file extension
        img_dest = os.path.join(scene_out_dir, f"{frame_id}{img_ext}")
        shutil.copy2(img_src, img_dest)
        
        # 2. Save the extrinsics (pose)
        ext_tensor = poses_tensor[i]  # Shape: (4, 4)
        ext_path = os.path.join(scene_out_dir, f"{frame_id}.ext.pt")
        torch.save(ext_tensor, ext_path)
        
        # 3. Save the intrinsics
        int_path = os.path.join(scene_out_dir, f"{frame_id}.int.pt")
        torch.save(camera_intrinsics, int_path)
    
    print(f"Saved {num_frames} frames for scene {scene_hash}")
    return scene_out_dir

def chunk_and_tar(scene_dirs, archives_folder, chunk_size_bytes):
    """
    Given a list of scene directories, group them into tars so that
    each tar's total size (sum of the dir sizes) is ≤ chunk_size_bytes (as best as possible).
    We name them: archive_000.tar, archive_001.tar, etc.
    """
    os.makedirs(archives_folder, exist_ok=True)

    current_tar = None
    current_sum = 0
    tar_index = 0

    def _open_new_tar(idx):
        tar_path = os.path.join(archives_folder, f"archive_{idx:03d}.tar")
        return tarfile.open(tar_path, mode="w")

    # sort by directory name so chunks are deterministic
    scene_dirs = sorted(scene_dirs)

    for scene_dir in tqdm(scene_dirs, desc="Chunking into tarballs"):
        # Get the size of the scene directory
        scene_size = 0
        for dirpath, dirnames, filenames in os.walk(scene_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                scene_size += os.path.getsize(fp)

        # If adding this scene would exceed chunk_size, start a new tar (unless current is empty)
        if current_tar is None:
            current_tar = _open_new_tar(tar_index)
            current_sum = 0

        if current_sum + scene_size > chunk_size_bytes and current_sum > 0:
            # close current and start next
            current_tar.close()
            tar_index += 1
            current_tar = _open_new_tar(tar_index)
            current_sum = 0

        # add the scene directory under its basename
        scene_name = os.path.basename(scene_dir)
        current_tar.add(scene_dir, arcname=scene_name)
        current_sum += scene_size

    # after loop, close the last tar
    if current_tar is not None:
        current_tar.close()

def process_all_scenes(input_root, output_root, chunk_size_bytes=None):
    """
    Process all scenes in the input directory and save frames individually.
    Optionally chunk and tar the scenes if chunk_size_bytes is provided.
    """
    # Gather all "<scene_hashid>" subfolders under input_root/1K/
    onek_folder = os.path.join(input_root, "1K")
    if not os.path.isdir(onek_folder):
        print(f"ERROR: could not find {onek_folder}. "
              f"Make sure --input_root is correct and contains a 1K/ subfolder.")
        sys.exit(1)

    scene_folders = sorted([
        os.path.join(onek_folder, d)
        for d in os.listdir(onek_folder)
        if os.path.isdir(os.path.join(onek_folder, d))
    ])
    
    if len(scene_folders) == 0:
        print(f"ERROR: found no subfolders under {onek_folder}")
        sys.exit(1)

    # Process each scene
    processed_scenes = []
    print(f"Processing {len(scene_folders)} scenes…")
    for scene_dir in scene_folders:
        try:
            scene_out_dir = save_frames_individually(scene_dir, output_root)
            processed_scenes.append(scene_out_dir)
        except Exception as e:
            print(f"  → Skipping {scene_dir} due to error: {e}")
    
    print(f"\nProcessed {len(processed_scenes)} scenes, saving individual frames for each")

    # If chunk_size_bytes is provided, create archive tarballs
    if chunk_size_bytes:
        # Create archives folder - make it consistent with main.py structure
        archives_folder = os.path.join(output_root, "archives")
        print(f"\nNow chunking {len(processed_scenes)} scene directories into ≈{chunk_size_bytes/(1024*1024):.1f} MB tarballs…")
        chunk_and_tar(processed_scenes, archives_folder, chunk_size_bytes)
        return processed_scenes, archives_folder
    
    return processed_scenes, None

if __name__ == "__main__":
    args = parse_args()
    
    input_root = os.path.abspath(args.input_root)
    output_root = os.path.abspath(args.output_root)
    chunk_bytes = int(floor(args.chunk_size_mb * 1024 * 1024)) if args.chunk_size_mb else None
    
    # Prepare output directories - match structure in main.py
    frames_folder = os.path.join(output_root, "frames")
    os.makedirs(frames_folder, exist_ok=True)
    
    processed_scenes, archives_folder = process_all_scenes(input_root, frames_folder, chunk_bytes)
    
    print("\nDone! You'll find:")
    print(f"  • Individual frames and their parameters under {frames_folder}/<scene_hash>/")
    if archives_folder:
        print(f"  • A series of tarballs under {archives_folder}/ (each ≤ {args.chunk_size_mb} MB)")
