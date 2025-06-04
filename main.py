#!/usr/bin/env python3
"""
tensorfy_and_chunk.py

1. Traverses an input root directory of the form:
       <input_root>/1K/<sc    scene_hash = os.path.basename(scene_folder.rstrip("/"))
    image_folder = os.path.join(scene_folder, "images_4")
    pose_file   = os.path.join(scene_folder, "transforms.json")

    if not os.path.isdir(image_folder):
        raise RuntimeError(f"Expected an "images_4/" subfolder in {scene_folder}")
    if not os.path.isfile(pose_file):
        raise RuntimeError(f"Expected a "transforms.json" file in {scene_folder}")id>/
   where each <scene_hashid> folder contains:
       - a subfolder “images/” with image files (e.g. frame0001.jpg, frame0002.jpg, …)
       - a “poses.json” file containing camera poses (as a list or array)

2. For each scene_hashid:
      • Loads all images (sorted lexicographically), turns them into a tensor of shape
          (num_frames, 3, H, W)  (dtype=torch.uint8)
      • Loads “poses.json” and turns it into a float32 tensor of shape (num_frames, …)
      • Saves a single “<scene_hashid>.pt” that contains
          {
            "images":  Tensor [num_framesx3xHxW],
            "poses":   Tensor [num_framesx…]
          }

3. After producing all “.pt” files in `<output_root>/tensors/`, it groups them into N tarballs
   such that each tarball is no larger than ~100 MB (on disk).

Usage:
    python tensorfy_and_chunk.py \
       --input_root PATH/TO/DATASET_ROOT \
       --output_root ./processed \
       --chunk_size_mb 100

You can adjust --chunk_size_mb if you want smaller/larger tar files.
"""

import os
import sys
import json
import argparse
import tarfile
from glob import glob
from math import floor

from tqdm import tqdm
import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(
        description="Tensorfy each scene (images + poses.json) and bundle into ~100 MB tars"
    )
    p.add_argument(
        "--input_root", required=True,
        help="Root of your dataset. Must contain a subfolder “1K/” with one folder per scene_id."
    )
    p.add_argument(
        "--output_root", required=True,
        help="Where to write “tensors/” and “archives/”."
    )
    p.add_argument(
        "--chunk_size_mb", type=float, default=100.0,
        help="Target maximum size (in MB) per tar file. Defaults to 100 MB."
    )
    return p.parse_args()


def load_images_as_tensor(image_folder):
    """
    Given a folder of images (jpg/png), sorted lexicographically, load each into a
    torch.uint8 tensor of shape (3, H, W), stack into (N, 3, H, W).
    """
    # glob for .jpg and .png (case-insensitive)
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    all_files = []
    for pat in patterns:
        all_files += glob(os.path.join(image_folder, pat))
    if len(all_files) == 0:
        raise RuntimeError(f"No image files found under {image_folder}")

    all_files = sorted(all_files)
    tensors = []
    for img_path in all_files:
        with Image.open(img_path) as img:
            img = img.convert("RGB")  # ensure 3-channel
            arr = torch.from_numpy(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())))
            # Pillow's raw buffer is H×W×3 in “RGB” byte order; we need to reshape then permute
            w, h = img.size
            arr = arr.view(h, w, 3)           # shape (H, W, 3), dtype=torch.uint8
            arr = arr.permute(2, 0, 1).contiguous()  # (3, H, W)
            tensors.append(arr)
    # Stack into (N, 3, H, W)
    video_tensor = torch.stack(tensors, dim=0)
    return video_tensor


def load_poses_as_tensor(json_path):
    """
    Load transforms.json, which contains a more complex structure including camera parameters
    and frames with transform_matrix information. We extract the frames' transform_matrix data
    and the applied_transform, and convert to torch.float32 tensor.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Extract the transform matrices from each frame
    transform_matrices = []
    for frame in data.get("frames", []):
        if "transform_matrix" in frame:
            transform_matrices.append(frame["transform_matrix"])
    
    # Also include the applied_transform if present
    if "applied_transform" in data:
        # We might want to process this separately or include it with each frame
        applied_transform = data["applied_transform"]
    
    # Convert to torch tensor (float32)
    poses_tensor = torch.tensor(transform_matrices, dtype=torch.float32)
    return poses_tensor


def save_scene_tensor(scene_folder, out_tensor_folder):
    """
    For a single scene_folder (…/1K/<scene_hashid>/):
      - load “images/” → video_tensor (Nx3xHxW)
      - load “poses.json” → poses_tensor
      - save torch.save({"images":…, "poses":…}, out_tensor_folder/<scene_hashid>.pt)
    Returns the path to the .pt file.
    """
    scene_hash = os.path.basename(scene_folder.rstrip("/"))
    image_folder = os.path.join(scene_folder, "images")
    pose_file   = os.path.join(scene_folder, "transforms.json")

    if not os.path.isdir(image_folder):
        raise RuntimeError(f"Expected an “images/” subfolder in {scene_folder}")
    if not os.path.isfile(pose_file):
        raise RuntimeError(f"Expected a “transforms.json” file in {scene_folder}")

    # 1) load images
    video_tensor = load_images_as_tensor(image_folder)  # shape: (N, 3, H, W)

    # 2) load poses
    poses_tensor = load_poses_as_tensor(pose_file)      # shape: (N, …)

    # 3) sanity check: same number of frames?
    if poses_tensor.shape[0] != video_tensor.shape[0]:
        print(
            f"Warning: scene {scene_hash}: #images={video_tensor.shape[0]} "
            f"but poses.shape[0]={poses_tensor.shape[0]}. Proceeding anyway."
        )

    # 4) save
    os.makedirs(out_tensor_folder, exist_ok=True)
    out_path = os.path.join(out_tensor_folder, f"{scene_hash}.pt")
    torch.save(video_tensor, out_path)  # save video tensor first
        # "images": video_tensor,
        # "poses": poses_tensor
        # , out_path)

    return out_path


def chunk_and_tar(pt_paths, tar_folder, chunk_size_bytes):
    """
    Given a list of file paths (the per-scene .pt files), group them into tars so that
    each tar's total size (sum of the file sizes) is ≤ chunk_size_bytes (as best as possible).
    We name them: archive_000.tar, archive_001.tar, etc.
    """
    os.makedirs(tar_folder, exist_ok=True)

    current_tar = None
    current_sum = 0
    tar_index = 0

    def _open_new_tar(idx):
        tar_path = os.path.join(tar_folder, f"archive_{idx:03d}.tar")
        return tarfile.open(tar_path, mode="w")

    # sort by filename so chunks are deterministic
    pt_paths = sorted(pt_paths)

    for pt_path in tqdm(pt_paths, desc="Chunking into tarballs"):
        fsize = os.path.getsize(pt_path)
        # If adding this file would exceed chunk_size, start a new tar (unless current is empty)
        if current_tar is None:
            current_tar = _open_new_tar(tar_index)
            current_sum = 0

        if current_sum + fsize > chunk_size_bytes and current_sum > 0:
            # close current and start next
            current_tar.close()
            tar_index += 1
            current_tar = _open_new_tar(tar_index)
            current_sum = 0

        # add the .pt file under its basename (no directories) so that tar content is flat
        arcname = os.path.basename(pt_path)
        current_tar.add(pt_path, arcname=arcname)
        current_sum += fsize

    # after loop, close the last tar
    if current_tar is not None:
        current_tar.close()


if __name__ == "__main__":
    args = parse_args()

    input_root  = os.path.abspath(args.input_root)
    output_root = os.path.abspath(args.output_root)
    chunk_bytes = int(floor(args.chunk_size_mb * 1024 * 1024))

    # 1) Gather all “<scene_hashid>” subfolders under input_root/1K/
    onek_folder = os.path.join(input_root, "1K")
    if not os.path.isdir(onek_folder):
        print(f"ERROR: could not find “{onek_folder}”. "
              f"Make sure --input_root is correct and contains a “1K/” subfolder.")
        sys.exit(1)

    scene_folders = sorted([
        os.path.join(onek_folder, d)
        for d in os.listdir(onek_folder)
        if os.path.isdir(os.path.join(onek_folder, d))
    ])
    if len(scene_folders) == 0:
        print(f"ERROR: found no subfolders under {onek_folder}")
        sys.exit(1)

    # 2) Prepare output subfolders
    tensors_folder = os.path.join(output_root, "tensors")
    archives_folder = os.path.join(output_root, "archives")
    os.makedirs(tensors_folder, exist_ok=True)
    os.makedirs(archives_folder, exist_ok=True)

    # 3) For each scene, load images + poses.json → save one .pt
    pt_paths = []
    print(f"Processing {len(scene_folders)} scenes…")
    for scene_dir in tqdm(scene_folders, desc="Scenes"):
        try:
            pt_file = save_scene_tensor(scene_dir, tensors_folder)
            pt_paths.append(pt_file)
        except Exception as e:
            print(f"  → Skipping {scene_dir} due to error: {e}")

    # 4) Once all .pt files exist, chunk them into ~100 MB tar files
    print(f"\nNow chunking {len(pt_paths)} “.pt” files into ≈{args.chunk_size_mb} MB tarballs…")
    chunk_and_tar(pt_paths, archives_folder, chunk_bytes)

    print("\nDone! You'll find:")
    print(f"  • One “.pt” file per scene under  {tensors_folder}/")
    print(f"  • A series of tarballs under {archives_folder}/ (each ≤ {args.chunk_size_mb} MB)\n")
