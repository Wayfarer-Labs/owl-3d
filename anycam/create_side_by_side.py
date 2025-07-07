#!/usr/bin/env python3
"""
Script to create a side-by-side video of RGB frames and depth maps.
Loads `<base>_video.pt` (shape n,3,H,W) and `<base>_depths.pt` (shape n,H,W) from an input directory,
visualizes depth with a colormap, concatenates each RGB frame with its depth, and writes out a video.
"""
import os
import argparse
import torch
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Create side-by-side video of RGB and depth")
    parser.add_argument("--input_dir", required=True, help="Directory containing *_video.pt and *_depths.pt")
    parser.add_argument("--base_name", required=True, help="Base name of the files (e.g., 'halo')")
    parser.add_argument("--output_video", required=True, help="Path to output video file (.mp4)")
    parser.add_argument("--fps", type=float, default=10.0, help="Target frames per second for output video")
    parser.add_argument("--colormap", type=str, default="JET", help="OpenCV colormap for depth (e.g., JET, HOT)")
    args = parser.parse_args()

    # Load tensors
    video_path = os.path.join(args.input_dir, f"{args.base_name}_video_resized.pt")
    depth_path = os.path.join(args.input_dir, f"{args.base_name}_depths.pt")
    if not os.path.isfile(video_path) or not os.path.isfile(depth_path):
        raise FileNotFoundError("Required files not found in input directory")

    video = torch.load(video_path)
    depths = torch.load(depth_path)
    print(f"Loaded video shape: {video.shape}, depths shape: {depths.shape}")
    n, c, H, W = video.shape

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output_video, fourcc, args.fps, (W*2, H))

    # Get OpenCV colormap constant
    cmap_attr = f'COLORMAP_{args.colormap.upper()}'
    if not hasattr(cv2, cmap_attr):
        raise ValueError(f"Invalid colormap: {args.colormap}")
    cmap = getattr(cv2, cmap_attr)

    for i in range(n):
        # RGB frame: convert tensor to numpy HxWx3 uint8
        frame = video[i].permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Depth: normalize to 0-255 and ensure uint8 single-channel
        depth = depths[i].cpu().numpy().astype(np.float32)
        # Remove any singleton channel dimension
        depth = np.squeeze(depth)
        # Normalize using OpenCV to guarantee CV_8UC1 format
        dnorm = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dnorm = dnorm.astype(np.uint8)

        print(dnorm.shape, dnorm.dtype)
        print(cmap, type(cmap))
        depth_color = cv2.applyColorMap(dnorm, cmap)

        # Concatenate side by side
        combined = np.concatenate((frame_bgr, depth_color), axis=1)
        out.write(combined)

    out.release()
    print(f"Saved side-by-side video to {args.output_video}")

if __name__ == "__main__":
    main()
