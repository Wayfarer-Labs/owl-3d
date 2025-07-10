#!/usr/bin/env python3
"""
test_s3_stream_vis.py

Test script to stream a limited number of frames from an S3-hosted video,
process them through AnyCam and the TSDF color pipeline,
and save a colored point cloud as a PLY file locally.

Usage:
    python test_s3_stream_vis.py --bucket BUCKET --key KEY [--num_frames N] [--output FILE.pLY]
"""
import argparse
import subprocess
import json
import os

import boto3
import ffmpeg
import numpy as np
import torch

from anycam_hub_processor import process_frames_with_anycam, load_anycam_model
from streaming_anycam_s3 import safe_probe, get_fps, get_presigned_url, get_ply_bytes, process_fusion_batch
from online_tsdf_pipeline import OnlineTSDFColorPipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--bucket', required=True, help='Source S3 bucket')
    p.add_argument('--key', required=True, help='S3 object key for video (e.g., path/to/video.mp4)')
    p.add_argument('--num_frames', type=int, default=50, help='Number of frames to stream')
    p.add_argument('--output', type=str, default='output.ply', help='Output PLY file path')
    return p.parse_args()


def main():
    args = parse_args()

    # AWS client
    aws_cfg = {
        'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'region_name': os.getenv('AWS_REGION', 'us-east-1')
    }
    if os.getenv('AWS_ENDPOINT_URL'):
        aws_cfg['endpoint_url'] = os.getenv('AWS_ENDPOINT_URL')
    s3 = boto3.client('s3', **aws_cfg)

    # Get presigned URL
    url = get_presigned_url(s3, args.bucket, args.key)
    print(f"Streaming from: {url}")

    # Probe video for dimensions
    probe = safe_probe(url)
    video_stream = next(s for s in probe['streams'] if s['codec_type']=='video')
    width, height = int(video_stream['width']), int(video_stream['height'])

    # Determine FPS
    fps = get_fps(url)
    print(f"Video resolution: {width}x{height} @ {fps:.2f} FPS")

    # Launch ffmpeg to stream raw RGB frames
    process = (
        ffmpeg.input(url, ss=90)
            #   .filter('scale', width, height)
              .output('pipe:', format='rawvideo', pix_fmt='rgb24')
              .global_args('-reconnect', '1', '-reconnect_streamed', '1', '-reconnect_delay_max', '2')
              .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    frame_size = width * height * 3
    frames = []
    print(f"Capturing {args.num_frames} frames of size {width}x{height}...")
    for i in range(args.num_frames):
        in_bytes = process.stdout.read(frame_size)
        print(f'.', end='', flush=True)
        if not in_bytes or len(in_bytes)<frame_size:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape((height, width, 3))
        frames.append(frame)
    # close pipes and terminate ffmpeg process gracefully
    process.stdout.close()
    process.stderr.close()
    try:
        # allow ffmpeg to clean up
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
    print(f"Captured {len(frames)} frames from stream.")

    # create side-by-side visualization of RGB and depth
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL is required for side-by-side visualization. Install with: pip install pillow")
    # prepare depth maps from AnyCam results (compute separately)
    # run temporary AnyCam process to get depths
    temp_model = load_anycam_model()
    temp_results = process_frames_with_anycam(temp_model, frames)
    temp_depths = temp_results.get('depths', [])
    depth_maps = [d[0].cpu().numpy() if hasattr(d, 'cpu') else d[0] for d in temp_depths]
    trajs = temp_results.get('trajectory', [])

    viz_dir = os.path.splitext(args.output)[0] + '_frames'
    os.makedirs(viz_dir, exist_ok=True)
    for i, (rgb, depth) in enumerate(zip(frames, depth_maps)):
        # normalize depth to [0,255]
        H_d, W_d = depth.shape
        dmin, dmax = depth.min(), depth.max()
        norm = (depth - dmin) / (dmax - dmin + 1e-8)
        gray = (norm * 255).astype(np.uint8)
        depth_img = np.stack([gray]*3, axis=2)
        rgb_img = Image.fromarray(rgb)
        rgb_resized = rgb_img.resize((W_d, H_d), Image.BILINEAR)
        rgb_arr = np.array(rgb_resized)

        side = np.hstack((rgb_arr, depth_img))
        Image.fromarray(side).save(os.path.join(viz_dir, f'side_{i:03d}.png'))
    print(f"Saved side-by-side RGB/depth frames to {viz_dir}")
    rgb_maps = frames
    # Convert and invert poses: ensure camera-to-world transforms
    poses = []
    for p in trajs:
        # load pose matrix
        pose_mat = torch.from_numpy(p) if isinstance(p, np.ndarray) else p.clone()
        # invert if input is world-to-camera
        cam2world = torch.inverse(pose_mat)
        poses.append(cam2world)
    pose_tensor = torch.stack(poses).float()
    # collect per-frame projection matrices from AnyCam results
    proj_ms = temp_results.get('projection_matrix', [])
    # convert to numpy arrays
    projection_matrices = [pm.cpu().numpy() if torch.is_tensor(pm) else pm for pm in proj_ms]
    
    # Use the first projection matrix for fusion (assuming camera intrinsics are consistent)
    if projection_matrices:
        projection_matrix = projection_matrices[0]
        print(f"Using projection matrix with shape: {projection_matrix.shape}")
    else:
        print("Warning: No projection matrix available, using identity")
        projection_matrix = np.eye(3)
    
    # Use Open3D TSDF fusion pipeline
    ply_bytes = process_fusion_batch(frames, depth_maps, projection_matrix)
    with open(args.output, 'wb') as f:
        f.write(ply_bytes)
    print(f"Saved colored point cloud to {args.output}")


if __name__=='__main__':
    main()
