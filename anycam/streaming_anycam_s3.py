#!/usr/bin/env python3
"""
stream_process_s3_videos_batches.py

Stream-download raw videos from S3 and process them in batches of frames.

Usage:
    python stream_process_s3_videos_batches.py \
        --bucket cod-yt-playlist-spmem-tensors \
        --prefix raw_videos/ \
        --ext mp4 \
        --frame-batch-size 32

Required Environment Variables:
    AWS_ACCESS_KEY_ID       - Your AWS access key ID
    AWS_SECRET_ACCESS_KEY   - Your AWS secret access key
    AWS_REGION             - AWS region (optional, defaults to us-east-1)
    AWS_ENDPOINT_URL       - Custom endpoint URL (optional)
"""

import os
import sys
import argparse
import time

import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from tqdm import tqdm
import imageio

import ffmpeg
import numpy as np
import torch
import io
from PIL import Image
from botocore.exceptions import ClientError
import math
import subprocess
import json

from anycam_hub_processor import process_frames_with_anycam, load_anycam_model  # Assuming this is your AnyCam processing function
from online_tsdf_pipeline import OnlineTSDFColorPipeline

# Create output directory from environment or default to 'outputs'
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_ply_bytes(data, colors=False) -> bytes:
    """
    Render an ASCII‐PLY in memory and return as bytes.
    - data: Nx3 or Nx6 array/tensor
    - colors: if True and data has ≥6 cols, treat cols 3–5 as RGB
    """
    # tensor → numpy
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    arr = np.asarray(data)
    N, D = arr.shape
    has_color = colors and D >= 6

    # header
    hdr = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        hdr += ["property uchar red", "property uchar green", "property uchar blue"]
    hdr.append("end_header")

    # build body
    lines = []
    for i in range(N):
        x, y, z = arr[i, :3]
        line = f"{x:.6f} {y:.6f} {z:.6f}"
        if has_color:
            rgb = arr[i, 3:6]
            if np.issubdtype(rgb.dtype, np.floating):
                rgb = (rgb * 255).clip(0, 255)
            rgb = rgb.astype(np.uint8)
            line += f" {rgb[0]} {rgb[1]} {rgb[2]}"
        lines.append(line)

    content = "\n".join(hdr + lines) + "\n"
    return content.encode("utf8")
# helper to upload per-frame data to S3
def upload_frame_data(s3_client, bucket, video_name, idx, frame_np, depth, pose, trajectory):
    idx_str = f"{idx:06d}"
    prefix = f"{video_name}/"
    keys = {
        'frame': prefix + f"frame_{idx_str}.png",
        'depth': prefix + f"depth_{idx_str}.pt",
        'pose': prefix + f"pose_3x3_{idx_str}.pt",
        'trajectory': prefix + f"trajectory_{idx_str}.pt"
    }
    # prepare tensors
    for kind, key in keys.items():
        # skip existing
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            print(f"Skipping existing {key}")
            continue
        except ClientError as e:
            if e.response['Error']['Code'] != '404': raise
        # upload content
        buf = io.BytesIO()
        if kind == 'frame':
            Image.fromarray(frame_np).save(buf, format='PNG')
            body = buf.getvalue()
            s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType='image/png')
        else:
            tensor = {'depth': depth, 'pose': pose, 'trajectory': trajectory}[kind] if kind in ['depth','pose','trajectory'] else None
            torch.save(tensor, buf)
            s3_client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
        # print(f"Uploaded {key}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Stream-download & batch-process videos from S3"
    )
    p.add_argument("--bucket",             required=True, help="S3 bucket name")
    p.add_argument("--prefix",             default="",     help="S3 prefix/folder for videos")
    p.add_argument("--ext",                default="mp4",  help="Video file extension filter")
    p.add_argument("--frame-batch-size",  type=int, default=50,
                   help="Number of frames per batch to process")
    p.add_argument("--target-bucket", required=True, help="S3 bucket for uploads")
    return p.parse_args()

def list_video_keys(s3, bucket, prefix, ext):
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(f".{ext.lower()}"):
                keys.append(key)
    return keys

def get_presigned_url(s3, bucket, key, expires=3600):
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )

def safe_probe(url):
    """
    Run ffprobe with reconnect flags via subprocess and retry on failure.
    """
    base_cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "2",
        url
    ]
    try:
                
        proc = subprocess.run(base_cmd, capture_output=True)
        if proc.returncode == 0:
    
            return json.loads(proc.stdout)
    except subprocess.CalledProcessError as e:
        print(f"ffprobe failed: {e.stderr.decode('utf8', errors='ignore')}")
        return None

def get_fps(url):
    # Run ffprobe and get JSON metadata
    reconnect_args = ['-reconnect', '1',
                      '-reconnect_streamed', '1',
                      '-reconnect_delay_max', '2']
    try:
        probe = safe_probe(url)
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr.decode('utf8', errors='ignore')}")

    # Find the first video stream
    video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
    if video_stream is None:
        raise RuntimeError('No video stream found')

    # r_frame_rate is a string like "30000/1001" or "25/1"
    num, den = video_stream['r_frame_rate'].split('/')
    fps = float(num) / float(den)
    return fps

import cv2
import open3d as o3d

def intrinsic_o3d_from_proj(P, width, height):
    fx, fy, cx, cy = extract_intrinsics_from_proj(P)
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

def extract_camera_params_from_proj(P):
    """
    Recover camera intrinsics and extrinsic pose from a 3×4 projection matrix P or 3×3 intrinsic matrix K.

    Args:
        P (np.ndarray or torch.Tensor): 3×4 projection matrix or 3×3 intrinsic matrix from AnyCam.

    Returns:
        intrinsics (dict): {'fx', 'fy', 'cx', 'cy'}
        extrinsic (np.ndarray): 4×4 world→camera homogeneous transform (identity if only K provided)
    """
    # Convert torch.Tensor to NumPy if necessary
    if not isinstance(P, np.ndarray):
        try:
            P = P.cpu().numpy()
        except AttributeError:
            P = np.array(P)
    
    print(f"Processing matrix with shape: {P.shape}")
    
    # Handle different input formats
    if P.shape == (3, 3):
        # Input is a 3x3 intrinsic matrix K
        print("Input is 3x3 intrinsic matrix")
        K = P.copy()
        
        # Normalize K so that K[2,2] == 1
        K = K / K[2, 2]
        
        # Extract intrinsics
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        
        intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        
        # Return identity extrinsic matrix since we don't have pose information
        extrinsic = np.eye(4, dtype=float)
        
        return intrinsics, extrinsic
        
    elif P.shape in [(3, 4), (4, 3)]:
        # Input is a 3x4 or 4x3 projection matrix
        print("Input is projection matrix")
        
        # Ensure P is 3x4
        if P.shape == (4, 3):
            P = P.T
        
        # Decompose P into cameraMatrix (K), rotation R, and translation t (homogeneous)
        K, R, t_hom, _, _, _, _ = cv2.decomposeProjectionMatrix(P)

        # Normalize K so that K[2,2] == 1
        K = K / K[2, 2]

        # Extract intrinsics
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

        # t_hom is 4×1 in homogeneous coords [tx, ty, tz, w]ᵀ
        # Convert to 3×1 translation vector by dividing by w
        t = (t_hom[:3] / t_hom[3]).reshape(3,)

        # Build 4×4 world-to-camera extrinsic: [ R | t; 0 0 0 1 ]
        world_to_cam = np.eye(4, dtype=float)
        world_to_cam[:3, :3] = R
        world_to_cam[:3,  3] = t
        
        # For TSDF integration, we need camera-to-world transform
        # Invert the world-to-camera matrix
        try:
            cam_to_world = np.linalg.inv(world_to_cam)
            print("Inverted world-to-camera to get camera-to-world transform")
        except np.linalg.LinAlgError:
            print("Warning: Could not invert pose matrix, using identity")
            cam_to_world = np.eye(4, dtype=float)

        return intrinsics, cam_to_world
    
    else:
        raise ValueError(f"Expected 3x3 intrinsic matrix, 3x4 or 4x3 projection matrix, got {P.shape}")


def process_fusion_batch(rgbs, depths, projection_matrix):
    """
    Process a batch of RGB and depth frames using TSDF fusion.
    
    Args:
        rgbs: List of RGB frames (numpy arrays)
        depths: List of depth maps (tensors or numpy arrays)
        projection_matrix: 3x3, 3x4 or 4x3 projection matrix
    
    Returns:
        bytes: PLY file content as bytes
    """
    print(f"Processing fusion batch with {len(rgbs)} RGB frames and {len(depths)} depth maps")
    
    # Validate inputs
    if len(rgbs) != len(depths):
        print(f"Warning: Mismatched frame counts - RGB: {len(rgbs)}, Depth: {len(depths)}")
        min_frames = min(len(rgbs), len(depths))
        rgbs = rgbs[:min_frames]
        depths = depths[:min_frames]
        print(f"Using first {min_frames} frames")

    # Create a scalable TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.004,          # 4mm voxels
        sdf_trunc=0.04,              # truncation distance
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    try:
        intrinsics_dict, base_extrinsic = extract_camera_params_from_proj(projection_matrix)
        print(f"Extracted intrinsics: {intrinsics_dict}")
        print(f"Base extrinsic determinant: {np.linalg.det(base_extrinsic[:3, :3]):.6f}")
    except Exception as e:
        print(f"Error extracting camera parameters: {e}")
        return b""

    for i, (frame, depth) in enumerate(zip(rgbs, depths)):
        try:
            # Convert numpy arrays to Open3D images
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            
            # Convert depth to proper format (ensure it's float32 and in meters)
            if hasattr(depth, 'cpu'):
                depth_np = depth.cpu().numpy().squeeze()
            else:
                depth_np = np.array(depth).squeeze()
            
            # Ensure depth is float32 for Open3D
            depth_np = depth_np.astype(np.float32)
            
            # Get depth dimensions
            if len(depth_np.shape) == 2:
                depth_height, depth_width = depth_np.shape
            else:
                print(f"Warning: Unexpected depth shape {depth_np.shape}, skipping frame {i}")
                continue
            
            # Get original frame dimensions for scaling intrinsics
            original_height, original_width = frame.shape[:2]
            
            # Resize RGB frame to match depth dimensions
            frame_resized = cv2.resize(frame, (depth_width, depth_height), interpolation=cv2.INTER_LINEAR)
            
            # Scale intrinsics to match the resized image
            scale_x = depth_width / original_width
            scale_y = depth_height / original_height
            
            scaled_fx = intrinsics_dict['fx'] * scale_x
            scaled_fy = intrinsics_dict['fy'] * scale_y
            scaled_cx = intrinsics_dict['cx'] * scale_x
            scaled_cy = intrinsics_dict['cy'] * scale_y
            
            print(f"Frame {i}: Original {original_width}x{original_height} -> Resized {depth_width}x{depth_height}")
            print(f"Intrinsics scaling: fx={intrinsics_dict['fx']:.1f}->{scaled_fx:.1f}, fy={intrinsics_dict['fy']:.1f}->{scaled_fy:.1f}")
            
            # Create Open3D images
            rgb_o3d = o3d.geometry.Image(frame_resized.astype(np.uint8))
            depth_o3d = o3d.geometry.Image(depth_np)
            
            # Create RGBD image
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
            )
            
            # Create camera intrinsics using the scaled parameters
            intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
                depth_width, depth_height,
                scaled_fx, scaled_fy, scaled_cx, scaled_cy
            )
            
            # Validate inputs before integration
            if not rgbd.color or not rgbd.depth:
                print(f"Warning: Invalid RGBD image for frame {i}, skipping")
                continue
                
            if scaled_fx <= 0 or scaled_fy <= 0:
                print(f"Warning: Invalid focal lengths fx={scaled_fx}, fy={scaled_fy} for frame {i}, skipping")
                continue
                
            if not np.isfinite(base_extrinsic).all():
                print(f"Warning: Non-finite values in extrinsic matrix for frame {i}, skipping")
                continue
                
            print(f"Using intrinsics: fx={scaled_fx:.1f}, fy={scaled_fy:.1f}, cx={scaled_cx:.1f}, cy={scaled_cy:.1f}")
            print(f"Extrinsic matrix shape: {base_extrinsic.shape}")
            print(f"RGBD image - color: {rgbd.color.width}x{rgbd.color.height}, depth: {rgbd.depth.width}x{rgbd.depth.height}")
            
            # Integrate into TSDF volume using extrinsic pose
            try:
                print(f"About to integrate frame {i+1}...")
                volume.integrate(rgbd, intrinsics_o3d, base_extrinsic)
                print(f"Successfully integrated frame {i+1}/{len(rgbs)}")
            except Exception as integrate_error:
                print(f"Error during TSDF integration for frame {i}: {integrate_error}")
                print(f"RGBD valid: color={rgbd.color is not None}, depth={rgbd.depth is not None}")
                print(f"Intrinsics valid: {intrinsics_o3d.width}x{intrinsics_o3d.height}")
                print(f"Extrinsic determinant: {np.linalg.det(base_extrinsic[:3, :3])}")
                continue
            
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue

    # Extract fused point cloud
    try:
        pcd = volume.extract_point_cloud()
        print(f"Extracted point cloud with {len(pcd.points)} points")
        
        # Convert to PLY bytes using our existing helper function
        # Get points and colors as numpy arrays
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Combine points and colors (colors are in [0,1] range, need to convert to [0,255])
        if len(colors) > 0 and len(points) > 0:
            colors_255 = (colors * 255).astype(np.uint8)
            point_cloud_data = np.hstack([points, colors_255])
            ply_bytes = get_ply_bytes(point_cloud_data, colors=True)
        elif len(points) > 0:
            ply_bytes = get_ply_bytes(points, colors=False)
        else:
            print("Warning: No points extracted from TSDF volume")
            ply_bytes = b""
        
        return ply_bytes
        
    except Exception as e:
        print(f"Error extracting point cloud: {e}")
        return b""

def process_batch(model, frames, batch_index, video_name, s3_client, target_bucket, start_idx=0, pipeline=None):
    """
    Placeholder for your batch processing logic.
    `frames` is a list/array of shape [batch_size, H, W, C]
    """
    print(f"    → processing batch {batch_index} ({len(frames)} frames) of {video_name}")
    # TODO: replace with your real batch work:
    # e.g. run inference on stack of frames, save outputs, etc.
    # Example: just save the batch as a video
    output_filename = os.path.join(OUTPUT_DIR, f"{video_name}_batch_{batch_index}.mp4")
    # print(f"    → saving batch {batch_index} to {output_filename}")
    # imageio.mimwrite(output_filename, frames, fps=30, quality=8)
    results = process_frames_with_anycam(model, frames)
    # update TSDF color pipeline
    projection_matrix = results.get('projection_matrix', [])
    print(f"Projection matrix type: {type(projection_matrix)}, shape/length: {getattr(projection_matrix, 'shape', len(projection_matrix) if hasattr(projection_matrix, '__len__') else 'unknown')}")
    
    # Handle different projection matrix formats
    if isinstance(projection_matrix, list) and len(projection_matrix) > 0:
        projection_matrix = projection_matrix[0]  # Take first matrix if it's a list
    elif isinstance(projection_matrix, torch.Tensor):
        projection_matrix = projection_matrix.cpu().numpy()
    elif not isinstance(projection_matrix, np.ndarray):
        print(f"Warning: Unexpected projection matrix format: {type(projection_matrix)}")
        return  # Skip this batch if we can't process the projection matrix
    
    # Validate projection matrix dimensions
    if hasattr(projection_matrix, 'shape'):
        print(f"Final projection matrix shape: {projection_matrix.shape}")
        if projection_matrix.shape not in [(3, 3), (3, 4), (4, 3)]:
            print(f"Warning: Invalid projection matrix shape {projection_matrix.shape}, expected (3,3), (3,4) or (4,3)")
            return
    
    depths = results.get('depths', [])
    rgbs = frames

    ply_bytes = process_fusion_batch(rgbs, depths, projection_matrix)
    
    # Only upload if we successfully generated PLY data
    if ply_bytes:
        # upload colored point cloud as tensor to S3
        s3_key = f"{video_name}/batch_{batch_index}_pointcloud.ply"
        s3_client.put_object(Bucket=target_bucket, Key=s3_key, Body=ply_bytes)
        print(f"Uploaded batch {batch_index} pointcloud to s3://{target_bucket}/{s3_key}")
    else:
        print(f"Warning: No PLY data generated for batch {batch_index}")

    # upload each frame's data
    # compute global frame index offset by start_idx
    # base_idx = start_idx + batch_index * len(frames)
    # for i, frame in tqdm(enumerate(frames), desc=f"Batch {batch_index+1}/{total_batches}", unit="frame"):
    # # for i, frame in enumerate(frames):
    #     global_idx = base_idx + i
    #     depth = results.get('depths')[i]
    #     proj = results.get('projection_matrix')
    #     # extract 3x3 pose
    #     pose = torch.from_numpy(proj)[:3, :3] if isinstance(proj, np.ndarray) else proj[:3, :3]
    #     traj = results.get('trajectory')[i]
    #     upload_frame_data(s3_client, target_bucket, video_name, global_idx, frame, depth, pose, traj)

def process_streaming_video(model, url, batch_size, s3_client, target_bucket):
    """
    Stream from `url` via ffmpeg, accumulate `batch_size` frames, then process each batch.
    """
    # Derive a nice name for logging
    video_name = url.split("/")[-1].split("?")[0]
    print(f"\n→ streaming {video_name}")

    # 1) Probe the stream to get its width/height
    # probe size with reconnect & retry
    probe = safe_probe(url)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width, height = int(video_stream['width']), int(video_stream['height'])
    # determine starting frame index from existing uploads
    prefix = f"{video_name}/frame_"
    resp = s3_client.list_objects_v2(Bucket=target_bucket, Prefix=prefix)
    existing = resp.get('Contents', [])
    if existing:
        idxs = [int(obj['Key'].split('_')[-1].split('.')[0]) for obj in existing]
        start_idx = max(idxs) + 1
        print(f"Resuming from frame index {start_idx}")
    else:
        start_idx = 0
    start_idx = 0
    # (fps-based batch counting removed)
    # compute batch count if nb_frames metadata is available
    if 'nb_frames' in video_stream and video_stream['nb_frames'].isdigit():
        total_frames = int(video_stream['nb_frames'])
        remaining = max(0, total_frames - start_idx)
        total_batches = math.ceil(remaining / batch_size)
        print(f"Remaining frames: {remaining}, batches: {total_batches} (batch size={batch_size})")
    else:
        print("Batch count unavailable (nb_frames missing)")

    fps = get_fps(url)
    start_seconds = start_idx / fps

    # 2) Launch ffmpeg as a subprocess, outputting rawvideo RGB24 to stdout
    # initialize TSDF color pipeline (identity intrinsics)
    intr = torch.eye(3)
    tsdfpipeline = OnlineTSDFColorPipeline(camera_intrinsics=intr, device='cuda' if torch.cuda.is_available() else 'cpu')
    # launch ffmpeg process, skipping to start_idx by frame number if resuming
    # reconnect flags
    recon_args = ['-reconnect', '1',
                  '-reconnect_streamed', '1',
                  '-reconnect_delay_max', '2']
    if start_idx > 0:
        process = (
            ffmpeg
            .input(url, ss=start_seconds)
            .filter('setpts', 'PTS-STARTPTS')
            .filter('scale', -1, 336)  # ensure correct resolution
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .global_args(*recon_args)
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
    else:
        process = (
            ffmpeg
            .input(url)
            .filter('scale', -1, 336)  # ensure correct resolution
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .global_args(*recon_args)
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

    frame_size = width * height * 3  # bytes per frame
    # initialize batch and batch counter
    batch = []
    batch_idx = 0
    # read frames
    while True:
        # read exactly one frame
        in_bytes = process.stdout.read(frame_size)
        if not in_bytes or len(in_bytes) < frame_size:
            break

        # turn bytes into H×W×3 uint8 numpy array
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape((height, width, 3))
        )

        batch.append(frame)
        if len(batch) >= batch_size:
            process_batch(model, batch, batch_idx, video_name, s3_client, target_bucket, start_idx, pipeline=tsdfpipeline)
            batch = []
            batch_idx += 1

    # final partial batch
    if batch:
        process_batch(model, batch, batch_idx, video_name, s3_client, target_bucket, start_idx, pipeline=tsdfpipeline)

    process.wait()
    # extract and upload static colored point cloud as a tensor to S3
    pc_color = tsdfpipeline.extract_colored_pointcloud(min_weight_threshold=0.1)
    ply_bytes = get_ply_bytes(pc_color, colors=True)

    s3_key = f"{video_name}/static_color_pointcloud.ply"
    s3_client.put_object(Bucket=target_bucket, Key=s3_key, Body=ply_bytes)
    print(f"Uploaded static colored pointcloud to s3://{target_bucket}/{s3_key}")

    if process.returncode != 0:
        err = process.stderr.read().decode('utf8', errors='ignore')
        print(f"ffmpeg exited {process.returncode}:\n{err}")
    else:
        print(f"   done {video_name}")


def main():
    args = parse_args()

    # AWS/S3 setup
    aws_cfg = {
        "aws_access_key_id":     os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "region_name":           os.getenv("AWS_REGION", "us-east-1"),
    }
    if os.getenv("AWS_ENDPOINT_URL"):
        aws_cfg["endpoint_url"] = os.getenv("AWS_ENDPOINT_URL")

    try:
        s3 = boto3.client("s3", **aws_cfg)
        s3.head_bucket(Bucket=args.bucket)
    except (NoCredentialsError, PartialCredentialsError):
        print("Error: AWS credentials missing or invalid.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error accessing bucket: {e}", file=sys.stderr)
        sys.exit(1)

    # List keys
    print(f"Listing .{args.ext} files in s3://{args.bucket}/{args.prefix}")
    keys = list_video_keys(s3, args.bucket, args.prefix, args.ext)
    if not keys:
        print("No videos found. Exiting.")
        return

    model = load_anycam_model()

    if model is None:
        print("Failed to load AnyCam model. Exiting.")
        return

    # Stream & batch-process each
    for key in tqdm(keys, desc="Videos", unit="video"):
        url = get_presigned_url(s3, args.bucket, key)
        process_streaming_video(model, url, batch_size=args.frame_batch_size,
                                s3_client=s3, target_bucket=args.target_bucket)

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nAll done! Total time: {time.time() - start:.2f}s")
