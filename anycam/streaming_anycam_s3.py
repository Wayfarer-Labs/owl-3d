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

from anycam_hub_processor import process_frames_with_anycam, load_anycam_model  # Assuming this is your AnyCam processing function

# Create output directory from environment or default to 'outputs'
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        print(f"Uploaded {key}")


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

def process_batch(model, frames, batch_index, video_name, s3_client, target_bucket, start_idx=0):
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

     # upload each frame's data
     # compute global frame index offset by start_idx
    base_idx = start_idx + batch_index * len(frames)
    for i, frame in enumerate(frames):
        global_idx = base_idx + i
        depth = results.get('depths')[i]
        proj = results.get('projection_matrix')
        # extract 3x3 pose
        pose = torch.from_numpy(proj)[:3, :3] if isinstance(proj, np.ndarray) else proj[:3, :3]
        traj = results.get('trajectory')[i]
        # upload_frame_data(s3_client, target_bucket, video_name, global_idx, frame, depth, pose, traj)

def process_streaming_video(model, url, batch_size, s3_client, target_bucket):
    """
    Stream from `url` via ffmpeg, accumulate `batch_size` frames, then process each batch.
    """
    # Derive a nice name for logging
    video_name = url.split("/")[-1].split("?")[0]
    print(f"\n→ streaming {video_name}")

    # 1) Probe the stream to get its width/height
    probe = ffmpeg.probe(url)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width  = int(video_stream['width'])
    height = int(video_stream['height'])

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

    # 2) Launch ffmpeg as a subprocess, outputting rawvideo RGB24 to stdout
    # launch ffmpeg process, skipping to start_idx by frame number if resuming
    if start_idx > 0:
        process = (
            ffmpeg
            .input(url)
            .filter('select', f'gte(n,{start_idx})')
            .filter('setpts', 'PTS-STARTPTS')
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
    else:
        process = (
            ffmpeg
            .input(url)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
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
            process_batch(model, batch, batch_idx, video_name, s3_client, target_bucket, start_idx)
            batch = []
            batch_idx += 1

    # final partial batch
    if batch:
        process_batch(model, batch, batch_idx, video_name, s3_client, target_bucket, start_idx)

    process.wait()
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
