#!/usr/bin/env python3
"""
upload_s3.py

A script to upload tensor files with shape [N, 9, 540, 960] to the Tigris S3 bucket.
This script can handle both:
1. Full scene tensors (from main.py)
2. Individual frame files (from save_frames_no_plucker.py)

Usage:
    python upload_s3.py --input_dir /path/to/tensors --bucket tigris-bucket-name
"""

import os
import sys
import argparse
import time
from glob import glob
from pathlib import Path

import torch
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(
        description="Upload tensor files to Tigris S3 bucket"
    )
    p.add_argument(
        "--input_dir", required=True,
        help="Directory containing tensor files (.pt) to upload"
    )
    p.add_argument(
        "--bucket", required=True,
        help="S3 bucket name (Tigris)"
    )
    p.add_argument(
        "--prefix", default="",
        help="Optional prefix to add to S3 keys"
    )
    p.add_argument(
        "--batch_size", type=int, default=10,
        help="Number of files to upload in parallel"
    )
    return p.parse_args()

def validate_tensor(file_path):
    """
    Validate that a tensor file contains data with the expected shape [N, 9, 540, 960]
    Returns a tuple of (is_valid, tensor_shape, tensor_type)
    """
    try:
        # Load the tensor
        tensor = torch.load(file_path)
        
        # Check if it's a tensor or a dictionary
        if isinstance(tensor, dict):
            if "features" in tensor:
                tensor = tensor["features"]
            elif "images" in tensor and "poses" in tensor:
                # Legacy format, needs conversion
                return (False, str(tensor["images"].shape), "legacy")
        
        # Check the shape
        shape = tensor.shape
        if len(shape) == 4 and shape[1] == 9:  # [N, 9, H, W]
            return (True, shape, tensor.dtype)
        else:
            return (False, shape, tensor.dtype)
    
    except Exception as e:
        return (False, f"Error: {e}", None)

def list_tensor_files(input_dir):
    """
    Find all tensor files in the input directory
    """
    # Look for .pt files recursively
    all_files = []
    for ext in [".pt"]:
        pattern = os.path.join(input_dir, f"**/*{ext}")
        all_files.extend(glob(pattern, recursive=True))
    
    if not all_files:
        print(f"Error: No tensor files found in {input_dir}")
        return []
    
    return sorted(all_files)

def upload_file_to_s3(s3_client, file_path, bucket, key):
    """
    Upload a single file to S3
    """
    try:
        s3_client.upload_file(file_path, bucket, key)
        return True
    except Exception as e:
        print(f"Error uploading {file_path}: {e}")
        return False

def batch_upload_to_s3(s3_client, file_list, bucket, prefix="", batch_size=10):
    """
    Upload files in batches
    """
    import concurrent.futures
    
    total_files = len(file_list)
    uploaded = 0
    failed = 0
    
    # Create a progress bar
    with tqdm(total=total_files, desc="Uploading") as pbar:
        # Process in batches
        for i in range(0, total_files, batch_size):
            batch = file_list[i:i+batch_size]
            futures = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                for file_path in batch:
                    # Determine the S3 key
                    rel_path = os.path.basename(file_path)
                    s3_key = f"{prefix}/{rel_path}" if prefix else rel_path
                    
                    # Submit the upload task
                    future = executor.submit(upload_file_to_s3, s3_client, file_path, bucket, s3_key)
                    futures.append((future, file_path))
                
                # Process results as they complete
                for future, file_path in futures:
                    success = future.result()
                    if success:
                        uploaded += 1
                    else:
                        failed += 1
                    pbar.update(1)
    
    return uploaded, failed

def main():
    args = parse_args()
    
    try:
        import boto3
    except ImportError:
        print("Error: boto3 is required. Please install it with 'pip install boto3'")
        sys.exit(1)
    
    # Create S3 client
    s3_client = boto3.client('s3')
    
    # List tensor files
    print(f"Searching for tensor files in {args.input_dir}...")
    tensor_files = list_tensor_files(args.input_dir)
    
    if not tensor_files:
        print("No files to upload. Exiting.")
        sys.exit(1)
    
    # Validate tensors
    print(f"Found {len(tensor_files)} files. Validating...")
    valid_files = []
    invalid_files = []
    
    for file_path in tqdm(tensor_files, desc="Validating"):
        is_valid, shape, dtype = validate_tensor(file_path)
        if is_valid:
            valid_files.append(file_path)
        else:
            invalid_files.append((file_path, shape, dtype))
    
    # Report validation results
    print(f"Validation complete: {len(valid_files)} valid, {len(invalid_files)} invalid.")
    
    if invalid_files:
        print("\nInvalid files:")
        for path, shape, dtype in invalid_files:
            print(f"  {path}: Shape {shape}, Type {dtype}")
    
    if not valid_files:
        print("No valid files to upload. Exiting.")
        sys.exit(1)
    
    # Confirm upload
    print(f"\nReady to upload {len(valid_files)} files to bucket '{args.bucket}'")
    if args.prefix:
        print(f"With prefix: {args.prefix}")
    
    confirm = input("Continue with upload? (y/n): ")
    if confirm.lower() not in ['y', 'yes']:
        print("Upload cancelled.")
        sys.exit(0)
    
    # Upload files
    start_time = time.time()
    uploaded, failed = batch_upload_to_s3(
        s3_client, valid_files, args.bucket, args.prefix, args.batch_size
    )
    elapsed = time.time() - start_time
    
    # Print summary
    print("\nUpload Summary:")
    print(f"  Files uploaded: {uploaded}")
    print(f"  Files failed: {failed}")
    print(f"  Time elapsed: {elapsed:.2f} seconds")
    
    if uploaded > 0:
        print(f"\nSuccessfully uploaded to s3://{args.bucket}/{args.prefix}")

if __name__ == "__main__":
    main()