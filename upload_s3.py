#!/usr/bin/env python3
"""
upload_s3.py

A script to upload tarball files to the Tigris S3 bucket.

Usage:
    python upload_s3.py --input_dir /path/to/tarballs --bucket tigris-bucket-name

Required Environment Variables:
    AWS_ACCESS_KEY_ID       - Your AWS/Tigris access key ID
    AWS_SECRET_ACCESS_KEY   - Your AWS/Tigris secret access key
    AWS_REGION             - AWS region (optional, defaults to us-east-1)
    AWS_ENDPOINT_URL       - Custom endpoint URL (optional, for Tigris or other S3-compatible services)

Example:
    export AWS_ACCESS_KEY_ID="your-access-key"
    export AWS_SECRET_ACCESS_KEY="your-secret-key"
    export AWS_ENDPOINT_URL="https://fly.storage.tigris.dev"
    python upload_s3.py --input_dir ./archives --bucket my-bucket
"""

import os
import sys
import argparse
import time
from glob import glob
from pathlib import Path

from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(
        description="Upload tarball files to Tigris S3 bucket"
    )
    p.add_argument(
        "--input_dir", required=True,
        help="Directory containing tarball files (.tar, .tar.gz, .tgz) to upload"
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

def validate_tarball(file_path):
    """
    Validate that a file is a valid tarball
    Returns a tuple of (is_valid, file_size, file_type)
    """
    try:
        import tarfile
        
        # Check if it's a valid tarball
        if tarfile.is_tarfile(file_path):
            size = os.path.getsize(file_path)
            # Determine the type based on extension
            ext = os.path.splitext(file_path)[1]
            if ext in ['.gz', '.bz2', '.xz']:
                file_type = f"compressed ({ext})"
            else:
                file_type = "uncompressed"
            
            return (True, f"{size / (1024 * 1024):.2f} MB", file_type)
        else:
            return (False, "Not a valid tarball", None)
    
    except Exception as e:
        return (False, f"Error: {e}", None)

def list_tarball_files(input_dir):
    """
    Find all tarball files in the input directory
    """
    # Look for common tarball extensions recursively
    all_files = []
    for ext in [".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz"]:
        pattern = os.path.join(input_dir, f"**/*{ext}")
        all_files.extend(glob(pattern, recursive=True))
    
    if not all_files:
        print(f"Error: No tarball files found in {input_dir}")
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
        from botocore.exceptions import NoCredentialsError, PartialCredentialsError
    except ImportError:
        print("Error: boto3 is required. Please install it with 'pip install boto3'")
        sys.exit(1)
    
    # Get AWS credentials from environment variables
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    aws_region = os.environ.get('AWS_REGION', 'us-east-1')  # Default to us-east-1
    aws_endpoint_url = os.environ.get('AWS_ENDPOINT_URL')  # For Tigris or other S3-compatible services
    
    # Check if credentials are provided
    if not aws_access_key_id or not aws_secret_access_key:
        print("Error: AWS credentials not found in environment variables.")
        print("Please set the following environment variables:")
        print("  - AWS_ACCESS_KEY_ID")
        print("  - AWS_SECRET_ACCESS_KEY")
        print("  - AWS_REGION (optional, defaults to us-east-1)")
        print("  - AWS_ENDPOINT_URL (optional, for Tigris or other S3-compatible services)")
        sys.exit(1)
    
    # Create S3 client with credentials
    try:
        s3_client_config = {
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
            'region_name': aws_region
        }
        
        # Add endpoint URL if provided (for Tigris or other S3-compatible services)
        if aws_endpoint_url:
            s3_client_config['endpoint_url'] = aws_endpoint_url
        
        s3_client = boto3.client('s3', **s3_client_config)
        
        # Test the connection by listing buckets (optional verification)
        print(f"Authenticating with AWS/S3 service...")
        s3_client.head_bucket(Bucket=args.bucket)
        print(f"✓ Successfully authenticated and verified access to bucket '{args.bucket}'")
        
    except NoCredentialsError:
        print("Error: AWS credentials not found or invalid.")
        sys.exit(1)
    except PartialCredentialsError:
        print("Error: Incomplete AWS credentials.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to authenticate or access bucket '{args.bucket}': {e}")
        sys.exit(1)
    
    # List tarball files
    print(f"Searching for tarball files in {args.input_dir}...")
    tarball_files = list_tarball_files(args.input_dir)
    
    if not tarball_files:
        print("No files to upload. Exiting.")
        sys.exit(1)
    
    # Validate tarballs
    print(f"Found {len(tarball_files)} files. Validating...")
    valid_files = []
    invalid_files = []
    
    for file_path in tqdm(tarball_files, desc="Validating"):
        is_valid, file_size, file_type = validate_tarball(file_path)
        if is_valid:
            valid_files.append(file_path)
        else:
            invalid_files.append((file_path, file_size, file_type))
    
    # Report validation results
    print(f"Validation complete: {len(valid_files)} valid, {len(invalid_files)} invalid.")
    
    if invalid_files:
        print("\nInvalid files:")
        for path, file_size, file_type in invalid_files:
            print(f"  {path}: {file_size}, {file_type}")
    
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