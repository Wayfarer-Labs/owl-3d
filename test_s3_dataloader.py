#!/usr/bin/env python3
"""
test_s3_dataloader.py

Test the DL3DV dataloader with S3 tar files.
"""

import os
import sys
import argparse

# Add the current directory to Python path to import the dataloader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dl3dv_10K_dataloader import DL3DVDataset, create_dataloader


def test_s3_dataloader():
    """Test the S3 dataloader."""
    
    parser = argparse.ArgumentParser(description="Test DL3DV S3 DataLoader")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="", help="S3 prefix for tar files")
    parser.add_argument("--cache_dir", default="./cache", help="Local cache directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--max_frames", type=int, default=5, help="Max frames per scene (for testing)")
    parser.add_argument("--num_batches", type=int, default=2, help="Number of batches to test")
    
    args = parser.parse_args()
    
    print("Testing DL3DV S3 DataLoader")
    print("=" * 40)
    
    # Check for AWS credentials
    if not os.environ.get('AWS_ACCESS_KEY_ID') or not os.environ.get('AWS_SECRET_ACCESS_KEY'):
        print("Error: AWS credentials not found in environment variables.")
        print("Please set the following environment variables:")
        print("  - AWS_ACCESS_KEY_ID")
        print("  - AWS_SECRET_ACCESS_KEY")
        print("  - AWS_REGION (optional, defaults to us-east-1)")
        print("  - AWS_ENDPOINT_URL (optional, for Tigris or other S3-compatible services)")
        return
    
    # Create dataset
    try:
        dataset = DL3DVDataset(
            bucket_name=args.bucket,
            tar_prefix=args.prefix,
            cache_dir=args.cache_dir,
            max_frames_per_scene=args.max_frames
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return
    
    print(f"Dataset size: {len(dataset)} scenes")
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Use 0 for debugging
    )
    
    # Test a few batches
    print("\nTesting dataloader...")
    try:
        for i, batch in enumerate(dataloader):
            if i >= args.num_batches:
                break
                
            print(f"Batch {i+1}:")
            print(f"  Shape: {batch.shape}")
            print(f"  Dtype: {batch.dtype}")
            print(f"  Memory usage: {batch.numel() * batch.element_size() / (1024**2):.2f} MB")
            
            if batch.shape[2] >= 9:  # Check if we have the expected 9 channels
                rgb_channels = batch[:, :, :3, :, :]
                plucker_channels = batch[:, :, 3:9, :, :]
                print(f"  RGB channels range: [{rgb_channels.min():.3f}, {rgb_channels.max():.3f}]")
                print(f"  Plucker channels range: [{plucker_channels.min():.3f}, {plucker_channels.max():.3f}]")
            else:
                print(f"  Warning: Expected 9 channels, got {batch.shape[2]}")
            print()
        
        print("S3 DataLoader test completed successfully!")
        
        # Test individual scene access
        print("\nTesting individual scene access...")
        scene_info = dataset.get_scene_info(0)
        print(f"First scene info: {scene_info}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_s3_dataloader()
