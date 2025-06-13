#!/usr/bin/env python3
"""
dl3dv_10K_dataloader.py

A PyTorch DataLoader that pulls tar files from S3, extracts preprocessed scene data,
and returns batches of 9-channel tensors (3 RGB + 6 Plucker coordinates).

Each scene is preprocessed as a tensor of shape (N, 9, H, W) where:
- N is the number of frames
- 9 channels = 3 RGB + 6 Plucker coordinates
- H, W are image dimensions

Usage:
    from dl3dv_10K_dataloader import DL3DVDataset, create_dataloader
    
    dataset = DL3DVDataset(
        bucket_name="your-tigris-bucket",
        tar_prefix="path/to/archives",
        cache_dir="./cache"
    )
    
    dataloader = create_dataloader(dataset, batch_size=4, num_workers=2)
    
    for batch in dataloader:
        # batch shape: (batch_size, N, 9, H, W)
        rgb_features = batch[:, :, :3, :, :]     # RGB channels
        plucker_features = batch[:, :, 3:, :, :] # Plucker channels
"""

import os
import sys
import tempfile
import tarfile
import shutil
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import threading
import queue

import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class S3TarCache:
    """
    Manages downloading and caching of tar files from S3.
    """
    
    def __init__(self, bucket_name: str, cache_dir: str, 
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 aws_region: str = "us-east-1",
                 aws_endpoint_url: Optional[str] = None):
        """
        Initialize S3 tar cache.
        
        Args:
            bucket_name: S3 bucket name
            cache_dir: Local directory to cache downloaded files
            aws_access_key_id: AWS access key (or from env AWS_ACCESS_KEY_ID)
            aws_secret_access_key: AWS secret key (or from env AWS_SECRET_ACCESS_KEY)
            aws_region: AWS region (or from env AWS_REGION)
            aws_endpoint_url: Custom endpoint URL (or from env AWS_ENDPOINT_URL, for Tigris)
        """
        self.bucket_name = bucket_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Get credentials from parameters or environment
        self.aws_access_key_id = aws_access_key_id or os.environ.get('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = aws_secret_access_key or os.environ.get('AWS_SECRET_ACCESS_KEY')
        self.aws_region = aws_region or os.environ.get('AWS_REGION', 'us-east-1')
        self.aws_endpoint_url = aws_endpoint_url or os.environ.get('AWS_ENDPOINT_URL')
        
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError(
                "AWS credentials not found. Please provide them as parameters or set "
                "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
            )
        
        # Initialize S3 client
        self._init_s3_client()
        
        # Thread-safe cache for extracted scenes
        self._extracted_scenes_cache = {}
        self._cache_lock = threading.Lock()
    
    def _init_s3_client(self):
        """Initialize S3 client with provided credentials."""
        s3_config = {
            'aws_access_key_id': self.aws_access_key_id,
            'aws_secret_access_key': self.aws_secret_access_key,
            'region_name': self.aws_region
        }
        
        if self.aws_endpoint_url:
            s3_config['endpoint_url'] = self.aws_endpoint_url
        
        self.s3_client = boto3.client('s3', **s3_config)
        
        # Test connection
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to S3 bucket '{self.bucket_name}': {e}")
    
    def list_tar_files(self, prefix: str = "") -> List[str]:
        """
        List all tar files in the S3 bucket with the given prefix.
        
        Args:
            prefix: S3 key prefix to filter files
            
        Returns:
            List of S3 keys for tar files
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            tar_files = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith('.tar'):
                    tar_files.append(key)
            
            return sorted(tar_files)
        
        except Exception as e:
            raise RuntimeError(f"Failed to list tar files from S3: {e}")
    
    def download_tar(self, s3_key: str) -> Path:
        """
        Download a tar file from S3 to local cache if not already present.
        
        Args:
            s3_key: S3 key of the tar file
            
        Returns:
            Path to the downloaded tar file
        """
        tar_filename = Path(s3_key).name
        local_tar_path = self.cache_dir / tar_filename
        
        # Check if already cached
        if local_tar_path.exists():
            return local_tar_path
        
        print(f"Downloading {s3_key} to {local_tar_path}...")
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_tar_path))
            return local_tar_path
        except Exception as e:
            if local_tar_path.exists():
                local_tar_path.unlink()  # Clean up partial download
            raise RuntimeError(f"Failed to download {s3_key}: {e}")
    
    def extract_scenes_from_tar(self, tar_path: Path) -> Dict[str, torch.Tensor]:
        """
        Extract all .pt files (scenes) from a tar file.
        
        Args:
            tar_path: Path to the tar file
            
        Returns:
            Dictionary mapping scene_id -> tensor data
        """
        with self._cache_lock:
            # Check cache first
            cache_key = tar_path.name
            if cache_key in self._extracted_scenes_cache:
                return self._extracted_scenes_cache[cache_key]
        
        scenes = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract tar file
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(temp_path)
            
            # Load all .pt files
            for pt_file in temp_path.glob("*.pt"):
                scene_id = pt_file.stem
                try:
                    tensor_data = torch.load(pt_file, map_location='cpu')
                    scenes[scene_id] = tensor_data
                except Exception as e:
                    print(f"Warning: Failed to load {pt_file}: {e}")
        
        # Cache the extracted scenes
        with self._cache_lock:
            self._extracted_scenes_cache[cache_key] = scenes
        
        return scenes


class DL3DVDataset(Dataset):
    """
    PyTorch Dataset for DL3DV scenes stored as tar files in S3.
    
    Each scene contains preprocessed RGB + Plucker coordinate features
    as a tensor of shape (N, 9, H, W) where N is number of frames.
    """
    
    def __init__(self, 
                 bucket_name: str,
                 tar_prefix: str = "",
                 cache_dir: str = "./cache",
                 max_frames_per_scene: Optional[int] = None,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 aws_region: str = "us-east-1",
                 aws_endpoint_url: Optional[str] = None):
        """
        Initialize the dataset.
        
        Args:
            bucket_name: S3 bucket name containing tar files
            tar_prefix: S3 key prefix for tar files (e.g., "archives/")
            cache_dir: Local directory to cache downloaded files
            max_frames_per_scene: Optional limit on frames per scene
            aws_access_key_id: AWS access key (or from env)
            aws_secret_access_key: AWS secret key (or from env)
            aws_region: AWS region (or from env)
            aws_endpoint_url: Custom endpoint URL (or from env, for Tigris)
        """
        self.max_frames_per_scene = max_frames_per_scene
        
        # Initialize S3 cache
        self.s3_cache = S3TarCache(
            bucket_name=bucket_name,
            cache_dir=cache_dir,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region=aws_region,
            aws_endpoint_url=aws_endpoint_url
        )
        
        # Discover all scenes
        self._discover_scenes(tar_prefix)
    
    def _discover_scenes(self, tar_prefix: str):
        """Discover all available scenes from tar files in S3."""
        print("Discovering scenes from S3...")
        
        # List all tar files
        tar_files = self.s3_cache.list_tar_files(tar_prefix)
        if not tar_files:
            raise ValueError(f"No tar files found with prefix '{tar_prefix}' in bucket '{self.s3_cache.bucket_name}'")
        
        print(f"Found {len(tar_files)} tar files")
        
        # Build scene index: each tar file is one scene
        self.scene_index = []
        
        for tar_file in tqdm(tar_files, desc="Indexing scenes"):
            try:
                # Download and extract scene directory name (without loading the full tensors)
                tar_path = self.s3_cache.download_tar(tar_file)
                
                # Quick peek at tar contents to get scene directory name
                with tarfile.open(tar_path, 'r') as tar:
                    members = tar.getmembers()
                    if not members:
                        continue
                    
                    # Find the scene directory (should be the first directory)
                    scene_dir = None
                    for member in members:
                        if member.isdir():
                            scene_dir = member.name
                            break
                    
                    if scene_dir:
                        scene_id = Path(scene_dir).name  # Extract just the hash part
                        self.scene_index.append((tar_file, scene_id))
                    
            except Exception as e:
                print(f"Warning: Failed to index {tar_file}: {e}")
        
        print(f"Total scenes discovered: {len(self.scene_index)}")
        
        if len(self.scene_index) == 0:
            raise ValueError("No valid scenes found in any tar files")
    
    def __len__(self) -> int:
        return len(self.scene_index)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a scene tensor by index.
        
        Args:
            idx: Scene index
            
        Returns:
            Tensor of shape (N, 9, H, W) where N <= max_frames_per_scene
        """
        import numpy as np
        from PIL import Image
        from plucker import plucker_ray
        
        tar_file, scene_id = self.scene_index[idx]
        
        # Download and extract scenes from tar
        tar_path = self.s3_cache.download_tar(tar_file)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract tar file
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(temp_path)
            
            # Find the scene directory
            scene_dir = temp_path / scene_id
            if not scene_dir.exists():
                raise RuntimeError(f"Scene directory {scene_id} not found in {tar_file}")
            
            # Find all frame files
            ext_files = sorted(scene_dir.glob("*.ext.pt"))
            int_files = sorted(scene_dir.glob("*.int.pt"))
            png_files = sorted(scene_dir.glob("*.png"))
            
            if not ext_files or not int_files or not png_files:
                raise RuntimeError(f"Missing frame files in scene {scene_id}")
            
            # Limit frames if requested
            if self.max_frames_per_scene is not None:
                ext_files = ext_files[:self.max_frames_per_scene]
                int_files = int_files[:self.max_frames_per_scene]
                png_files = png_files[:self.max_frames_per_scene]
            
            # Load all extrinsics and intrinsics for this scene
            extrinsics_list = []
            intrinsics_list = []
            images_list = []
            
            for ext_file, int_file, png_file in zip(ext_files, int_files, png_files):
                # Load camera parameters
                extrinsics = torch.load(ext_file, map_location='cpu')  # 4x4 C2W matrix
                intrinsics = torch.load(int_file, map_location='cpu')  # [fx, fy, cx, cy]
                
                # Load image
                image = Image.open(png_file).convert('RGB')
                image_np = np.array(image)
                image_tensor = torch.from_numpy(image_np).float() / 255.0  # Normalize to [0,1]
                image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
                
                extrinsics_list.append(extrinsics)
                intrinsics_list.append(intrinsics)
                images_list.append(image_tensor)
            
            # Stack camera parameters for batch processing
            # Add batch and view dimensions: (1, N, 4, 4) and (1, N, 4)
            C2W = torch.stack(extrinsics_list, dim=0).unsqueeze(0)  # (1, N, 4, 4)
            fxfycxcy = torch.stack(intrinsics_list, dim=0).unsqueeze(0)  # (1, N, 4)
            
            # Get image dimensions from first image
            H, W = images_list[0].shape[1], images_list[0].shape[2]
            
            # Compute Plucker coordinates for all frames at once
            plucker_coords, (ray_o, ray_d) = plucker_ray(H, W, C2W, fxfycxcy)
            
            # Remove batch dimension: (1, N, 6, H, W) -> (N, 6, H, W)
            plucker_coords = plucker_coords.squeeze(0)
            
            # Stack all frames and combine RGB + Plucker
            frame_tensors = []
            for i, image_tensor in enumerate(images_list):
                # Combine RGB (3) + Plucker (6) = 9 channels
                frame_tensor = torch.cat([image_tensor, plucker_coords[i]], dim=0)
                frame_tensors.append(frame_tensor)
            
            # Stack all frames
            scene_tensor = torch.stack(frame_tensors, dim=0)  # Shape: (N, 9, H, W)
        
        return scene_tensor
    
    def get_scene_info(self, idx: int) -> Dict[str, Any]:
        """
        Get metadata about a scene without loading the full tensor.
        
        Args:
            idx: Scene index
            
        Returns:
            Dictionary with scene metadata
        """
        tar_file, scene_id = self.scene_index[idx]
        return {
            'scene_id': scene_id,
            'tar_file': tar_file,
            'index': idx
        }


def collate_scenes(batch: List[torch.Tensor]) -> torch.Tensor:
    """
    Custom collate function to handle variable-length sequences.
    
    Pads scenes to the same number of frames within a batch.
    
    Args:
        batch: List of scene tensors, each of shape (N_i, 9, H, W)
        
    Returns:
        Batched tensor of shape (batch_size, max_N, 9, H, W)
    """
    if not batch:
        raise ValueError("Empty batch")
    
    # Find max frames in this batch
    max_frames = max(scene.shape[0] for scene in batch)
    batch_size = len(batch)
    
    # Get tensor dimensions from first scene
    _, channels, height, width = batch[0].shape
    
    # Create padded batch tensor
    batched = torch.zeros(batch_size, max_frames, channels, height, width, 
                         dtype=batch[0].dtype)
    
    for i, scene in enumerate(batch):
        num_frames = scene.shape[0]
        batched[i, :num_frames] = scene
    
    return batched


def create_dataloader(dataset: DL3DVDataset, 
                     batch_size: int = 1,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     **kwargs) -> DataLoader:
    """
    Create a DataLoader for the DL3DV dataset.
    
    Args:
        dataset: DL3DVDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_scenes,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DL3DV DataLoader")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="", help="S3 prefix for tar files")
    parser.add_argument("--cache_dir", default="./cache", help="Local cache directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--max_frames", type=int, help="Max frames per scene")
    parser.add_argument("--num_batches", type=int, default=3, help="Number of batches to test")
    
    args = parser.parse_args()
    
    # Create dataset
    dataset = DL3DVDataset(
        bucket_name=args.bucket,
        tar_prefix=args.prefix,
        cache_dir=args.cache_dir,
        max_frames_per_scene=args.max_frames
    )
    
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
    for i, batch in enumerate(dataloader):
        if i >= args.num_batches:
            break
            
        print(f"Batch {i+1}:")
        print(f"  Shape: {batch.shape}")
        print(f"  Dtype: {batch.dtype}")
        print(f"  RGB channels range: [{batch[:,:,:3].min():.3f}, {batch[:,:,:3].max():.3f}]")
        print(f"  Plucker channels range: [{batch[:,:,3:].min():.3f}, {batch[:,:,3:].max():.3f}]")
        print()
    
    print("DataLoader test completed successfully!")