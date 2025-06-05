#!/usr/bin/env python3
"""
test_local_dataloader.py

Test the DL3DV dataloader with local tar files (for development/testing).
This script uses the existing local tar files in the archives/ directory.
"""

import os
import sys
import tempfile
import tarfile
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader

# Add the current directory to Python path to import the dataloader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dl3dv_10K_dataloader import collate_scenes


class LocalDL3DVDataset(Dataset):
    """
    Local version of DL3DVDataset for testing with local tar files.
    """
    
    def __init__(self, archives_dir: str, max_frames_per_scene: int = None):
        """
        Initialize the local dataset.
        
        Args:
            archives_dir: Directory containing local tar files
            max_frames_per_scene: Optional limit on frames per scene
        """
        self.archives_dir = Path(archives_dir)
        self.max_frames_per_scene = max_frames_per_scene
        
        if not self.archives_dir.exists():
            raise ValueError(f"Archives directory {archives_dir} does not exist")
        
        # Discover scenes from local tar files
        self._discover_local_scenes()
    
    def _discover_local_scenes(self):
        """Discover all available scenes from local tar files."""
        print(f"Discovering scenes from local directory: {self.archives_dir}")
        
        # Find all tar files
        tar_files = list(self.archives_dir.glob("*.tar"))
        if not tar_files:
            raise ValueError(f"No tar files found in {self.archives_dir}")
        
        print(f"Found {len(tar_files)} tar files")
        
        # Build scene index - each tar file is one scene
        self.scene_index = []
        
        for tar_file in sorted(tar_files):
            try:
                # Get the scene directory name from tar contents
                with tarfile.open(tar_file, 'r') as tar:
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
        
        # Extract scene from tar file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract tar file
            with tarfile.open(tar_file, 'r') as tar:
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
    
    def get_scene_info(self, idx: int) -> Dict[str, any]:
        """Get metadata about a scene."""
        tar_file, scene_id = self.scene_index[idx]
        return {
            'scene_id': scene_id,
            'tar_file': str(tar_file),
            'index': idx
        }


def test_local_dataloader():
    """Test the dataloader with local tar files."""
    
    # Path to local archives
    archives_dir = "./archives"
    
    if not os.path.exists(archives_dir):
        print(f"Error: Archives directory '{archives_dir}' not found.")
        print("This test requires local tar files in the ./archives/ directory.")
        return
    
    print("Testing Local DL3DV DataLoader")
    print("=" * 40)
    
    # Create dataset
    try:
        dataset = LocalDL3DVDataset(
            archives_dir=archives_dir,
            max_frames_per_scene=100  # Limit to 10 frames for testing
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return
    
    print(f"Dataset size: {len(dataset)} scenes")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=5,
        shuffle=True,
        num_workers=0,  # Use 0 for debugging
        collate_fn=collate_scenes
    )
    
    # Test a few batches
    print("\nTesting dataloader...")
    try:
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Test only 3 batches
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
        
        print("Local DataLoader test completed successfully!")
        
        # Test individual scene access
        print("\nTesting individual scene access...")
        scene_info = dataset.get_scene_info(0)
        print(f"First scene info: {scene_info}")
        
        first_scene = dataset[0]
        print(f"First scene shape: {first_scene.shape}")
        print(f"First scene dtype: {first_scene.dtype}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_local_dataloader()
