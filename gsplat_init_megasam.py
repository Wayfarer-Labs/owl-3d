#!/usr/bin/env python3
"""
Initialize GSplat from MEGA-SAM outputs with optional TSDF filtering
This script loads depth maps and camera parameters from MEGA-SAM outputs
and initializes a GSplat model, optionally using TSDF fusion for filtering dynamic elements.

Usage:
    uv run gsplat_init_megasam.py --outputs_dir ./output/path
    uv run gsplat_init_megasam.py --outputs_dir ./output/path --use_tsdf_filtering
    uv run gsplat_init_megasam.py --outputs_dir ./output/path --use_tsdf_filtering --max_frames 50 --epochs 200
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

import gsplat

# Import TSDF filtering from the visualization script
try:
    from tsdf_fusion import TSDFFusion
    from create_3d_visualization_filtered import (
        load_data_for_tsdf, 
        process_with_tsdf_fusion,
        estimate_scene_bounds
    )
    TSDF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TSDF filtering not available: {e}")
    TSDF_AVAILABLE = False

# ---------------------------------------
# 1. MEGA-SAM Dataset Definition
# ---------------------------------------
class MegaSamDataset(Dataset):
    """Dataset for MEGA-SAM outputs (.npz files)"""
    
    def __init__(self, output_file, max_frames=None, subsample_points=True, subsample_factor=4):
        """
        Initialize dataset from MEGA-SAM output file.
        
        Args:
            output_file: Path to .npz file with MEGA-SAM outputs
            max_frames: Maximum number of frames to use (None for all)
            subsample_points: Whether to subsample points for efficiency
            subsample_factor: Factor to subsample depth maps (higher = less points)
        """
        self.output_file = output_file
        self.subsample_points = subsample_points
        self.subsample_factor = subsample_factor
        
        # Load data
        data = np.load(output_file)
        self.images = data['images']
        self.depths = data['depths']
        self.intrinsic = data['intrinsic']
        self.cam_c2w = data['cam_c2w']  # Camera to world transforms
        
        # Limit frames if specified
        if max_frames is not None:
            n_frames = min(max_frames, len(self.images))
            frame_step = max(1, len(self.images) // n_frames)
            indices = list(range(0, len(self.images), frame_step))[:n_frames]
            
            self.images = self.images[indices]
            self.depths = self.depths[indices]
            self.cam_c2w = self.cam_c2w[indices]
        
        print(f"Loaded {len(self.images)} frames from {output_file}")
        print(f"Image shape: {self.images[0].shape}")
        print(f"Depth shape: {self.depths[0].shape}")
        print(f"Intrinsics: {self.intrinsic}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Return image, depth, intrinsics, and camera pose for a frame."""
        img = self.images[idx].astype(np.float32) / 255.0
        depth = self.depths[idx].astype(np.float32)
        intrinsic = self.intrinsic.astype(np.float32)
        cam_pose = self.cam_c2w[idx].astype(np.float32)  # Camera to world
        
        return img, depth, intrinsic, cam_pose

# ---------------------------------------
# 2. Point Cloud Initialization
# ---------------------------------------
def initialize_points_from_dataset(dataset, device='cuda', subsample_factor=4):
    """
    Initialize 3D points and colors from depth maps in the dataset.
    
    Args:
        dataset: MegaSamDataset instance
        device: Device to use
        subsample_factor: Factor to subsample depth maps
    
    Returns:
        points: 3D points tensor [N, 3]
        colors: RGB colors tensor [N, 3]
    """
    all_pts, all_cols = [], []
    
    # Get camera intrinsics
    fx, fy = dataset.intrinsic[0, 0], dataset.intrinsic[1, 1]
    cx, cy = dataset.intrinsic[0, 2], dataset.intrinsic[1, 2]
    
    print("Initializing 3D points from depth maps...")
    
    for i in tqdm(range(len(dataset)), desc="Processing frames"):
        img, depth, intrinsic, cam_pose = dataset[i]
        
        # Subsample for efficiency
        if subsample_factor > 1:
            depth_sub = depth[::subsample_factor, ::subsample_factor]
            img_sub = img[::subsample_factor, ::subsample_factor]
        else:
            depth_sub = depth
            img_sub = img
        
        # Find valid depth pixels
        valid_mask = depth_sub > 0
        if not valid_mask.any():
            continue
            
        # Get pixel coordinates
        H, W = depth_sub.shape
        v_coords, u_coords = np.meshgrid(
            np.arange(0, H) * subsample_factor,
            np.arange(0, W) * subsample_factor,
            indexing='ij'
        )
        
        u_valid = u_coords[valid_mask]
        v_valid = v_coords[valid_mask]
        z_valid = depth_sub[valid_mask]
        
        # Unproject to camera coordinates
        x_cam = (u_valid - cx) * z_valid / fx
        y_cam = (v_valid - cy) * z_valid / fy
        pts_cam = np.stack([x_cam, y_cam, z_valid], axis=-1)  # [N, 3]
        
        # Transform to world coordinates
        # cam_pose is camera-to-world transform
        pts_hom = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
        pts_world = (cam_pose @ pts_hom.T).T[:, :3]
        
        # Get colors
        colors = img_sub[valid_mask]
        
        all_pts.append(pts_world)
        all_cols.append(colors)
    
    # Concatenate all points
    if not all_pts:
        raise ValueError("No valid points found in dataset!")
    
    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_cols, axis=0)
    
    print(f"Initialized {len(points)} 3D points")
    
    # Convert to tensors
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    colors_tensor = torch.tensor(colors, dtype=torch.float32, device=device)
    
    return points_tensor, colors_tensor

def initialize_points_with_tsdf_filtering(output_file, device='cuda', 
                                        voxel_size=0.03, max_frames=50,
                                        min_weight_threshold=3.0):
    """
    Initialize 3D points using TSDF filtering to remove dynamic elements.
    
    Args:
        output_file: Path to MEGA-SAM output file
        device: Device to use
        voxel_size: TSDF voxel size
        max_frames: Maximum frames to process
        min_weight_threshold: Minimum weight for point extraction
    
    Returns:
        points: Filtered 3D points tensor [N, 3]
        colors: Corresponding colors tensor [N, 3]
    """
    if not TSDF_AVAILABLE:
        raise ImportError("TSDF filtering requires tsdf_fusion module")
    
    print("Initializing points with TSDF filtering...")
    
    # Load data for TSDF processing
    images, depths, intrinsic, cam_c2w = load_data_for_tsdf(output_file)
    
    # Process with TSDF fusion
    filtered_points, filtered_colors, tsdf_info = process_with_tsdf_fusion(
        images, depths, intrinsic, cam_c2w,
        voxel_size=voxel_size,
        max_frames=max_frames,
        min_weight_threshold=min_weight_threshold,
        device=device
    )
    
    if len(filtered_points) == 0:
        raise ValueError("No points extracted after TSDF filtering! Try lowering min_weight_threshold")
    
    print(f"TSDF filtering extracted {len(filtered_points)} points")
    print(f"TSDF info: {tsdf_info}")
    
    # Convert to tensors
    points_tensor = torch.tensor(filtered_points, dtype=torch.float32, device=device)
    colors_tensor = torch.tensor(filtered_colors, dtype=torch.float32, device=device)
    
    return points_tensor, colors_tensor

# ---------------------------------------
# 3. GSplat Training
# ---------------------------------------
def train_gsplat(dataset, points, colors, device='cuda', epochs=100, lr=1e-2,
                init_scale=0.01, init_opacity=0.5, save_model=True, model_path="gsplat_model.pth"):
    """
    Train GSplat model with initialized points and colors.
    
    Args:
        dataset: MegaSamDataset instance
        points: Initial 3D points [N, 3]
        colors: Initial colors [N, 3]
        device: Device to use
        epochs: Number of training epochs
        lr: Learning rate
        init_scale: Initial scale for Gaussians
        init_opacity: Initial opacity for Gaussians
        save_model: Whether to save the trained model
        model_path: Path to save the model
    """
    print(f"Training GSplat with {len(points)} points for {epochs} epochs...")
    
    # Initialize GSplat parameters
    means = torch.nn.Parameter(points.clone())
    scales = torch.nn.Parameter(torch.ones_like(points) * init_scale)
    quats = torch.nn.Parameter(
        torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=device).repeat(points.shape[0], 1)
    )
    colors_param = torch.nn.Parameter(colors.clone())
    opacities = torch.nn.Parameter(torch.ones(points.shape[0], device=device) * init_opacity)
    
    # Optimizer
    params = [means, scales, quats, colors_param, opacities]
    optimizer = torch.optim.Adam(params, lr=lr)
    
    # Get camera intrinsics
    fx, fy = dataset.intrinsic[0, 0], dataset.intrinsic[1, 1]
    cx, cy = dataset.intrinsic[0, 2], dataset.intrinsic[1, 2]
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for i in range(len(dataset)):
            # Get frame data
            img, depth, intrinsic, cam_pose = dataset[i]
            
            # Convert to tensors
            img_tensor = torch.tensor(img, dtype=torch.float32, device=device)
            cam_pose_tensor = torch.tensor(cam_pose, dtype=torch.float32, device=device)
            
            # Convert camera-to-world to world-to-camera for GSplat
            world_to_cam = torch.linalg.inv(cam_pose_tensor)
            
            optimizer.zero_grad()
            
            # Project Gaussians
            try:
                # Use the new GSplat v1.5+ API
                renders, alphas, info = gsplat.rasterization(
                    means,
                    quats,
                    scales,
                    opacities,
                    colors_param,
                    viewmats=world_to_cam.unsqueeze(0),  # Add batch dimension
                    Ks=torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0),  # Add batch dimension
                    width=img.shape[1],
                    height=img.shape[0],
                )
                
                # Extract the rendered image (first element of batch)
                rendered = renders[0]
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(rendered, img_tensor)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error in frame {i}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[Epoch {epoch+1}/{epochs}] Average Loss: {avg_loss:.6f}")
    
    # Save model
    if save_model:
        model_state = {
            'means': means.detach().cpu(),
            'scales': scales.detach().cpu(),
            'quats': quats.detach().cpu(),
            'colors': colors_param.detach().cpu(),
            'opacities': opacities.detach().cpu(),
            'intrinsics': dataset.intrinsic,
            'num_points': len(points),
            'epochs_trained': epochs
        }
        torch.save(model_state, model_path)
        print(f"Model saved to {model_path}")
    
    return {
        'means': means,
        'scales': scales,
        'quats': quats,
        'colors': colors_param,
        'opacities': opacities
    }

# ---------------------------------------
# 4. Argument Parsing and Main
# ---------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Initialize GSplat from MEGA-SAM outputs")
    parser.add_argument('--outputs_dir', type=str, required=True, 
                       help="Path to MEGA-SAM outputs directory")
    parser.add_argument('--use_tsdf_filtering', action='store_true',
                       help="Use TSDF filtering for initialization (removes dynamic elements)")
    parser.add_argument('--max_frames', type=int, default=50,
                       help="Maximum frames to process (default: 50)")
    parser.add_argument('--epochs', type=int, default=100,
                       help="Number of training epochs (default: 100)")
    parser.add_argument('--lr', type=float, default=1e-2,
                       help="Learning rate (default: 1e-2)")
    parser.add_argument('--init_scale', type=float, default=0.01,
                       help="Initial scale for Gaussians (default: 0.01)")
    parser.add_argument('--init_opacity', type=float, default=0.5,
                       help="Initial opacity for Gaussians (default: 0.5)")
    parser.add_argument('--voxel_size', type=float, default=0.03,
                       help="TSDF voxel size (default: 0.03)")
    parser.add_argument('--weight_threshold', type=float, default=3.0,
                       help="TSDF minimum weight threshold (default: 3.0)")
    parser.add_argument('--subsample_factor', type=int, default=4,
                       help="Depth subsampling factor for efficiency (default: 4)")
    parser.add_argument('--model_path', type=str, default="gsplat_megasam_model.pth",
                       help="Path to save trained model (default: gsplat_megasam_model.pth)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find MEGA-SAM output file
    output_files = []
    outputs_dir = args.outputs_dir
    
    if outputs_dir:
        # Look for CoD-sequence file first
        cod_output_file = os.path.join(outputs_dir, "CoD-sequence_droid.npz")
        if os.path.exists(cod_output_file):
            output_files.append(cod_output_file)
        else:
            # Fallback: look for files containing "CoD"
            if os.path.exists(outputs_dir):
                for f in os.listdir(outputs_dir):
                    if "CoD" in f and f.endswith("_droid.npz"):
                        output_files.append(os.path.join(outputs_dir, f))
                
                # If no CoD files, use any available
                if not output_files:
                    for f in os.listdir(outputs_dir):
                        if f.endswith("_droid.npz"):
                            output_files.append(os.path.join(outputs_dir, f))
    
    if not output_files:
        print("No MEGA-SAM output files found!")
        print(f"Please check that {outputs_dir} contains .npz files from MEGA-SAM")
        return
    
    output_file = output_files[0]
    print(f"Processing: {output_file}")
    
    # Initialize points
    if args.use_tsdf_filtering:
        if not TSDF_AVAILABLE:
            print("Error: TSDF filtering requested but not available!")
            print("Make sure tsdf_fusion.py and create_3d_visualization_filtered.py are in the workspace")
            return
        
        print("\n" + "="*50)
        print("INITIALIZING WITH TSDF FILTERING")
        print("="*50)
        
        points, colors = initialize_points_with_tsdf_filtering(
            output_file, device=device,
            voxel_size=args.voxel_size,
            max_frames=args.max_frames,
            min_weight_threshold=args.weight_threshold
        )
        
        # Create dataset for training (without subsampling since TSDF already filtered)
        dataset = MegaSamDataset(output_file, max_frames=args.max_frames, 
                               subsample_points=False, subsample_factor=1)
    else:
        print("\n" + "="*50)
        print("INITIALIZING FROM DEPTH MAPS")
        print("="*50)
        
        # Create dataset
        dataset = MegaSamDataset(output_file, max_frames=args.max_frames,
                               subsample_points=True, subsample_factor=args.subsample_factor)
        
        # Initialize points from depth maps
        points, colors = initialize_points_from_dataset(
            dataset, device=device, subsample_factor=args.subsample_factor
        )
    
    print(f"\nInitialized {len(points)} points with colors")
    
    # Train GSplat
    print("\n" + "="*50)
    print("TRAINING GSPLAT")
    print("="*50)
    
    model_params = train_gsplat(
        dataset, points, colors, device=device,
        epochs=args.epochs, lr=args.lr,
        init_scale=args.init_scale, init_opacity=args.init_opacity,
        save_model=True, model_path=args.model_path
    )
    
    print(f"\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Model saved to: {args.model_path}")
    print(f"Final model has {len(model_params['means'])} Gaussians")
    
    if args.use_tsdf_filtering:
        print("Model was initialized with TSDF-filtered points (dynamic elements removed)")
    else:
        print("Model was initialized from raw depth maps")

if __name__ == "__main__":
    main()
