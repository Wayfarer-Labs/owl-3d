#!/usr/bin/env python3
"""
Render initial GSplat state before any optimization to check the initialization.
This script loads the initial 3D points and colors directly from MEGA-SAM data.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
import imageio

import gsplat

def load_megasam_data(output_file, max_frames=None, subsample_factor=4):
    """Load and process MEGA-SAM data directly."""
    print(f"Loading MEGA-SAM data from {output_file}")
    
    data = np.load(output_file)
    images = data['images']
    depths = data['depths'] 
    intrinsic = data['intrinsic']
    cam_c2w = data['cam_c2w']
    
    # Limit frames if specified
    if max_frames is not None:
        n_frames = min(max_frames, len(images))
        frame_step = max(1, len(images) // n_frames)
        indices = list(range(0, len(images), frame_step))[:n_frames]
        
        images = images[indices]
        depths = depths[indices]
        cam_c2w = cam_c2w[indices]
    
    print(f"Processing {len(images)} frames")
    return images, depths, intrinsic, cam_c2w

def initialize_points_for_rendering(images, depths, intrinsic, cam_c2w, subsample_factor=4, device='cuda'):
    """Initialize 3D points and colors for rendering."""
    all_pts, all_cols = [], []
    
    # Get camera intrinsics
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    print("Initializing 3D points from depth maps...")
    
    for i in range(len(images)):
        img = images[i].astype(np.float32) / 255.0
        depth = depths[i].astype(np.float32)
        cam_pose = cam_c2w[i].astype(np.float32)
        
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
        pts_cam = np.stack([x_cam, y_cam, z_valid], axis=-1)
        
        # Transform to world coordinates
        pts_hom = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
        pts_world = (cam_pose @ pts_hom.T).T[:, :3]
        
        # Get colors and ensure they're in [0, 1] range
        colors = img_sub[valid_mask]
        colors = np.clip(colors, 0, 1)  # Ensure colors are in valid range
        
        all_pts.append(pts_world)
        all_cols.append(colors)
    
    if not all_pts:
        raise ValueError("No valid points found!")
    
    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_cols, axis=0)
    
    print(f"Initialized {len(points)} 3D points")
    print(f"Color range: {colors.min():.3f} to {colors.max():.3f}")
    
    # Convert to tensors
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    colors_tensor = torch.tensor(colors, dtype=torch.float32, device=device)
    
    return points_tensor, colors_tensor, intrinsic

def create_initial_gsplat_model(points, colors, device='cuda'):
    """Create initial GSplat model parameters."""
    print("Creating initial GSplat model...")
    
    # Initialize GSplat parameters
    means = points.clone()
    scales = torch.ones_like(points) * 0.01  # Small initial scale
    quats = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=device).repeat(points.shape[0], 1)
    colors_param = colors.clone()
    opacities = torch.ones(points.shape[0], device=device) * 0.5  # Semi-transparent
    
    return {
        'means': means,
        'scales': scales,
        'quats': quats,
        'colors': colors_param,
        'opacities': opacities
    }

def generate_camera_poses(center, radius, num_views=8):
    """Generate camera poses around the scene."""
    poses = []
    
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        
        # Camera position
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2] + radius * 0.3
        
        camera_pos = np.array([x, y, z])
        
        # Look at center
        forward = center - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Create camera-to-world matrix
        cam_to_world = np.eye(4)
        cam_to_world[:3, 0] = right
        cam_to_world[:3, 1] = up
        cam_to_world[:3, 2] = -forward
        cam_to_world[:3, 3] = camera_pos
        
        poses.append(cam_to_world)
    
    return poses

def render_initial_view(model, camera_pose, intrinsic, width=800, height=600, device='cuda'):
    """Render a view from the initial GSplat model."""
    
    # Scale intrinsics to desired resolution
    original_width = intrinsic[0, 2] * 2
    original_height = intrinsic[1, 2] * 2
    
    scale_x = width / original_width
    scale_y = height / original_height
    
    fx = intrinsic[0, 0] * scale_x
    fy = intrinsic[1, 1] * scale_y
    cx = width / 2
    cy = height / 2
    
    # Convert camera pose to tensor
    world_to_cam = torch.linalg.inv(torch.tensor(camera_pose, dtype=torch.float32, device=device))
    
    # Create camera intrinsics matrix
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device)
    
    try:
        # Render using GSplat
        renders, alphas, info = gsplat.rasterization(
            model['means'],
            model['quats'],
            model['scales'],
            model['opacities'],
            model['colors'],
            viewmats=world_to_cam.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=width,
            height=height,
        )
        
        # Convert to numpy
        rendered_image = renders[0].detach().cpu().numpy()
        rendered_image = np.clip(rendered_image, 0, 1)
        
        return rendered_image
        
    except Exception as e:
        print(f"Error rendering view: {e}")
        return None

def render_initial_state(megasam_file, output_dir="initial_renders", num_views=8, max_frames=30):
    """Render the initial state before any optimization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load MEGA-SAM data
    images, depths, intrinsic, cam_c2w = load_megasam_data(megasam_file, max_frames=max_frames)
    
    # Initialize points and colors
    points, colors, intrinsic = initialize_points_for_rendering(
        images, depths, intrinsic, cam_c2w, subsample_factor=4, device=device
    )
    
    # Create initial GSplat model
    model = create_initial_gsplat_model(points, colors, device)
    
    # Estimate scene center and radius
    means_np = points.cpu().numpy()
    center = np.mean(means_np, axis=0)
    std = np.std(means_np, axis=0)
    radius = np.linalg.norm(std)
    
    print(f"Scene center: {center}")
    print(f"Scene radius: {radius:.2f}")
    
    # Generate camera poses
    poses = generate_camera_poses(center, radius, num_views)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Render views
    rendered_images = []
    for i, pose in enumerate(poses):
        print(f"Rendering initial view {i+1}/{num_views}")
        
        rendered_image = render_initial_view(model, pose, intrinsic, device=device)
        
        if rendered_image is not None:
            # Save render
            output_path = os.path.join(output_dir, f"initial_render_{i:03d}.png")
            img_uint8 = (rendered_image * 255).astype(np.uint8)
            imageio.imwrite(output_path, img_uint8)
            print(f"Saved to {output_path}")
            
            rendered_images.append(rendered_image)
        else:
            print(f"Failed to render view {i}")
    
    # Create grid visualization
    if rendered_images:
        create_grid_visualization(rendered_images, os.path.join(output_dir, "initial_grid.png"))
        print(f"Created grid: {os.path.join(output_dir, 'initial_grid.png')}")
    
    # Also render one of the original training views for comparison
    print("Rendering original training view for comparison...")
    original_pose = cam_c2w[0]  # First camera pose
    original_img = render_initial_view(model, original_pose, intrinsic, 
                                     width=images.shape[2], height=images.shape[1], device=device)
    
    if original_img is not None:
        # Save comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(images[0])
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(original_img)
        axes[1].set_title("Initial GSplat Render")
        axes[1].axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, "original_vs_initial.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison: {comparison_path}")

def create_grid_visualization(images, output_path):
    """Create a grid visualization of multiple images."""
    if not images:
        return
    
    num_images = len(images)
    grid_cols = int(np.ceil(np.sqrt(num_images)))
    grid_rows = int(np.ceil(num_images / grid_cols))
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))
    if grid_rows == 1:
        axes = [axes]
    if grid_cols == 1:
        axes = [[ax] for ax in axes]
    
    for i, img in enumerate(images):
        row = i // grid_cols
        col = i % grid_cols
        axes[row][col].imshow(img)
        axes[row][col].set_title(f"View {i}")
        axes[row][col].axis('off')
    
    # Hide empty subplots
    for i in range(len(images), grid_rows * grid_cols):
        row = i // grid_cols
        col = i % grid_cols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Render initial GSplat state before optimization")
    parser.add_argument('--megasam_file', type=str, required=True,
                       help="Path to MEGA-SAM output .npz file")
    parser.add_argument('--output_dir', type=str, default="initial_renders",
                       help="Output directory for renders")
    parser.add_argument('--num_views', type=int, default=8,
                       help="Number of views to render")
    parser.add_argument('--max_frames', type=int, default=30,
                       help="Maximum frames to process from MEGA-SAM data")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.megasam_file):
        print(f"Error: MEGA-SAM file {args.megasam_file} not found!")
        return
    
    print(f"Rendering initial state from: {args.megasam_file}")
    print(f"Output directory: {args.output_dir}")
    
    render_initial_state(args.megasam_file, args.output_dir, args.num_views, args.max_frames)
    
    print("\nInitial rendering complete!")
    print(f"Check the '{args.output_dir}' directory for output files.")

if __name__ == "__main__":
    main()
