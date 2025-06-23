#!/usr/bin/env python3
"""
Visualize GSplat models trained from MEGA-SAM outputs
This script loads trained GSplat models and renders views from different camera positions.

Usage:
    uv run gsplat_visualizer.py --model_path gsplat_megasam_model.pth
    uv run gsplat_visualizer.py --model_path gsplat_megasam_model_tsdf.pth --output_dir renders_tsdf
    uv run gsplat_visualizer.py --model_path gsplat_megasam_model.pth --render_video
"""
import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import imageio

import gsplat

def load_gsplat_model(model_path, device='cuda'):
    """Load a trained GSplat model from file."""
    print(f"Loading GSplat model from {model_path}")
    
    # Load with weights_only=False to handle numpy arrays in the saved model
    model_state = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract model parameters
    means = model_state['means'].to(device)
    raw_scales = model_state['scales'].to(device)
    raw_quats = model_state['quats'].to(device)
    raw_colors = model_state['colors'].to(device)
    raw_opacities = model_state['opacities'].to(device)
    intrinsics = model_state['intrinsics']
    
    # Apply necessary activations to trained parameters
    scales = torch.exp(raw_scales)  # Scales are stored in log space
    colors = torch.clamp(raw_colors, 0, 1)  # Clamp colors to [0,1] range
    opacities = torch.sigmoid(raw_opacities)  # Apply sigmoid to opacities
    quats = torch.nn.functional.normalize(raw_quats, p=2, dim=-1)  # Normalize quaternions
    
    print(f"Loaded model with {len(means)} Gaussians")
    print(f"Trained for {model_state.get('epochs_trained', 'unknown')} epochs")
    print(f"Applied activations: scales (exp), colors (clamp 0-1), opacities (sigmoid), quats (normalize)")
    
    return {
        'means': means,
        'scales': scales,
        'quats': quats,
        'colors': colors,
        'opacities': opacities,
        'intrinsics': intrinsics
    }

def generate_camera_poses(center=np.array([0, 0, 0]), radius=3.0, num_views=8, height_offset=0.5):
    """Generate camera poses in a circle around the scene."""
    poses = []
    
    for i in range(num_views):
        # Angle around the circle
        angle = 2 * np.pi * i / num_views
        
        # Camera position
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2] + height_offset
        
        camera_pos = np.array([x, y, z])
        
        # Look at center
        forward = center - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        # Up vector (slightly tilted for better view)
        up = np.array([0, 0, 1])
        
        # Right vector
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        # Recompute up to be orthogonal
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Create camera-to-world matrix
        cam_to_world = np.eye(4)
        cam_to_world[:3, 0] = right
        cam_to_world[:3, 1] = up
        cam_to_world[:3, 2] = -forward  # Camera looks in -Z direction
        cam_to_world[:3, 3] = camera_pos
        
        poses.append(cam_to_world)
    
    return poses

def estimate_scene_center_and_radius(means):
    """Estimate scene center and radius from Gaussian means."""
    means_np = means.cpu().numpy()
    
    # Compute bounding box
    min_coords = np.min(means_np, axis=0)
    max_coords = np.max(means_np, axis=0)
    
    center = (min_coords + max_coords) / 2
    radius = np.linalg.norm(max_coords - min_coords) / 2
    
    print(f"Estimated scene center: {center}")
    print(f"Estimated scene radius: {radius:.2f}")
    
    return center, radius

def render_view(model, camera_pose, width=800, height=600, device='cuda'):
    """Render a view from the GSplat model given a camera pose."""
    
    # Get camera intrinsics
    intrinsics = model['intrinsics']
    
    # Scale intrinsics to desired resolution
    original_width = intrinsics[0, 2] * 2  # cx * 2
    original_height = intrinsics[1, 2] * 2  # cy * 2
    
    scale_x = width / original_width
    scale_y = height / original_height
    
    fx = intrinsics[0, 0] * scale_x
    fy = intrinsics[1, 1] * scale_y
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
            viewmats=world_to_cam.unsqueeze(0),  # Add batch dimension
            Ks=K.unsqueeze(0),  # Add batch dimension
            width=width,
            height=height,
        )
        
        # Convert to numpy
        rendered_image = renders[0].detach().cpu().numpy()
        alpha_map = alphas[0].detach().cpu().numpy()
        
        # Clip values to [0, 1]
        rendered_image = np.clip(rendered_image, 0, 1)
        
        return rendered_image, alpha_map
        
    except Exception as e:
        print(f"Error rendering view: {e}")
        return None, None

def create_renders(model_path, output_dir="renders", num_views=8, resolution=(800, 600)):
    """Create multiple renders from different viewpoints."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_gsplat_model(model_path, device)
    
    # Estimate scene parameters
    center, radius = estimate_scene_center_and_radius(model['means'])
    
    # Generate camera poses
    poses = generate_camera_poses(center, radius * 1.5, num_views, height_offset=radius * 0.3)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Render views
    rendered_images = []
    for i, pose in enumerate(poses):
        print(f"Rendering view {i+1}/{num_views}")
        
        rendered_image, alpha_map = render_view(model, pose, resolution[0], resolution[1], device)
        
        if rendered_image is not None:
            # Save individual render
            output_path = os.path.join(output_dir, f"render_{i:03d}.png")
            
            # Convert to uint8
            img_uint8 = (rendered_image * 255).astype(np.uint8)
            
            # Save image
            imageio.imwrite(output_path, img_uint8)
            print(f"Saved render to {output_path}")
            
            rendered_images.append(rendered_image)
        else:
            print(f"Failed to render view {i}")
    
    # Create a grid visualization
    if rendered_images:
        create_grid_visualization(rendered_images, os.path.join(output_dir, "grid_view.png"))
        print(f"Created grid visualization: {os.path.join(output_dir, 'grid_view.png')}")
    
    return rendered_images

def create_grid_visualization(images, output_path, grid_size=None):
    """Create a grid visualization of multiple images."""
    if not images:
        return
    
    num_images = len(images)
    
    if grid_size is None:
        # Determine grid size
        grid_cols = int(np.ceil(np.sqrt(num_images)))
        grid_rows = int(np.ceil(num_images / grid_cols))
    else:
        grid_rows, grid_cols = grid_size
    
    # Get image dimensions
    img_height, img_width = images[0].shape[:2]
    
    # Create grid
    grid_height = grid_rows * img_height
    grid_width = grid_cols * img_width
    
    if len(images[0].shape) == 3:
        grid_image = np.zeros((grid_height, grid_width, images[0].shape[2]))
    else:
        grid_image = np.zeros((grid_height, grid_width))
    
    for i, img in enumerate(images):
        row = i // grid_cols
        col = i % grid_cols
        
        y_start = row * img_height
        y_end = y_start + img_height
        x_start = col * img_width
        x_end = x_start + img_width
        
        grid_image[y_start:y_end, x_start:x_end] = img
    
    # Save grid
    grid_uint8 = (np.clip(grid_image, 0, 1) * 255).astype(np.uint8)
    imageio.imwrite(output_path, grid_uint8)

def create_video(model_path, output_path="gsplat_video.mp4", num_frames=60, resolution=(800, 600), fps=30):
    """Create a rotating video of the GSplat model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_gsplat_model(model_path, device)
    
    # Estimate scene parameters
    center, radius = estimate_scene_center_and_radius(model['means'])
    
    # Generate camera poses for smooth rotation
    poses = generate_camera_poses(center, radius * 1.5, num_frames, height_offset=radius * 0.3)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    
    print(f"Creating video with {num_frames} frames at {fps} FPS")
    
    for i, pose in enumerate(poses):
        print(f"Rendering frame {i+1}/{num_frames}")
        
        rendered_image, _ = render_view(model, pose, resolution[0], resolution[1], device)
        
        if rendered_image is not None:
            # Convert to BGR for OpenCV
            img_bgr = cv2.cvtColor((rendered_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            video_writer.write(img_bgr)
        else:
            print(f"Failed to render frame {i}")
    
    video_writer.release()
    print(f"Video saved to {output_path}")

def compare_models(model_path1, model_path2, output_dir="comparison", num_views=4):
    """Compare renders from two different models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load both models
    model1 = load_gsplat_model(model_path1, device)
    model2 = load_gsplat_model(model_path2, device)
    
    # Use the first model to estimate scene parameters
    center, radius = estimate_scene_center_and_radius(model1['means'])
    
    # Generate camera poses
    poses = generate_camera_poses(center, radius * 1.5, num_views, height_offset=radius * 0.3)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Render comparison views
    for i, pose in enumerate(poses):
        print(f"Rendering comparison view {i+1}/{num_views}")
        
        # Render from both models
        img1, _ = render_view(model1, pose, 400, 300, device)
        img2, _ = render_view(model2, pose, 400, 300, device)
        
        if img1 is not None and img2 is not None:
            # Create side-by-side comparison
            comparison = np.hstack([img1, img2])
            
            # Add labels
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            ax.imshow(comparison)
            ax.set_title(f"View {i+1}: Left = {Path(model_path1).stem}, Right = {Path(model_path2).stem}")
            ax.axis('off')
            
            # Save comparison
            comparison_path = os.path.join(output_dir, f"comparison_view_{i:03d}.png")
            plt.savefig(comparison_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"Saved comparison to {comparison_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize GSplat models from MEGA-SAM outputs")
    parser.add_argument('--model_path', type=str, required=True,
                       help="Path to the trained GSplat model (.pth file)")
    parser.add_argument('--output_dir', type=str, default="renders",
                       help="Output directory for renders (default: renders)")
    parser.add_argument('--num_views', type=int, default=8,
                       help="Number of views to render (default: 8)")
    parser.add_argument('--resolution', type=int, nargs=2, default=[800, 600],
                       help="Render resolution [width height] (default: 800 600)")
    parser.add_argument('--render_video', action='store_true',
                       help="Create a rotating video")
    parser.add_argument('--video_frames', type=int, default=60,
                       help="Number of frames for video (default: 60)")
    parser.add_argument('--fps', type=int, default=30,
                       help="Video FPS (default: 30)")
    parser.add_argument('--compare_with', type=str, default=None,
                       help="Path to second model for comparison")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found!")
        return
    
    print(f"Visualizing GSplat model: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    
    if args.compare_with:
        print(f"Comparing with: {args.compare_with}")
        compare_models(args.model_path, args.compare_with, args.output_dir, args.num_views)
    
    elif args.render_video:
        video_name = f"{Path(args.model_path).stem}_video.mp4"
        video_path = os.path.join(args.output_dir, video_name)
        os.makedirs(args.output_dir, exist_ok=True)
        create_video(args.model_path, video_path, args.video_frames, args.resolution, args.fps)
    
    else:
        # Create static renders
        create_renders(args.model_path, args.output_dir, args.num_views, args.resolution)
    
    print("\nVisualization complete!")
    print(f"Check the '{args.output_dir}' directory for output files.")

if __name__ == "__main__":
    main()
