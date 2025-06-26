#!/usr/bin/env python3
"""
Debug script to investigate the black rendering issue.
This will help identify where the problem lies in the rendering pipeline.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import gsplat

def debug_megasam_data(megasam_file):
    """Debug the MEGA-SAM data to understand the issue."""
    print("=== Debugging MEGA-SAM Data ===")
    
    data = np.load(megasam_file)
    images = data['images']
    depths = data['depths']
    intrinsic = data['intrinsic']
    cam_c2w = data['cam_c2w']
    
    print(f"Images shape: {images.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Images range: {images.min()} to {images.max()}")
    
    print(f"Depths shape: {depths.shape}")
    print(f"Depths dtype: {depths.dtype}")
    print(f"Depths range: {depths.min()} to {depths.max()}")
    
    print(f"Intrinsic matrix:\n{intrinsic}")
    print(f"First camera pose:\n{cam_c2w[0]}")
    
    # Check if images are actually valid
    first_img = images[0]
    print(f"First image stats: mean={first_img.mean():.3f}, std={first_img.std():.3f}")
    
    # Check depth validity
    first_depth = depths[0]
    valid_depth = first_depth > 0
    print(f"Valid depth pixels: {valid_depth.sum()}/{valid_depth.size} ({100*valid_depth.sum()/valid_depth.size:.1f}%)")
    
    # Visualize first image and depth
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(first_img)
    axes[0].set_title("First Image")
    axes[0].axis('off')
    
    depth_vis = first_depth.copy()
    depth_vis[depth_vis <= 0] = np.nan
    im = axes[1].imshow(depth_vis, cmap='viridis')
    axes[1].set_title("First Depth Map")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('/home/ben/workspace/owl-3d/debug_input_data.png', dpi=150)
    plt.close()
    
    return images, depths, intrinsic, cam_c2w

def debug_point_initialization(images, depths, intrinsic, cam_c2w, max_points=10000):
    """Debug the 3D point initialization process."""
    print("\n=== Debugging Point Initialization ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process just the first frame for debugging
    img = images[0].astype(np.float32) / 255.0
    depth = depths[0].astype(np.float32)
    cam_pose = cam_c2w[0].astype(np.float32)
    
    print(f"Image shape: {img.shape}")
    print(f"Depth shape: {depth.shape}")
    print(f"Image range after normalization: {img.min():.3f} to {img.max():.3f}")
    
    # Get camera intrinsics
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    print(f"Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    # Find valid depth pixels
    valid_mask = depth > 0
    print(f"Valid depth pixels: {valid_mask.sum()}/{valid_mask.size}")
    
    if valid_mask.sum() == 0:
        print("ERROR: No valid depth pixels found!")
        return None, None
    
    # Get pixel coordinates
    H, W = depth.shape
    v_coords, u_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    u_valid = u_coords[valid_mask]
    v_valid = v_coords[valid_mask]
    z_valid = depth[valid_mask]
    
    # Subsample if too many points
    if len(u_valid) > max_points:
        indices = np.random.choice(len(u_valid), max_points, replace=False)
        u_valid = u_valid[indices]
        v_valid = v_valid[indices]
        z_valid = z_valid[indices]
    
    print(f"Using {len(u_valid)} points for initialization")
    
    # Unproject to camera coordinates
    x_cam = (u_valid - cx) * z_valid / fx
    y_cam = (v_valid - cy) * z_valid / fy
    pts_cam = np.stack([x_cam, y_cam, z_valid], axis=-1)
    
    print(f"Camera space points range:")
    print(f"  X: {x_cam.min():.3f} to {x_cam.max():.3f}")
    print(f"  Y: {y_cam.min():.3f} to {y_cam.max():.3f}")
    print(f"  Z: {z_valid.min():.3f} to {z_valid.max():.3f}")
    
    # Transform to world coordinates
    pts_hom = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
    pts_world = (cam_pose @ pts_hom.T).T[:, :3]
    
    print(f"World space points range:")
    print(f"  X: {pts_world[:, 0].min():.3f} to {pts_world[:, 0].max():.3f}")
    print(f"  Y: {pts_world[:, 1].min():.3f} to {pts_world[:, 1].max():.3f}")
    print(f"  Z: {pts_world[:, 2].min():.3f} to {pts_world[:, 2].max():.3f}")
    
    # Get colors
    colors = img[valid_mask]
    if len(colors) > max_points:
        colors = colors[indices]
    
    print(f"Colors shape: {colors.shape}")
    print(f"Colors range: {colors.min():.3f} to {colors.max():.3f}")
    
    # Convert to tensors
    points_tensor = torch.tensor(pts_world, dtype=torch.float32, device=device)
    colors_tensor = torch.tensor(colors, dtype=torch.float32, device=device)
    
    return points_tensor, colors_tensor

def debug_gsplat_model(points, colors):
    """Debug the GSplat model creation."""
    print("\n=== Debugging GSplat Model ===")
    
    device = points.device
    
    # Initialize GSplat parameters
    means = points.clone()
    scales = torch.ones_like(points) * 0.005  # Very small initial scale
    quats = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=device).repeat(points.shape[0], 1)
    colors_param = colors.clone()
    opacities = torch.ones(points.shape[0], device=device) * 0.8  # Correct shape (N,)
    
    print(f"Means shape: {means.shape}, range: {means.min():.3f} to {means.max():.3f}")
    print(f"Scales shape: {scales.shape}, range: {scales.min():.6f} to {scales.max():.6f}")
    print(f"Quats shape: {quats.shape}")
    print(f"Colors shape: {colors_param.shape}, range: {colors_param.min():.3f} to {colors_param.max():.3f}")
    print(f"Opacities shape: {opacities.shape}, range: {opacities.min():.3f} to {opacities.max():.3f}")
    
    # Check for NaN or inf values
    def check_tensor(name, tensor):
        if torch.isnan(tensor).any():
            print(f"WARNING: {name} contains NaN values!")
        if torch.isinf(tensor).any():
            print(f"WARNING: {name} contains inf values!")
    
    check_tensor("means", means)
    check_tensor("scales", scales)
    check_tensor("quats", quats)
    check_tensor("colors", colors_param)
    check_tensor("opacities", opacities)
    
    return {
        'means': means,
        'scales': scales,
        'quats': quats,
        'colors': colors_param,
        'opacities': opacities
    }

def debug_rendering(model, cam_pose, intrinsic, width=400, height=300):
    """Debug the rendering process."""
    print("\n=== Debugging Rendering ===")
    
    device = model['means'].device
    
    # Prepare camera matrices
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # Scale intrinsics
    original_width = cx * 2
    original_height = cy * 2
    
    scale_x = width / original_width
    scale_y = height / original_height
    
    fx_scaled = fx * scale_x
    fy_scaled = fy * scale_y
    cx_scaled = width / 2
    cy_scaled = height / 2
    
    print(f"Original resolution: {original_width:.0f}x{original_height:.0f}")
    print(f"Target resolution: {width}x{height}")
    print(f"Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
    print(f"Scaled intrinsics: fx={fx_scaled:.1f}, fy={fy_scaled:.1f}, cx={cx_scaled:.1f}, cy={cy_scaled:.1f}")
    
    # Convert camera pose
    world_to_cam = torch.linalg.inv(torch.tensor(cam_pose, dtype=torch.float32, device=device))
    print(f"World-to-camera matrix shape: {world_to_cam.shape}")
    
    # Create intrinsics matrix
    K = torch.tensor([[fx_scaled, 0, cx_scaled], 
                      [0, fy_scaled, cy_scaled], 
                      [0, 0, 1]], dtype=torch.float32, device=device)
    
    print(f"Intrinsics matrix:\n{K}")
    
    # Check if points are in front of camera
    points_cam = (world_to_cam[:3, :3] @ model['means'].T + world_to_cam[:3, 3:4]).T
    z_cam = points_cam[:, 2]
    in_front = z_cam > 0
    print(f"Points in front of camera: {in_front.sum()}/{len(in_front)} ({100*in_front.sum()/len(in_front):.1f}%)")
    print(f"Camera space Z range: {z_cam.min():.3f} to {z_cam.max():.3f}")
    
    # Project points to screen
    points_screen = torch.zeros_like(points_cam[:, :2])
    valid_z = z_cam > 0
    if valid_z.any():
        points_screen[valid_z, 0] = fx_scaled * points_cam[valid_z, 0] / points_cam[valid_z, 2] + cx_scaled
        points_screen[valid_z, 1] = fy_scaled * points_cam[valid_z, 1] / points_cam[valid_z, 2] + cy_scaled
        
        # Check if points are in screen bounds
        in_screen_x = (points_screen[:, 0] >= 0) & (points_screen[:, 0] < width)
        in_screen_y = (points_screen[:, 1] >= 0) & (points_screen[:, 1] < height)
        in_screen = in_screen_x & in_screen_y & valid_z
        
        print(f"Points in screen bounds: {in_screen.sum()}/{len(in_screen)} ({100*in_screen.sum()/len(in_screen):.1f}%)")
        print(f"Screen coords range: x=[{points_screen[:, 0].min():.1f}, {points_screen[:, 0].max():.1f}], y=[{points_screen[:, 1].min():.1f}, {points_screen[:, 1].max():.1f}]")
    
    # Try rendering
    try:
        print("Attempting to render...")
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
        
        print(f"Render successful!")
        print(f"Rendered image shape: {renders.shape}")
        print(f"Rendered image range: {renders.min():.6f} to {renders.max():.6f}")
        
        # Convert to numpy
        rendered_image = renders[0].detach().cpu().numpy()
        
        # Check for non-zero pixels
        non_zero = rendered_image > 1e-6
        print(f"Non-zero pixels: {non_zero.sum()}/{non_zero.size} ({100*non_zero.sum()/non_zero.size:.3f}%)")
        
        # Analyze alpha channel
        alpha_np = alphas[0].detach().cpu().numpy()
        print(f"Alpha range: {alpha_np.min():.6f} to {alpha_np.max():.6f}")
        alpha_non_zero = alpha_np > 1e-6
        print(f"Non-zero alpha pixels: {alpha_non_zero.sum()}/{alpha_non_zero.size} ({100*alpha_non_zero.sum()/alpha_non_zero.size:.3f}%)")
        
        return rendered_image, alpha_np
        
    except Exception as e:
        print(f"Rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    megasam_file = "/home/ben/workspace/mega-sam/outputs/CoD-sequence-test_droid.npz"
    
    # Debug MEGA-SAM data
    images, depths, intrinsic, cam_c2w = debug_megasam_data(megasam_file)
    
    # Debug point initialization
    points, colors = debug_point_initialization(images, depths, intrinsic, cam_c2w)
    
    if points is None:
        print("Failed to initialize points!")
        return
    
    # Debug GSplat model
    model = debug_gsplat_model(points, colors)
    
    # Debug rendering
    rendered_image, alpha = debug_rendering(model, cam_c2w[0], intrinsic)
    
    if rendered_image is not None:
        # Save debug visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(images[0])
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(depths[0], cmap='viridis')
        axes[0, 1].set_title("Depth Map")
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(rendered_image)
        axes[1, 0].set_title("GSplat Render")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(alpha, cmap='gray')
        axes[1, 1].set_title("Alpha Channel")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('/home/ben/workspace/owl-3d/debug_rendering_comparison.png', dpi=150)
        plt.close()
        
        print("\nDebug visualization saved to 'debug_rendering_comparison.png'")
    
    print("\nDebugging complete!")

if __name__ == "__main__":
    main()
