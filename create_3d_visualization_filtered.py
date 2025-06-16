#!/usr/bin/env python3
"""
Generate and visualize 3D point cloud from MEGA-SAM results with TSDF fusion filtering
This version uses TSDF fusion to filter out dynamic elements before creating visualizations.

Usage:
    uv run create_3d_visualization_filtered.py --outputs_dir ./output/path
    uv run create_3d_visualization_filtered.py --outputs_dir ./output/path --comparison
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch
from pathlib import Path
from tsdf_fusion import TSDFFusion

def load_data_for_tsdf(output_file):
    """Load and prepare data for TSDF processing."""
    data = np.load(output_file)
    images = data['images']
    depths = data['depths']
    intrinsic = data['intrinsic']
    cam_c2w = data['cam_c2w']
    
    print(f"Loaded data: {len(images)} frames")
    print(f"Image shape: {images[0].shape}")
    print(f"Depth shape: {depths[0].shape}")
    print(f"Intrinsics: {intrinsic}")
    
    return images, depths, intrinsic, cam_c2w

def process_with_tsdf_fusion(images, depths, intrinsic, cam_c2w, 
                           voxel_size=0.03, max_frames=50, 
                           min_weight_threshold=3.0,
                           device='cuda'):
    """
    Process the sequence through TSDF fusion to filter dynamic elements.
    
    Args:
        images: RGB images array
        depths: Depth maps array  
        intrinsic: Camera intrinsics matrix
        cam_c2w: Camera poses (camera to world)
        voxel_size: Size of each voxel
        max_frames: Maximum number of frames to process
        min_weight_threshold: Minimum weight for point extraction
        device: Device to run on
    
    Returns:
        filtered_points: Filtered 3D points
        filtered_colors: Corresponding colors
        tsdf_info: Information about TSDF processing
    """
    device = device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Estimate scene bounds from first few frames
    bounds = estimate_scene_bounds(depths[:min(5, len(depths))], intrinsic, cam_c2w[:min(5, len(cam_c2w))])
    print(f"Estimated scene bounds: {bounds}")
    
    # Initialize TSDF fusion
    tsdf = TSDFFusion(
        voxel_size=voxel_size,
        volume_bounds=bounds,
        truncation_distance=voxel_size * 2,  # 2x voxel size
        max_weight=100.0,
        device=device
    )
    
    print(f"Initialized TSDF volume with dimensions: {tsdf.grid_dims}")
    
    # Process frames with TSDF fusion
    # frame_step = max(1, len(images) // max_frames)
    frame_step = 1
    processed_frames = 0
    
    for i in range(0, len(images), frame_step):
        if processed_frames >= max_frames:
            break
            
        print(f"Processing frame {i}/{len(images)} (step {frame_step})")
        
        # Convert to torch tensors
        depth_torch = torch.from_numpy(depths[i]).float().to(device)
        pose_torch = torch.from_numpy(cam_c2w[i]).float().to(device)
        intrinsic_torch = torch.from_numpy(intrinsic).float().to(device)
        
        # Compute frame weight (progressive weighting - later frames get higher weight)
        frame_weight = 1.0 + (processed_frames / max_frames) * 2.0  # Weight from 1.0 to 3.0
        
        # Update TSDF volume
        tsdf.update_volume(
            depth_map=depth_torch,
            camera_pose=pose_torch,
            camera_intrinsics=intrinsic_torch,
            frame_weight=frame_weight
        )
        
        processed_frames += 1
        
        # Print progress info
        if processed_frames % 5 == 0:
            volume_info = tsdf.get_volume_info()
            print(f"  Occupied voxels: {volume_info['num_occupied_voxels']}")
            print(f"  Mean weight: {volume_info['mean_weight']:.2f}")
    
    print(f"\nTSDF fusion completed. Processed {processed_frames} frames.")
    
    # Extract filtered point cloud
    print(f"Extracting points with weight threshold >= {min_weight_threshold}")
    filtered_points_torch = tsdf.extract_pointcloud(min_weight_threshold=min_weight_threshold)
    
    if len(filtered_points_torch) == 0:
        print("Warning: No points extracted! Try lowering min_weight_threshold")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), tsdf.get_volume_info()
    
    # Convert back to numpy
    filtered_points = filtered_points_torch.cpu().numpy()
    print(f"Extracted {len(filtered_points)} filtered points")
    
    # Generate colors for filtered points by projecting back to original images
    print("Mapping colors from original images...")
    filtered_colors = map_colors_from_images(filtered_points, images, depths, intrinsic, cam_c2w, processed_frames)
    
    # Get volume information
    tsdf_info = tsdf.get_volume_info()
    tsdf_info['processed_frames'] = processed_frames
    tsdf_info['extraction_threshold'] = min_weight_threshold
    
    return filtered_points, filtered_colors, tsdf_info

def estimate_scene_bounds(depths, intrinsic, poses, percentile=95):
    """Estimate scene bounds from a few depth maps."""
    all_points = []
    
    for depth, pose in zip(depths, poses):
        # Simple unprojection for bounds estimation
        H, W = depth.shape
        valid_mask = depth > 0
        
        if not valid_mask.any():
            continue
            
        # Subsample for efficiency
        subsample = 4
        depth_sub = depth[::subsample, ::subsample]
        valid_sub = valid_mask[::subsample, ::subsample]
        
        if not valid_sub.any():
            continue
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(0, W, subsample), np.arange(0, H, subsample))
        u = u[valid_sub]
        v = v[valid_sub]
        depth_vals = depth_sub[valid_sub]
        
        # Unproject to camera coordinates
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        x_cam = (u - cx) * depth_vals / fx
        y_cam = (v - cy) * depth_vals / fy
        z_cam = depth_vals
        
        # Transform to world coordinates
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        points_hom = np.hstack([points_cam, np.ones((len(points_cam), 1))])
        points_world = (pose @ points_hom.T).T[:, :3]
        
        all_points.append(points_world)
    
    if not all_points:
        print("Warning: No valid points for bounds estimation, using default bounds")
        return (-2.0, -2.0, -2.0, 2.0, 2.0, 2.0)
    
    all_points = np.vstack(all_points)
    
    # Use percentiles to avoid outliers
    min_bounds = np.percentile(all_points, 100 - percentile, axis=0)
    max_bounds = np.percentile(all_points, percentile, axis=0)
    
    # Add some padding
    padding = 0.5
    min_bounds -= padding
    max_bounds += padding
    
    bounds = tuple(min_bounds.tolist() + max_bounds.tolist())
    return bounds

def map_colors_from_images(points_3d, images, depths, intrinsic, poses, max_frames_used):
    """
    Map colors to 3D points by projecting them back to the original images.
    
    Args:
        points_3d: 3D points to color (N, 3)
        images: Original RGB images
        depths: Original depth maps
        intrinsic: Camera intrinsics
        poses: Camera poses (cam2world)
        max_frames_used: Number of frames processed
        
    Returns:
        colors: RGB colors for each point (N, 3)
    """
    if len(points_3d) == 0:
        return np.array([]).reshape(0, 3)
    
    colors = np.zeros((len(points_3d), 3))
    color_counts = np.zeros(len(points_3d))  # Track how many frames contributed to each point
    
    # Use fewer frames for efficiency in color mapping
    color_frame_step = max(1, max_frames_used // 10)  # Use up to 10 frames for coloring
    
    for i in range(0, min(max_frames_used, len(images)), color_frame_step):
        if i % 5 == 0:
            print(f"  Color mapping frame {i+1}/{max_frames_used}")
        
        image = images[i]
        depth = depths[i]
        pose = poses[i]
        
        # Transform world points to camera frame
        world2cam = np.linalg.inv(pose)
        points_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        points_cam = (world2cam @ points_homo.T).T[:, :3]
        
        # Project to image plane
        points_proj = (intrinsic @ points_cam.T).T
        points_proj = points_proj[:, :2] / (points_proj[:, 2:3] + 1e-8)
        
        # Check which points are visible and in front of camera
        H, W = image.shape[:2]
        valid_mask = (
            (points_proj[:, 0] >= 0) & (points_proj[:, 0] < W) &
            (points_proj[:, 1] >= 0) & (points_proj[:, 1] < H) &
            (points_cam[:, 2] > 0)  # In front of camera
        )
        
        if not valid_mask.any():
            continue
        
        # Get pixel coordinates
        pixel_coords = points_proj[valid_mask].astype(int)
        pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, W - 1)
        pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, H - 1)
        
        # Check depth consistency (optional - helps with occlusion)
        observed_depths = depth[pixel_coords[:, 1], pixel_coords[:, 0]]
        point_depths = points_cam[valid_mask, 2]
        
        # Allow some tolerance for depth matching
        depth_tolerance = 0.1  # 10cm tolerance
        depth_consistent = np.abs(observed_depths - point_depths) < depth_tolerance
        depth_consistent = depth_consistent & (observed_depths > 0)
        
        if not depth_consistent.any():
            continue
        
        # Get colors for valid, depth-consistent points
        valid_indices = np.where(valid_mask)[0][depth_consistent]
        valid_pixels = pixel_coords[depth_consistent]
        
        point_colors = image[valid_pixels[:, 1], valid_pixels[:, 0]] / 255.0
        
        # Accumulate colors
        colors[valid_indices] += point_colors
        color_counts[valid_indices] += 1
    
    # Average colors and handle points without color
    colored_mask = color_counts > 0
    colors[colored_mask] /= color_counts[colored_mask][:, np.newaxis]
    
    # For points without color, use a default color (light gray)
    uncolored_mask = ~colored_mask
    if uncolored_mask.any():
        colors[uncolored_mask] = [0.7, 0.7, 0.7]  # Light gray
        print(f"  Warning: {uncolored_mask.sum()} points couldn't be colored, using default color")
    
    print(f"  Successfully colored {colored_mask.sum()}/{len(points_3d)} points")
    return colors

def create_filtered_point_cloud_ply(output_file, ply_path="cod_pointcloud_filtered.ply", 
                                  max_frames=30, voxel_size=0.03, 
                                  min_weight_threshold=3.0):
    """Create a PLY point cloud file using TSDF filtering."""
    try:
        # Load data
        images, depths, intrinsic, cam_c2w = load_data_for_tsdf(output_file)
        
        # Process with TSDF fusion
        print("\n" + "="*50)
        print("PROCESSING WITH TSDF FUSION")
        print("="*50)
        
        filtered_points, filtered_colors, tsdf_info = process_with_tsdf_fusion(
            images, depths, intrinsic, cam_c2w,
            voxel_size=voxel_size,
            max_frames=max_frames,
            min_weight_threshold=min_weight_threshold
        )
        
        if len(filtered_points) == 0:
            print("No points to save!")
            return False
        
        print(f"\nTSDF Processing Summary:")
        print(f"  Grid dimensions: {tsdf_info['grid_dims']}")
        print(f"  Voxel size: {tsdf_info['voxel_size']}")
        print(f"  Occupied voxels: {tsdf_info['num_occupied_voxels']}")
        print(f"  Mean weight: {tsdf_info['mean_weight']:.2f}")
        print(f"  Processed frames: {tsdf_info['processed_frames']}")
        print(f"  Extraction threshold: {tsdf_info['extraction_threshold']}")
        print(f"  Final point count: {len(filtered_points)}")
        
        # Write PLY file
        print(f"\nWriting filtered point cloud to {ply_path}")
        
        with open(ply_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(filtered_points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for i, point in enumerate(filtered_points):
                color = (filtered_colors[i] * 255).astype(np.uint8)
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {color[0]} {color[1]} {color[2]}\n")
        
        print(f"Filtered point cloud saved to {ply_path}")
        print("You can open this file in:")
        print("  - MeshLab")
        print("  - CloudCompare") 
        print("  - Open3D viewer")
        print("  - Blender")
        
        return True
        
    except Exception as e:
        print(f"Error creating filtered point cloud: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_comparison_visualization(output_file, base_ply="cod_pointcloud.ply", 
                                  filtered_ply="cod_pointcloud_filtered.ply"):
    """Create both original and filtered point clouds for comparison."""
    from create_3d_visualization import create_point_cloud_ply
    
    print("Creating original (unfiltered) point cloud...")
    success_original = create_point_cloud_ply(output_file, base_ply)
    
    print("\nCreating TSDF-filtered point cloud...")
    success_filtered = create_filtered_point_cloud_ply(output_file, filtered_ply)
    
    if success_original and success_filtered:
        print(f"\nComparison files created:")
        print(f"  Original: {base_ply}")
        print(f"  Filtered: {filtered_ply}")
        print("\nLoad both files in a 3D viewer to compare the filtering effect.")
        print("The filtered version should have dynamic elements removed.")
    
    return success_original and success_filtered

def create_depth_video_with_info(output_file, video_path="cod_depth_video_with_tsdf.mp4", 
                                fps=10, max_frames=30, voxel_size=0.03):
    """Create a video showing RGB, depth, and TSDF processing info."""
    try:
        data = np.load(output_file)
        images = data['images']
        depths = data['depths']
        intrinsic = data['intrinsic']
        cam_c2w = data['cam_c2w']
        
        print(f"Creating enhanced depth video from {len(images)} frames...")
        
        # Process with TSDF to get frame-by-frame info
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        bounds = estimate_scene_bounds(depths[:5], intrinsic, cam_c2w[:5])
        
        tsdf = TSDFFusion(
            voxel_size=voxel_size,
            volume_bounds=bounds,
            truncation_distance=voxel_size * 2,
            max_weight=100.0,
            device=device
        )
        
        # Setup video writer
        H, W = images.shape[1], images.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W*2, H + 100))  # Extra height for info
        
        frame_step = max(1, len(images) // max_frames)
        processed_count = 0
        
        for i in range(0, len(images), frame_step):
            if processed_count >= max_frames:
                break
                
            print(f"Processing video frame {i}/{len(images)}")
            
            # RGB image
            img_rgb = images[i]
            
            # Colorize depth
            depth = depths[i]
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_colored = plt.cm.viridis(depth_norm)[:, :, :3]
            depth_colored = (depth_colored * 255).astype(np.uint8)
            
            # Update TSDF
            depth_torch = torch.from_numpy(depth).float().to(device)
            pose_torch = torch.from_numpy(cam_c2w[i]).float().to(device)
            intrinsic_torch = torch.from_numpy(intrinsic).float().to(device)
            
            frame_weight = 1.0 + (processed_count / max_frames) * 2.0
            tsdf.update_volume(depth_torch, pose_torch, intrinsic_torch, frame_weight)
            
            # Get TSDF info
            volume_info = tsdf.get_volume_info()
            
            # Create info panel
            info_panel = np.zeros((100, W*2, 3), dtype=np.uint8)
            
            # Add text info
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (255, 255, 255)
            thickness = 1
            
            info_texts = [
                f"Frame: {i+1}/{len(images)} (processed: {processed_count+1})",
                f"TSDF Occupied Voxels: {volume_info['num_occupied_voxels']}",
                f"Mean Weight: {volume_info['mean_weight']:.2f}",
                f"Frame Weight: {frame_weight:.2f}"
            ]
            
            for j, text in enumerate(info_texts):
                y_pos = 20 + j * 20
                cv2.putText(info_panel, text, (10, y_pos), font, font_scale, color, thickness)
            
            # Combine images
            combined_frames = np.hstack([img_rgb, depth_colored])
            final_frame = np.vstack([combined_frames, info_panel])
            final_frame_bgr = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
            
            video_writer.write(final_frame_bgr)
            processed_count += 1
        
        video_writer.release()
        print(f"Enhanced depth video saved to {video_path}")
        
        return True
        
    except Exception as e:
        print(f"Error creating video: {e}")
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create 3D visualizations with TSDF filtering from MEGA-SAM outputs.")
    parser.add_argument('--outputs_dir', type=str, default=None, help="Path to the CoD sequence output directory")
    parser.add_argument('--voxel_size', type=float, default=0.03, help="TSDF voxel size (default: 0.03)")
    parser.add_argument('--weight_threshold', type=float, default=3.0, help="Minimum weight threshold for point extraction (default: 3.0)")
    parser.add_argument('--max_frames', type=int, default=30, help="Maximum frames to process (default: 30)")
    parser.add_argument('--comparison', action='store_true', help="Create both original and filtered point clouds for comparison")
    
    args = parser.parse_args()
    outputs_dir = args.outputs_dir
    
    # Find CoD sequence output files
    output_files = []
    
    # Look for CoD-sequence file first
    if outputs_dir:
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
            if not output_files and os.path.exists(outputs_dir):
                for f in os.listdir(outputs_dir):
                    if f.endswith("_droid.npz"):
                        output_files.append(os.path.join(outputs_dir, f))
    
    if not output_files:
        print("No output files found. Please specify --outputs_dir with valid MEGA-SAM outputs.")
        print("Example: uv run create_3d_visualization_filtered.py --outputs_dir ./outputs")
        print("Or:      uv run create_3d_visualization_filtered.py --outputs_dir ./megasam/outputs --comparison")
        exit(1)
    
    output_file = output_files[0]
    print(f"Processing: {output_file}")
    print(f"TSDF Parameters:")
    print(f"  Voxel size: {args.voxel_size}")
    print(f"  Weight threshold: {args.weight_threshold}")
    print(f"  Max frames: {args.max_frames}")
    
    if args.comparison:
        print("\n" + "="*60)
        print("CREATING COMPARISON VISUALIZATION")
        print("="*60)
        success = create_comparison_visualization(output_file)
    else:
        print("\n" + "="*60)
        print("CREATING TSDF-FILTERED VISUALIZATION")
        print("="*60)
        
        # Create filtered visualizations
        print("1. Creating TSDF-filtered point cloud...")
        success1 = create_filtered_point_cloud_ply(
            output_file, 
            "cod_pointcloud_filtered.ply",
            max_frames=args.max_frames,
            voxel_size=args.voxel_size,
            min_weight_threshold=args.weight_threshold
        )
        
        print("\n2. Creating enhanced depth video...")
        success2 = create_depth_video_with_info(
            output_file, 
            "cod_depth_video_with_tsdf.mp4",
            max_frames=args.max_frames,
            voxel_size=args.voxel_size
        )
        
        success = success1 and success2
    
    if success:
        print("\n" + "="*60)
        print("VISUALIZATION COMPLETE!")
        print("="*60)
        print("Files created:")
        if args.comparison:
            print("  - cod_pointcloud.ply (original)")
            print("  - cod_pointcloud_filtered.ply (TSDF filtered)")
        else:
            print("  - cod_pointcloud_filtered.ply (TSDF filtered point cloud)")
            print("  - cod_depth_video_with_tsdf.mp4 (enhanced depth video)")
        print("\nThe TSDF filtering has removed dynamic elements, keeping only static structure.")
        print("Use a 3D viewer like MeshLab or CloudCompare to visualize the results.")
    else:
        print("Some errors occurred during processing. Check the output above.")
