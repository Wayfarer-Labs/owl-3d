#!/usr/bin/env python3
"""
Generate and visualize 3D point cloud from MEGA-SAM results
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path

def unproject_depth_to_pointcloud(depth, intrinsic, pose, image=None, subsample=2):
    """Convert depth map to 3D point cloud."""
    H, W = depth.shape
    
    # Create pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u[::subsample, ::subsample].flatten()
    v = v[::subsample, ::subsample].flatten()
    depth_sub = depth[::subsample, ::subsample].flatten()
    
    # Remove invalid depths
    valid = depth_sub > 0
    u, v, depth_sub = u[valid], v[valid], depth_sub[valid]
    
    if len(u) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    # Unproject to camera coordinates
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    x_cam = (u - cx) * depth_sub / fx
    y_cam = (v - cy) * depth_sub / fy
    z_cam = depth_sub
    
    # Camera coordinates
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    
    # Transform to world coordinates
    points_hom = np.hstack([points_cam, np.ones((len(points_cam), 1))])
    points_world = (pose @ points_hom.T).T[:, :3]
    
    # Get colors if image provided
    colors = None
    if image is not None:
        image_sub = image[::subsample, ::subsample]
        colors = image_sub[v[valid]//subsample, u[valid]//subsample] / 255.0
    
    return points_world, colors

def create_point_cloud_ply(output_file, ply_path="cod_pointcloud.ply", max_frames=50, subsample=8):
    """Create a PLY point cloud file."""
    try:
        data = np.load(output_file)
        images = data['images']
        depths = data['depths']
        intrinsic = data['intrinsic']
        cam_c2w = data['cam_c2w']
        
        print(f"Creating point cloud from {len(images)} frames...")
        
        all_points = []
        all_colors = []
        
        # Process frames (subsample to avoid too many points)
        frame_step = max(1, len(images) // max_frames)
        
        for i in range(0, len(images), frame_step):
            print(f"Processing frame {i}/{len(images)}")
            
            depth = depths[i]
            image = images[i]
            pose = cam_c2w[i]
            
            points, colors = unproject_depth_to_pointcloud(
                depth, intrinsic, pose, image, subsample
            )
            
            if len(points) > 0:
                all_points.append(points)
                if colors is not None:
                    all_colors.append(colors)
        
        if not all_points:
            print("No valid points found!")
            return False
        
        # Combine all points
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors) if all_colors else None
        
        print(f"Generated {len(all_points)} 3D points")
        
        # Write PLY file
        with open(ply_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(all_points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if all_colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for i, point in enumerate(all_points):
                if all_colors is not None:
                    color = (all_colors[i] * 255).astype(np.uint8)
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {color[0]} {color[1]} {color[2]}\n")
                else:
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        
        print(f"Point cloud saved to {ply_path}")
        print("You can open this file in:")
        print("  - MeshLab")
        print("  - CloudCompare") 
        print("  - Open3D viewer")
        print("  - Blender")
        
        return True
        
    except Exception as e:
        print(f"Error creating point cloud: {e}")
        return False

def create_depth_video(output_file, video_path="cod_depth_video.mp4", fps=10):
    """Create a video showing RGB and depth side by side."""
    try:
        data = np.load(output_file)
        images = data['images']
        depths = data['depths']
        
        print(f"Creating depth video from {len(images)} frames...")
        
        # Setup video writer
        H, W = images.shape[1], images.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W*2, H))
        
        for i in range(len(images)):
            # RGB image
            img_rgb = images[i]
            
            # Colorize depth
            depth = depths[i]
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_colored = plt.cm.viridis(depth_norm)[:, :, :3]
            depth_colored = (depth_colored * 255).astype(np.uint8)
            
            # Combine side by side
            combined = np.hstack([img_rgb, depth_colored])
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            
            video_writer.write(combined_bgr)
        
        video_writer.release()
        print(f"Depth video saved to {video_path}")
        
        return True
        
    except Exception as e:
        print(f"Error creating video: {e}")
        return False

if __name__ == "__main__":
    # Find CoD sequence output files specifically
    import argparse

    parser = argparse.ArgumentParser(description="Create 3D visualizations from MEGA-SAM outputs.")
    parser.add_argument('--outputs_dir', type=str, default=None, help="Path to the CoD sequence output file (default: search for CoD-sequence_droid.npz)")
    output_files = []
    args = parser.parse_args()
    outputs_dir = args.outputs_dir
    
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
        if not output_files and os.path.exists(outputs_dir):
            for f in os.listdir(outputs_dir):
                if f.endswith("_droid.npz"):
                    output_files.append(os.path.join(outputs_dir, f))
    
    if not output_files:
        print("No output files found. Run evaluate_cod.sh first to generate CoD sequence outputs.")
        exit(1)
    
    output_file = output_files[0]
    print(f"Processing: {output_file}")
    
    # Create visualizations
    print("\n1. Creating point cloud...")
    create_point_cloud_ply(output_file, "cod_pointcloud.ply")
    
    print("\n2. Creating depth video...")
    create_depth_video(output_file, "cod_depth_video.mp4")
    
    print("\nVisualization complete!")
    print("Files created:")
    print("  - cod_pointcloud.ply (3D point cloud)")
    print("  - cod_depth_video.mp4 (RGB + depth video)")
