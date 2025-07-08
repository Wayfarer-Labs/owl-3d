#!/usr/bin/env python3
"""
online_tsdf_pipeline.py

Provides a TSDF-based online reconstruction pipeline for filtering raw point clouds.
"""

import torch
import numpy as np
from typing import List, Optional, Union, Tuple, Dict

# Import the TSDF fusion implementation
from tsdf_fusion import TSDFFusion


def online_tsdf_reconstruction(
    depth_maps: List[np.ndarray],
    camera_poses: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    voxel_size: float = 0.05,
    volume_bounds: Union[Tuple[float, float, float, float, float, float], torch.Tensor] = (-2.0, -2.0, -2.0, 2.0, 2.0, 2.0),
    truncation_distance: float = 0.1,
    min_weight_threshold: float = 1.0,
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """
    Fuse a sequence of depth maps and camera poses online using TSDF, and
    extract a filtered point cloud of surface voxels.

    Args:
        depth_maps: List of depth images as numpy arrays (H×W).
        camera_poses: Tensor of shape (N, 4, 4) with camera-to-world poses.
        camera_intrinsics: Tensor of shape (3, 3) intrinsics matrix.
        voxel_size: Size of each TSDF voxel.
        volume_bounds: Bounds of TSDF volume (x_min, y_min, z_min, x_max, y_max, z_max).
        truncation_distance: Truncation distance for TSDF update.
        min_weight_threshold: Minimum voxel weight to keep in final point cloud.
        device: Device to run on ('cuda' or 'cpu').

    Returns:
        filtered_pc: Tensor of shape (M, 3) of filtered point coordinates.
    """
    # Select device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Initialize TSDF fusion module
    tsdf = TSDFFusion(
        voxel_size=voxel_size,
        volume_bounds=volume_bounds,
        truncation_distance=truncation_distance,
        device=device
    )

    # Process each frame
    for idx, (depth_np, pose) in enumerate(zip(depth_maps, camera_poses)):
        # Convert depth to tensor
        depth_tensor = torch.from_numpy(depth_np).float().to(device)
        # Ensure pose is on device
        pose = pose.to(device)
        # Update TSDF volume with this frame
        tsdf.update_volume(depth_tensor, pose, camera_intrinsics.to(device))

    # Extract final filtered point cloud
    filtered_pc = tsdf.extract_pointcloud(min_weight_threshold=min_weight_threshold)
    return filtered_pc


class OnlineTSDFColorPipeline:
    """
    Online TSDF reconstruction with color integration.
    Call update_batch for new depth/rgb/pose batches and extract_colored_pointcloud.
    """
    def __init__(
        self,
        camera_intrinsics: torch.Tensor,
        voxel_size: float = 0.05,
        volume_bounds: Union[Tuple[float, float, float, float, float, float], torch.Tensor] = (-2.0, -2.0, -2.0, 2.0, 2.0, 2.0),
        truncation_distance: float = 0.1,
        device: Optional[Union[str, torch.device]] = None
    ):
        # init device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        self.device = device
        # store intrinsics
        self.camera_intrinsics = camera_intrinsics.to(device)
        # TSDF module
        self.tsdf = TSDFFusion(
            voxel_size=voxel_size,
            volume_bounds=volume_bounds,
            truncation_distance=truncation_distance,
            device=device
        )
        # store frames for color mapping
        self.rgb_frames: List[np.ndarray] = []
        self.poses: List[torch.Tensor] = []

    def update_batch(
        self,
        depth_maps: List[np.ndarray],
        rgb_maps: List[np.ndarray],
        poses: torch.Tensor
    ):
        """
        Fuse batch of frames into TSDF and store rgb/poses for later color projection.
        depth_maps: list of HxW arrays, rgb_maps: list of HxWx3 arrays, poses: Nx4x4 tensor
        """
        for depth_np, rgb_np, pose in zip(depth_maps, rgb_maps, poses):
            # update TSDF
            depth_tensor = torch.from_numpy(depth_np).float().to(self.device)
            self.tsdf.update_volume(depth_tensor, pose.to(self.device), self.camera_intrinsics)
            # store for color projection
            self.rgb_frames.append(rgb_np)
            self.poses.append(pose.to(self.device))

    def extract_colored_pointcloud(
        self,
        min_weight_threshold: float = 1.0
    ) -> torch.Tensor:
        """
        Extract static surface point cloud from TSDF and assign colors by projecting into stored frames.
        Returns Nx6 tensor [x,y,z,r,g,b]
        """
        # extract points
        points = self.tsdf.extract_pointcloud(min_weight_threshold)
        pts = points.cpu().numpy()
        colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
        # for each point, sample first available color
        inv_intr = torch.inverse(self.camera_intrinsics)
        for i, p in enumerate(pts):
            xyz = torch.tensor([p[0], p[1], p[2], 1.0], device=self.device)
            for rgb, pose in zip(self.rgb_frames, self.poses):
                # world2cam
                cam_coords = (torch.inverse(pose) @ xyz)[:3]
                if cam_coords[2] <= 0:
                    continue
                uv = (self.camera_intrinsics @ cam_coords).cpu().numpy()
                u, v = int(uv[0]/uv[2]), int(uv[1]/uv[2])
                H, W, _ = rgb.shape
                if 0 <= u < W and 0 <= v < H:
                    colors[i] = rgb[v, u]
                    break
        # concatenate
        pc_color = torch.from_numpy(np.hstack([pts, colors])).float()
        return pc_color
