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
        # TSDF module
        self.tsdf = TSDFFusion(
            voxel_size=voxel_size,
            volume_bounds=volume_bounds,
            truncation_distance=truncation_distance,
            device=device
        )
        # store intrinsics
        self.camera_intrinsics = camera_intrinsics.to(device)
        # store per-frame projection matrices for color
        self.projection_matrices: List[torch.Tensor] = []
        self.rgb_frames: List[np.ndarray] = []  # list of HxWx3 RGB frames

    def update_batch(
        self,
        depth_maps: List[np.ndarray],
        rgb_maps: List[np.ndarray],
        poses: torch.Tensor,
        projection_matrices: List[np.ndarray]
    ):
        """
        Fuse batch of frames into TSDF and store rgb/poses for later color projection.
        depth_maps: list of HxW arrays, rgb_maps: list of HxWx3 arrays, poses: Nx4x4 tensor
        """
        for depth_np, rgb_np, pose, proj in zip(depth_maps, rgb_maps, poses, projection_matrices):
            # update TSDF with fixed intrinsics
            depth_tensor = torch.from_numpy(depth_np).float().to(self.device)
            self.tsdf.update_volume(depth_tensor, pose.to(self.device), self.camera_intrinsics)
            # store for color projection
            self.rgb_frames.append(rgb_np)
            self.projection_matrices.append(torch.from_numpy(proj).float().to(self.device))

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
        # for each point, sample colors from all frames and average valid samples
        for i, p in enumerate(pts):
            xyz = torch.tensor([p[0], p[1], p[2], 1.0], device=self.device)
            color_samples = []
            # project using per-frame full projection matrix
            for rgb, proj in zip(self.rgb_frames, self.projection_matrices):
                uvh = proj @ xyz  # [u', v', w']
                if uvh[2] <= 0:
                    continue
                u = int((uvh[0] / uvh[2]).cpu().item())
                v = int((uvh[1] / uvh[2]).cpu().item())
                H, W, _ = rgb.shape
                if 0 <= u < W and 0 <= v < H:
                    color_samples.append(rgb[v, u])
            if color_samples:
                colors[i] = np.mean(color_samples, axis=0).astype(np.uint8)
        # concatenate
        pc_color = torch.from_numpy(np.hstack([pts, colors])).float()
        return pc_color
