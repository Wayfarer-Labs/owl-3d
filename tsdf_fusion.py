"""
TSDF (Truncated Signed Distance Function) Fusion Implementation in PyTorch

This implementation follows the standard TSDF fusion algorithm with weighted averaging
to incrementally build 3D reconstruction while filtering out dynamic elements.

The voxel update rule follows:
D'(v) = (W(v) * D(v) + w_i * d_i(v)) / (W(v) + w_i)
W'(v) = W(v) + w_i

where:
- D(v), W(v): current TSDF value and weight of voxel v
- d_i(v): truncated signed distance from frame i
- w_i: frame-dependent confidence weight
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union


class TSDFFusion(nn.Module):
    """
    TSDF Fusion module for incremental 3D reconstruction with dynamic filtering.
    
    This implementation creates a voxel grid and fuses depth observations from multiple
    frames using weighted averaging. Dynamic elements are naturally filtered out due
    to inconsistent depth observations across frames.
    """
    
    def __init__(
        self,
        voxel_size: float = 0.05,
        volume_bounds: Union[Tuple[float, float, float, float, float, float], torch.Tensor] = (-2.0, -2.0, -2.0, 2.0, 2.0, 2.0),
        truncation_distance: float = 0.1,
        max_weight: float = 100.0,
        device: str = 'cuda'
    ):
        """
        Initialize TSDF Fusion module.
        
        Args:
            voxel_size: Size of each voxel in world coordinates
            volume_bounds: (x_min, y_min, z_min, x_max, y_max, z_max) in world coordinates
            truncation_distance: Distance at which TSDF is truncated
            max_weight: Maximum weight for any voxel to prevent overflow
            device: Device to run computations on
        """
        super().__init__()
        
        self.voxel_size = voxel_size
        self.truncation_distance = truncation_distance
        self.max_weight = max_weight
        self.device = device
        
        if isinstance(volume_bounds, (list, tuple)):
            volume_bounds = torch.tensor(volume_bounds, dtype=torch.float32, device=device)
        
        self.volume_bounds = volume_bounds.to(device)
        
        # Calculate volume dimensions
        volume_size = self.volume_bounds[3:] - self.volume_bounds[:3]
        self.grid_dims = (volume_size / voxel_size).ceil().long()
        
        # Initialize TSDF and weight volumes
        self.register_buffer('tsdf_volume', torch.ones(*self.grid_dims, dtype=torch.float32, device=device))
        self.register_buffer('weight_volume', torch.zeros(*self.grid_dims, dtype=torch.float32, device=device))
        
        # Precompute voxel coordinates for efficiency
        self._init_voxel_coords()
    
    def _init_voxel_coords(self):
        """Precompute world coordinates for each voxel center."""
        x = torch.linspace(
            self.volume_bounds[0] + self.voxel_size / 2,
            self.volume_bounds[3] - self.voxel_size / 2,
            self.grid_dims[0],
            device=self.device
        )
        y = torch.linspace(
            self.volume_bounds[1] + self.voxel_size / 2,
            self.volume_bounds[4] - self.voxel_size / 2,
            self.grid_dims[1],
            device=self.device
        )
        z = torch.linspace(
            self.volume_bounds[2] + self.voxel_size / 2,
            self.volume_bounds[5] - self.voxel_size / 2,
            self.grid_dims[2],
            device=self.device
        )
        
        # Create meshgrid and reshape to (N, 3) where N = total voxels
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        self.register_buffer('voxel_coords', torch.stack([
            grid_x.flatten(), grid_y.flatten(), grid_z.flatten()
        ], dim=1))  # Shape: (N, 3)
    
    def world_to_voxel(self, world_coords: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to voxel indices."""
        voxel_coords = (world_coords - self.volume_bounds[:3]) / self.voxel_size
        return voxel_coords.floor().long()
    
    def compute_tsdf_values(
        self,
        depth_map: torch.Tensor,
        camera_pose: torch.Tensor,
        camera_intrinsics: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute TSDF values for all voxels given a depth observation.
        
        Args:
            depth_map: Depth map of shape (H, W)
            camera_pose: Camera pose matrix of shape (4, 4) - camera to world
            camera_intrinsics: Camera intrinsics matrix of shape (3, 3)
            
        Returns:
            tsdf_values: TSDF values for each voxel
            valid_mask: Mask indicating which voxels are within view
            distances: Raw distances before truncation
        """
        # Transform voxel coordinates to camera frame
        world_coords_homo = torch.cat([
            self.voxel_coords, 
            torch.ones(len(self.voxel_coords), 1, device=self.device)
        ], dim=1)  # (N, 4)
        
        # Camera pose is cam2world, so we need world2cam
        world2cam = torch.inverse(camera_pose)
        cam_coords = (world2cam @ world_coords_homo.T).T[:, :3]  # (N, 3)
        
        # Project to image plane
        pixel_coords = (camera_intrinsics @ cam_coords.T).T  # (N, 3)
        pixel_coords = pixel_coords[:, :2] / (pixel_coords[:, 2:3] + 1e-8)  # (N, 2)
        
        # Get depth values from camera
        voxel_depths = cam_coords[:, 2]  # Z coordinate in camera frame
        
        # Check which voxels project within image bounds
        H, W = depth_map.shape
        valid_projection = (
            (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < W) &
            (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < H) &
            (voxel_depths > 0)  # In front of camera
        )
        
        # Sample depth values from depth map
        pixel_coords_int = pixel_coords.round().long()
        pixel_coords_int[:, 0] = torch.clamp(pixel_coords_int[:, 0], 0, W - 1)
        pixel_coords_int[:, 1] = torch.clamp(pixel_coords_int[:, 1], 0, H - 1)
        
        observed_depths = depth_map[pixel_coords_int[:, 1], pixel_coords_int[:, 0]]
        
        # Compute signed distances
        signed_distances = observed_depths - voxel_depths
        
        # Apply truncation
        tsdf_values = torch.clamp(
            signed_distances / self.truncation_distance,
            min=-1.0, max=1.0
        )
        
        # Mask out invalid voxels
        valid_mask = valid_projection & (observed_depths > 0)
        
        return tsdf_values, valid_mask, signed_distances
    
    def update_volume(
        self,
        depth_map: torch.Tensor,
        camera_pose: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        frame_weight: float = 1.0,
        confidence_map: Optional[torch.Tensor] = None,
        auto_confidence: bool = True,
        confidence_method: str = "gradient_based"
    ):
        """
        Update TSDF volume with a new depth observation.
        
        Args:
            depth_map: Depth map of shape (H, W)
            camera_pose: Camera pose matrix of shape (4, 4)
            camera_intrinsics: Camera intrinsics matrix of shape (3, 3)
            frame_weight: Base weight for this frame
            confidence_map: Optional confidence map of shape (H, W)
            auto_confidence: Whether to automatically generate confidence from depth
            confidence_method: Method for auto-generating confidence ("gradient_based", "smoothness", "combined")
        """
        # Compute TSDF values for this frame
        tsdf_values, valid_mask, _ = self.compute_tsdf_values(
            depth_map, camera_pose, camera_intrinsics
        )
        
        # Reshape to match volume grid
        tsdf_values = tsdf_values.view(self.grid_dims.tolist())
        valid_mask = valid_mask.view(self.grid_dims.tolist())
        
        # Handle confidence mapping
        if confidence_map is not None:
            # Use provided confidence map
            voxel_confidence = self._map_confidence_to_voxels(
                confidence_map, camera_pose, camera_intrinsics
            )
            # Reshape confidence to match volume grid
            voxel_confidence = voxel_confidence.view(self.grid_dims.tolist())
            weights = frame_weight * voxel_confidence * valid_mask.float()
        elif auto_confidence:
            # Generate confidence from depth
            confidence_map = self.generate_depth_confidence(depth_map, confidence_method)
            voxel_confidence = self._map_confidence_to_voxels(
                confidence_map, camera_pose, camera_intrinsics
            )
            # Reshape confidence to match volume grid
            voxel_confidence = voxel_confidence.view(self.grid_dims.tolist())
            weights = frame_weight * voxel_confidence * valid_mask.float()
        else:
            # Use uniform weighting
            weights = frame_weight * valid_mask.float()
        
        # No need to reshape weights again since they already match volume grid dimensions
        
        # Update TSDF volume using weighted averaging
        # D'(v) = (W(v) * D(v) + w_i * d_i(v)) / (W(v) + w_i)
        current_weights = self.weight_volume
        new_weights = current_weights + weights
        
        # Avoid division by zero
        valid_update = new_weights > 0
        
        updated_tsdf = torch.where(
            valid_update,
            (current_weights * self.tsdf_volume + weights * tsdf_values) / new_weights,
            self.tsdf_volume
        )
        
        # Update volumes
        self.tsdf_volume = updated_tsdf
        self.weight_volume = torch.clamp(new_weights, max=self.max_weight)
    
    def extract_mesh(self, min_weight_threshold: float = 1.0):
        """
        Extract mesh from TSDF volume using marching cubes.
        
        Args:
            min_weight_threshold: Minimum weight threshold for valid voxels
            
        Returns:
            vertices, faces: Mesh vertices and faces (requires skimage for marching cubes)
        """
        try:
            from skimage import measure
        except ImportError:
            raise ImportError("skimage is required for mesh extraction. Install with: pip install scikit-image")
        
        # Mask out low-confidence voxels
        valid_mask = self.weight_volume >= min_weight_threshold
        volume = self.tsdf_volume.clone()
        volume[~valid_mask] = 1.0  # Set invalid voxels to outside surface
        
        # Run marching cubes
        volume_np = volume.cpu().numpy()
        vertices, faces, _, _ = measure.marching_cubes(volume_np, level=0.0)
        
        # Convert voxel coordinates back to world coordinates
        vertices = vertices * self.voxel_size + self.volume_bounds[:3].cpu().numpy()
        
        return vertices, faces
    
    def extract_pointcloud(self, min_weight_threshold: float = 1.0) -> torch.Tensor:
        """
        Extract point cloud from TSDF volume.
        
        Args:
            min_weight_threshold: Minimum weight threshold for valid voxels
            
        Returns:
            points: Point cloud coordinates of shape (N, 3)
        """
        # Find voxels near the surface (TSDF close to 0) with sufficient weight
        surface_mask = (
            (torch.abs(self.tsdf_volume) < 0.1) & 
            (self.weight_volume >= min_weight_threshold)
        )
        
        # Get indices of surface voxels
        surface_indices = torch.nonzero(surface_mask, as_tuple=False)
        
        # Convert to world coordinates
        surface_coords = surface_indices.float() * self.voxel_size + self.volume_bounds[:3]
        
        return surface_coords
    
    def get_confidence_mask(self, min_weight_threshold: float = 5.0) -> torch.Tensor:
        """
        Get confidence mask indicating reliable voxels.
        
        Dynamic elements will have low weights due to inconsistent observations.
        
        Args:
            min_weight_threshold: Minimum weight for reliable voxels
            
        Returns:
            confidence_mask: Boolean mask of reliable voxels
        """
        return self.weight_volume >= min_weight_threshold
    
    def reset_volume(self):
        """Reset TSDF and weight volumes."""
        self.tsdf_volume.fill_(1.0)
        self.weight_volume.fill_(0.0)
    
    def get_volume_info(self) -> dict:
        """Get information about the current volume state."""
        # Handle empty weight volume case
        has_weights = (self.weight_volume > 0).any()
        
        return {
            'grid_dims': self.grid_dims.tolist(),
            'voxel_size': self.voxel_size,
            'volume_bounds': self.volume_bounds.tolist(),
            'num_occupied_voxels': (self.weight_volume > 0).sum().item(),
            'mean_weight': self.weight_volume[self.weight_volume > 0].mean().item() if has_weights else 0,
            'max_weight': self.weight_volume.max().item() if self.weight_volume.numel() > 0 else 0.0,
        }
    
    def _map_confidence_to_voxels(
        self,
        confidence_map: torch.Tensor,
        camera_pose: torch.Tensor,
        camera_intrinsics: torch.Tensor
    ) -> torch.Tensor:
        """
        Map 2D confidence map to 3D voxel confidence values with distance-based weighting.
        
        Args:
            confidence_map: Confidence map of shape (H, W) with values in [0, 1]
            camera_pose: Camera pose matrix of shape (4, 4) - camera to world
            camera_intrinsics: Camera intrinsics matrix of shape (3, 3)
            
        Returns:
            voxel_confidence: Confidence values for each voxel in the grid
        """
        # Transform voxel coordinates to camera frame
        world_coords_homo = torch.cat([
            self.voxel_coords, 
            torch.ones(len(self.voxel_coords), 1, device=self.device)
        ], dim=1)  # (N, 4)
        
        # Camera pose is cam2world, so we need world2cam
        world2cam = torch.inverse(camera_pose)
        cam_coords = (world2cam @ world_coords_homo.T).T[:, :3]  # (N, 3)
        
        # Project to image plane
        pixel_coords = (camera_intrinsics @ cam_coords.T).T  # (N, 3)
        pixel_coords = pixel_coords[:, :2] / (pixel_coords[:, 2:3] + 1e-8)  # (N, 2)
        
        # Check which voxels project within image bounds
        H, W = confidence_map.shape
        valid_projection = (
            (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < W) &
            (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < H) &
            (cam_coords[:, 2] > 0)  # In front of camera
        )
        
        # Initialize confidence values (default to 0 for invalid projections)
        voxel_confidence = torch.zeros(len(self.voxel_coords), device=self.device)
        
        # Use bilinear interpolation to sample confidence values
        if valid_projection.any():
            valid_pixels = pixel_coords[valid_projection]
            valid_depths = cam_coords[valid_projection, 2]  # Z coordinates
            
            # Bilinear interpolation
            x = valid_pixels[:, 0]
            y = valid_pixels[:, 1]
            
            # Get integer coordinates
            x0 = torch.floor(x).long()
            x1 = torch.clamp(x0 + 1, max=W - 1)
            y0 = torch.floor(y).long()
            y1 = torch.clamp(y0 + 1, max=H - 1)
            
            # Get fractional parts
            fx = x - x0.float()
            fy = y - y0.float()
            
            # Clamp coordinates to valid range
            x0 = torch.clamp(x0, 0, W - 1)
            x1 = torch.clamp(x1, 0, W - 1)
            y0 = torch.clamp(y0, 0, H - 1)
            y1 = torch.clamp(y1, 0, H - 1)
            
            # Sample confidence values at corners
            c00 = confidence_map[y0, x0]
            c01 = confidence_map[y1, x0]
            c10 = confidence_map[y0, x1]
            c11 = confidence_map[y1, x1]
            
            # Bilinear interpolation
            interpolated_confidence = (
                c00 * (1 - fx) * (1 - fy) +
                c10 * fx * (1 - fy) +
                c01 * (1 - fx) * fy +
                c11 * fx * fy
            )
            
            # Apply distance-based weighting (closer points get higher confidence)
            # Use inverse distance weighting with a reasonable falloff
            distance_weight = 1.0 / (1.0 + 0.1 * valid_depths)  # Gradual falloff
            
            # Combine image confidence with distance weighting
            final_confidence = interpolated_confidence * distance_weight
            
            # Assign final confidence to valid voxels
            voxel_confidence[valid_projection] = final_confidence
        
        return voxel_confidence
    
    def generate_depth_confidence(
        self,
        depth_map: torch.Tensor,
        method: str = "gradient_based"
    ) -> torch.Tensor:
        """
        Generate confidence map from depth information.
        
        Args:
            depth_map: Input depth map of shape (H, W)
            method: Method for confidence generation ("gradient_based", "smoothness", "combined")
            
        Returns:
            confidence_map: Confidence values in [0, 1] of shape (H, W)
        """
        if method == "gradient_based":
            return self._gradient_based_confidence(depth_map)
        elif method == "smoothness":
            return self._smoothness_based_confidence(depth_map)
        elif method == "combined":
            grad_conf = self._gradient_based_confidence(depth_map)
            smooth_conf = self._smoothness_based_confidence(depth_map)
            return 0.6 * grad_conf + 0.4 * smooth_conf
        else:
            raise ValueError(f"Unknown confidence method: {method}")
    
    def _gradient_based_confidence(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Generate confidence based on depth gradients (lower gradient = higher confidence)."""
        # Compute depth gradients
        grad_x = torch.abs(depth_map[:, 1:] - depth_map[:, :-1])
        grad_y = torch.abs(depth_map[1:, :] - depth_map[:-1, :])
        
        # Pad to match original size
        grad_x = torch.cat([grad_x, grad_x[:, -1:]], dim=1)
        grad_y = torch.cat([grad_y, grad_y[-1:, :]], dim=0)
        
        # Combine gradients
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Convert to confidence (inverse relationship)
        # Use sigmoid-like function to map to [0, 1]
        confidence = torch.exp(-gradient_magnitude * 10.0)
        
        # Handle invalid depths
        confidence[depth_map <= 0] = 0.0
        
        return confidence
    
    def _smoothness_based_confidence(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Generate confidence based on local depth smoothness."""
        # Apply Gaussian filter for smoothing
        from torch.nn.functional import conv2d
        
        # Create Gaussian kernel
        kernel_size = 5
        sigma = 1.0
        kernel_1d = torch.exp(-0.5 * ((torch.arange(kernel_size, device=depth_map.device) - kernel_size // 2) / sigma)**2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        
        # Apply smoothing
        depth_smooth = conv2d(
            depth_map.unsqueeze(0).unsqueeze(0),
            kernel_2d,
            padding=kernel_size // 2
        ).squeeze()
        
        # Compute difference from smooth version
        smoothness_error = torch.abs(depth_map - depth_smooth)
        
        # Convert to confidence
        confidence = torch.exp(-smoothness_error * 5.0)
        
        # Handle invalid depths
        confidence[depth_map <= 0] = 0.0
        
        return confidence


def demo_tsdf_fusion():
    """
    Demonstration of TSDF fusion with synthetic data.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize TSDF fusion
    tsdf = TSDFFusion(
        voxel_size=0.02,
        volume_bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        truncation_distance=0.05,
        device=device
    )
    
    print(f"Initialized TSDF volume with dimensions: {tsdf.grid_dims}")
    print(f"Total voxels: {tsdf.grid_dims.prod().item()}")
    
    # Simulate multiple depth observations
    for frame_idx in range(5):
        # Create synthetic depth map (sphere)
        H, W = 480, 640
        y, x = torch.meshgrid(
            torch.arange(H, device=device), 
            torch.arange(W, device=device), 
            indexing='ij'
        )
        center_x, center_y = W // 2, H // 2
        radius = 100
        
        # Distance from center
        dist_from_center = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        
        # Create sphere depth
        sphere_mask = dist_from_center < radius
        depth_map = torch.ones(H, W, dtype=torch.float32, device=device) * 2.0
        depth_map[sphere_mask] = 1.0 - (dist_from_center[sphere_mask] / radius) * 0.3
        
        # Add some noise to simulate real depth
        depth_map += torch.randn_like(depth_map) * 0.01
        depth_map = torch.clamp(depth_map, 0.1, 5.0)
        
        # Camera pose (slightly different for each frame)
        angle = frame_idx * 0.2
        camera_pose = torch.eye(4, dtype=torch.float32, device=device)
        camera_pose[0, 3] = 0.1 * torch.sin(torch.tensor(angle))
        camera_pose[2, 3] = -1.5 + 0.1 * torch.cos(torch.tensor(angle))
        
        # Camera intrinsics
        fx = fy = 500.0
        cx, cy = W // 2, H // 2
        camera_intrinsics = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        
        # Update TSDF volume
        tsdf.update_volume(depth_map, camera_pose, camera_intrinsics, frame_weight=1.0)
        
        print(f"Processed frame {frame_idx + 1}")
        volume_info = tsdf.get_volume_info()
        print(f"  Occupied voxels: {volume_info['num_occupied_voxels']}")
        print(f"  Mean weight: {volume_info['mean_weight']:.2f}")
    
    # Extract results
    print("\nExtracting point cloud...")
    points = tsdf.extract_pointcloud(min_weight_threshold=2.0)
    print(f"Extracted {len(points)} surface points")
    
    # Get confidence mask
    confidence_mask = tsdf.get_confidence_mask(min_weight_threshold=3.0)
    confident_voxels = confidence_mask.sum().item()
    print(f"Confident voxels: {confident_voxels}")
    
    return tsdf, points


if __name__ == "__main__":
    # Run demonstration
    tsdf_fusion, surface_points = demo_tsdf_fusion()
    print("TSDF Fusion demonstration completed!")
