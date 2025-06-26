"""
Example: TSDF Fusion with CUT3R for Online Reconstruction

This script demonstrates how to integrate TSDF fusion with CUT3R's autoregressive
generation for online 3D reconstruction with dynamic filtering.

Usage:
    python tsdf_demo_cut3r.py --input_dir examples/001 --output_dir output/tsdf_reconstruction
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

# Import CUT3R modules (assuming they're available)
# try:
# from demo import load_model, load_images
from src.dust3r.model import load_model
# from src.dust3r.utils import load_model
from src.dust3r.utils.image import load_images
from src.dust3r.inference import inference
from src.dust3r.utils.device import to_numpy
DUST3R_AVAILABLE = True
# except ImportError:
#     print("Warning: CUT3R modules not found. Running in simulation mode.")
#     DUST3R_AVAILABLE = False

from cut3r_tsdf_integration import CUT3RTSDFIntegration


def parse_args():
    parser = argparse.ArgumentParser(description='TSDF Fusion with CUT3R')
    parser.add_argument('--input_dir', type=str, default='examples/001', 
                       help='Input directory with images')
    parser.add_argument('--output_dir', type=str, default='output/tsdf_reconstruction',
                       help='Output directory for reconstruction')
    parser.add_argument('--model_path', type=str, default='src/cut3r_224_linear_4.pth',
                       help='Path to CUT3R model')
    parser.add_argument('--voxel_size', type=float, default=0.1,
                       help='Voxel size for TSDF grid')
    parser.add_argument('--volume_bounds', nargs=6, type=float,
                       default=[-5.0, -4.0, -6.0, 4.0, 3.0, 6.0],
                       help='Volume bounds: x_min y_min z_min x_max y_max z_max')
    parser.add_argument('--truncation_distance', type=float, default=.1,
                       help='TSDF truncation distance')
    parser.add_argument('--confidence_threshold', type=float, default=1.0,
                       help='Minimum confidence threshold (CUT3R typically ranges 0-10, recommend 1.0-2.0)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--simulate', action='store_true',
                       help='Run simulation mode without real CUT3R model')
    parser.add_argument('--show_confidence_stats', action='store_true',
                       help='Show detailed confidence statistics to help choose threshold')
    return parser.parse_args()


class OnlineTSDFReconstruction:
    """
    Online TSDF reconstruction using CUT3R for autoregressive generation.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = args.device if torch.cuda.is_available() else 'cpu'
        
        # Initialize TSDF integration
        self.tsdf_integration = CUT3RTSDFIntegration(
            voxel_size=args.voxel_size,
            volume_bounds=tuple(args.volume_bounds),
            truncation_distance=args.truncation_distance,
            confidence_threshold=args.confidence_threshold,
            device=self.device
        )
        
        # Load CUT3R model if available
        if DUST3R_AVAILABLE and not args.simulate:
            self.model = load_model(args.model_path, device=self.device)
            self.cut3r_available = True
        else:
            self.model = None
            self.cut3r_available = False
            print("Running in simulation mode")
    
    def load_image_sequence(self, input_dir: str):
        """Load sequence of images from directory."""
        input_path = Path(input_dir)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend([str(f) for f in input_path.glob(f'*{ext}')])
            image_files.extend([str(f) for f in input_path.glob(f'*{ext.upper()}')])
        
        image_files.sort()
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {input_dir}")
        
        print(f"Found {len(image_files)} images")
        return image_files
    
    def run_cut3r_inference(self, image_files):
        """Run CUT3R inference on image sequence."""
        if not self.cut3r_available:
            print("CUT3R model not available, running in simulation mode.")
            return self.simulate_cut3r_output(image_files)
        
        # Load images for CUT3R
        images = load_images(image_files, size=512)
        
        # Convert images to proper view format expected by CUT3R model
        views = []
        for i, img_data in enumerate(images):
            result = tuple(img_data["true_shape"].tolist())
            print(f"Processing image {i+1}/{len(images)}: {result}")
            view = {
                "img": img_data["img"],
                "ray_map": torch.full(
                    (
                        img_data["img"].shape[0],
                        6,
                        img_data["img"].shape[-2],
                        img_data["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(img_data["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            views.append(view)
        
        # Run inference
        output, state_args = inference(views, self.model, self.device, verbose=True)
        
        # Convert output to view data format
        view_sequence = []
        
        preds = output['pred']
        views_out = output['views']
        
        for i in range(len(images)):
            # Extract 3D points from predictions - try different keys
            pts3d = None
            if 'pts3d' in preds[i]:
                pts3d = to_numpy(preds[i]['pts3d'])
                print(f"DEBUG: Frame {i} pts3d shape: {pts3d.shape}")
            elif 'pts3d_in_self_view' in preds[i]:
                pts3d = to_numpy(preds[i]['pts3d_in_self_view'])
                print(f"DEBUG: Frame {i} pts3d_in_self_view shape: {pts3d.shape}")
            else:
                # Generate dummy 3D points if not available
                H, W = tuple(images[i]['true_shape'].flatten())
                pts3d = np.zeros((H, W, 3), dtype=np.float32)
                print(f"DEBUG: Frame {i} pts3d dummy shape: {pts3d.shape}")
                print(f"DEBUG: Frame {i} available keys in preds: {list(preds[i].keys())}")
            
            # Extract confidence and debug its shape
            conf = None
            if 'conf' in preds[i]:
                conf = to_numpy(preds[i]['conf'])
                print(f"DEBUG: Frame {i} conf shape: {conf.shape}")
            
            # Extract camera pose - handle different formats
            if 'camera_pose' in preds[i]:
                camera_pose_raw = to_numpy(preds[i]['camera_pose'])
                print(f"DEBUG: Frame {i} camera_pose_raw shape: {camera_pose_raw.shape}")
                
                if camera_pose_raw.shape == (1, 7):  # Quaternion + translation format
                    # Convert from quaternion + translation to 4x4 matrix
                    camera_pose = self._convert_pose_from_quat_trans(camera_pose_raw[0])
                elif camera_pose_raw.shape == (1, 4, 4):
                    camera_pose = camera_pose_raw[0]
                else:
                    camera_pose = np.eye(4, dtype=np.float32)
                    print(f"DEBUG: Frame {i} unknown camera_pose format, using identity")
            else:
                camera_pose = np.eye(4, dtype=np.float32)
                print(f"DEBUG: Frame {i} no camera_pose, using identity")
            
            print(f"DEBUG: Frame {i} final camera_pose shape: {camera_pose.shape}")
            
            # Use default camera intrinsics since model doesn't provide them
            camera_intrinsics = self._get_default_intrinsics()
            
            # Ensure pts3d has the right shape (H, W, 3)
            if pts3d is not None:
                if pts3d.ndim == 4 and pts3d.shape[0] == 1:  # Remove batch dimension
                    pts3d = pts3d.squeeze(0)
                    print(f"DEBUG: Frame {i} pts3d after squeeze: {pts3d.shape}")
            
            # Ensure conf has the right shape (H, W)
            if conf is not None:
                if conf.ndim == 3 and conf.shape[0] == 1:  # Remove batch dimension
                    conf = conf.squeeze(0)
                    print(f"DEBUG: Frame {i} conf after squeeze: {conf.shape}")
            
            view_data = {
                'pts3d': pts3d,
                'camera_pose': camera_pose,
                'camera_intrinsics': camera_intrinsics,
                'conf': conf,
                'img': to_numpy(images[i]['img'][0]).transpose(1, 2, 0)  # Convert from CHW to HWC
            }
            view_sequence.append(view_data)
        
        return view_sequence
    
    def simulate_cut3r_output(self, image_files):
        """Simulate CUT3R output for demonstration."""
        print("Simulating CUT3R output...")
        
        view_sequence = []
        
        for i, image_file in enumerate(image_files):
            # Load image to get dimensions
            img = cv2.imread(str(image_file))
            if img is None:
                continue
                
            H, W = img.shape[:2]
            
            # Simulate 3D points
            y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            
            # Create synthetic depth with some scene structure
            depth = 2.0 + 0.5 * np.sin(x / 50.0) * np.cos(y / 50.0)
            depth += 0.3 * np.random.randn(H, W)  # Add noise
            depth = np.clip(depth, 0.5, 5.0)
            
            # Create 3D points in camera coordinates
            fx = fy = max(H, W) * 0.8  # Typical focal length
            cx, cy = W // 2, H // 2
            
            pts3d = np.zeros((H, W, 3), dtype=np.float32)
            pts3d[..., 0] = (x - cx) * depth / fx
            pts3d[..., 1] = (y - cy) * depth / fy
            pts3d[..., 2] = depth
            
            # Simulate camera pose (orbiting motion)
            angle = i * 0.2
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[0, 3] = 1.5 * np.sin(angle)
            camera_pose[1, 3] = 0.5 * np.sin(angle * 2)
            camera_pose[2, 3] = 3.0 + 1.0 * np.cos(angle)
            
            # Add rotation
            camera_pose[0, 0] = np.cos(angle * 0.5)
            camera_pose[0, 2] = np.sin(angle * 0.5)
            camera_pose[2, 0] = -np.sin(angle * 0.5)
            camera_pose[2, 2] = np.cos(angle * 0.5)
            
            # Camera intrinsics
            camera_intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Simulate confidence (higher in center, lower at edges)
            center_dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            max_dist = np.sqrt(cx**2 + cy**2)
            confidence = np.exp(-2 * center_dist / max_dist)
            
            # Add some dynamic regions (low confidence areas)
            if i % 3 == 0:  # Every third frame has "dynamic" regions
                dynamic_mask = (x > W * 0.7) & (y > H * 0.7)
                confidence[dynamic_mask] *= 0.3  # Reduce confidence in dynamic regions
            
            view_data = {
                'pts3d': pts3d,
                'camera_pose': camera_pose,
                'camera_intrinsics': camera_intrinsics,
                'conf': confidence,
                'img': img
            }
            
            view_sequence.append(view_data)
        
        return view_sequence
    
    def _get_default_intrinsics(self, image_size=(224, 224)):
        """Get default camera intrinsics."""
        H, W = image_size
        fx = fy = max(H, W) * 0.8
        cx, cy = W // 2, H // 2
        
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def _convert_pose_from_quat_trans(self, pose_7d):
        """Convert 7D pose (quaternion + translation) to 4x4 matrix."""
        # pose_7d format: [qw, qx, qy, qz, tx, ty, tz] or [tx, ty, tz, qx, qy, qz, qw]
        # We need to figure out which format it is
        
        # Try common format: [tx, ty, tz, qx, qy, qz, qw]
        if len(pose_7d) == 7:
            tx, ty, tz = pose_7d[:3]
            qx, qy, qz, qw = pose_7d[3:]
        else:
            # Fallback to identity
            return np.eye(4, dtype=np.float32)
        
        # Convert quaternion to rotation matrix
        # Normalize quaternion
        q_norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if q_norm > 0:
            qx, qy, qz, qw = qx/q_norm, qy/q_norm, qz/q_norm, qw/q_norm
        
        # Quaternion to rotation matrix
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ], dtype=np.float32)
        
        # Create 4x4 transformation matrix
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        
        return T
    
    def process_sequence(self, image_files):
        """Process image sequence with online TSDF fusion."""
        print("Running CUT3R inference...")
        view_sequence = self.run_cut3r_inference(image_files)
        
        print("Processing views with TSDF fusion...")
        reconstruction_info = self.tsdf_integration.process_cut3r_sequence(
            view_sequence,
            progressive_weights=True,
            min_confidence=self.args.confidence_threshold
        )
        
        return reconstruction_info
    
    def save_results(self, output_dir: str):
        """Save reconstruction results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving results to {output_dir}")
        
        # Save filtered point cloud
        points = self.tsdf_integration.extract_filtered_pointcloud(
            min_weight_threshold=2.0
        )
        
        if len(points) > 0:
            np.save(output_path / 'filtered_pointcloud.npy', points.cpu().numpy())
            print(f"Saved {len(points)} filtered points")
            
            # Save as PLY for visualization
            self.save_pointcloud_ply(
                points.cpu().numpy(),
                str(output_path / 'filtered_pointcloud.ply')
            )
        
        # Try to save mesh
        try:
            self.tsdf_integration.save_reconstruction(
                str(output_path / 'mesh.ply'),
                format='ply'
            )
            print("Saved mesh reconstruction")
        except Exception as e:
            print(f"Could not save mesh: {e}")
        
        # Save reconstruction summary
        summary = self.tsdf_integration.get_reconstruction_summary()
        
        import json
        with open(output_path / 'reconstruction_summary.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(v) for v in obj]
                return obj
            
            json.dump(convert_types(summary), f, indent=2)
        
        print("Saved reconstruction summary")
        
        # Save TSDF volume for debugging
        volume_data = {
            'tsdf_volume': self.tsdf_integration.tsdf.tsdf_volume.cpu().numpy(),
            'weight_volume': self.tsdf_integration.tsdf.weight_volume.cpu().numpy(),
            'volume_bounds': self.tsdf_integration.tsdf.volume_bounds.cpu().numpy(),
            'voxel_size': self.tsdf_integration.tsdf.voxel_size
        }
        np.savez(output_path / 'tsdf_volume.npz', **volume_data)
        print("Saved TSDF volume data")
    
    def save_pointcloud_ply(self, points: np.ndarray, filepath: str):
        """Save point cloud as PLY file."""
        with open(filepath, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(points)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('end_header\n')
            
            for point in points:
                f.write(f'{point[0]} {point[1]} {point[2]}\n')
    
    def run(self):
        """Run the complete reconstruction pipeline."""
        print("Starting Online TSDF Reconstruction with CUT3R")
        print(f"Device: {self.device}")
        print(f"Voxel size: {self.args.voxel_size}")
        print(f"Volume bounds: {self.args.volume_bounds}")
        
        # Load images
        image_files = self.load_image_sequence(self.args.input_dir)
        
        # Process sequence
        reconstruction_info = self.process_sequence(image_files)
        
        # Print results
        print("\nReconstruction Results:")
        print(f"Frames processed: {reconstruction_info['frames_processed']}")
        print(f"Frames skipped: {reconstruction_info['frames_skipped']}")
        print(f"Volume info: {reconstruction_info['volume_info']}")
        
        if reconstruction_info['confidence_stats']:
            conf_stats = reconstruction_info['confidence_stats']
            print(f"Confidence stats - Mean: {np.mean(conf_stats):.3f}, "
                  f"Std: {np.std(conf_stats):.3f}")
        
        # Save results
        self.save_results(self.args.output_dir)
        
        print(f"\nReconstruction completed! Results saved to {self.args.output_dir}")
        
        return self.tsdf_integration


def main():
    args = parse_args()
    
    # Create reconstruction system
    reconstruction = OnlineTSDFReconstruction(args)
    
    # Run reconstruction
    tsdf_integration = reconstruction.run()
    
    # Additional analysis
    print("\nAdditional Analysis:")
    summary = tsdf_integration.get_reconstruction_summary()
    print(f"Confident voxels ratio: {summary['confidence_ratio']:.3f}")
    print(f"Total reconstruction volume: {np.prod(summary['volume_info']['grid_dims']) * args.voxel_size**3:.3f} m³")
    
    # Extract and analyze filtered points
    filtered_points = tsdf_integration.extract_filtered_pointcloud(min_weight_threshold=3.0)
    if len(filtered_points) > 0:
        points_np = filtered_points.cpu().numpy()
        print(f"High-confidence points: {len(points_np)}")
        print(f"Point cloud bounds:")
        print(f"  X: [{points_np[:, 0].min():.3f}, {points_np[:, 0].max():.3f}]")
        print(f"  Y: [{points_np[:, 1].min():.3f}, {points_np[:, 1].max():.3f}]")
        print(f"  Z: [{points_np[:, 2].min():.3f}, {points_np[:, 2].max():.3f}]")


if __name__ == "__main__":
    main()
