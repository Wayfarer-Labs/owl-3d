import open3d as o3d
import cv2
import numpy as np
from ply_exporter import get_ply_bytes
from tqdm import tqdm

# ←– Add this block right after the imports to catch segfaults
import faulthandler, signal, sys
faulthandler.enable()  # turn on the fault handler to print a traceback on crash

def _handle_sigsegv(signum, frame):
    print("Fatal: Caught segmentation fault (SIGSEGV), exiting gracefully.", file=sys.stderr)
    sys.exit(1)

signal.signal(signal.SIGSEGV, _handle_sigsegv)

def extract_camera_params_from_proj(P):
    """
    Recover camera intrinsics and extrinsic pose from a 3×4 projection matrix P or 3×3 intrinsic matrix K.

    Args:
        P (np.ndarray or torch.Tensor): 3×4 projection matrix or 3×3 intrinsic matrix from AnyCam.

    Returns:
        intrinsics (dict): {'fx', 'fy', 'cx', 'cy'}
        extrinsic (np.ndarray): 4×4 world→camera homogeneous transform (identity if only K provided)
    """
    # Convert torch.Tensor to NumPy if necessary
    if not isinstance(P, np.ndarray):
        try:
            P = P.cpu().numpy()
        except AttributeError:
            P = np.array(P)
    
    print(f"Processing matrix with shape: {P.shape}")
    
    # Handle different input formats
    if P.shape == (3, 3):
        # Input is a 3x3 intrinsic matrix K
        print("Input is 3x3 intrinsic matrix")
        K = P.copy()
        
        # Normalize K so that K[2,2] == 1
        K = K / K[2, 2]
        
        # Extract intrinsics
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        
        intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        
        # Return identity extrinsic matrix since we don't have pose information
        extrinsic = np.eye(4, dtype=float)
        
        return intrinsics, extrinsic
        
    elif P.shape in [(3, 4), (4, 3)]:
        # Input is a 3x4 or 4x3 projection matrix
        print("Input is projection matrix")
        
        # Ensure P is 3x4
        if P.shape == (4, 3):
            P = P.T
        
        # Decompose P into cameraMatrix (K), rotation R, and translation t (homogeneous)
        K, R, t_hom, _, _, _, _ = cv2.decomposeProjectionMatrix(P)

        # Normalize K so that K[2,2] == 1
        K = K / K[2, 2]

        # Extract intrinsics
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

        # t_hom is 4×1 in homogeneous coords [tx, ty, tz, w]ᵀ
        # Convert to 3×1 translation vector by dividing by w
        t = (t_hom[:3] / t_hom[3]).reshape(3,)

        # Build 4×4 world-to-camera extrinsic: [ R | t; 0 0 0 1 ]
        world_to_cam = np.eye(4, dtype=float)
        world_to_cam[:3, :3] = R
        world_to_cam[:3,  3] = t
        
        # For TSDF integration, we need camera-to-world transform
        # Invert the world-to-camera matrix
        try:
            cam_to_world = np.linalg.inv(world_to_cam)
            print("Inverted world-to-camera to get camera-to-world transform")
        except np.linalg.LinAlgError:
            print("Warning: Could not invert pose matrix, using identity")
            cam_to_world = np.eye(4, dtype=float)

        return intrinsics, cam_to_world
    
    else:
        raise ValueError(f"Expected 3x3 intrinsic matrix, 3x4 or 4x3 projection matrix, got {P.shape}")


def process_fusion_batch(rgbs, depths, trajectory, projection_matrix):
    """
    Process a batch of RGB and depth frames using TSDF fusion.
    Assumes all inputs from process_frames_with_anycam are properly formatted.
    
    Args:
        rgbs: List of RGB frames (numpy arrays, already resized to match depths)
        depths: List of depth maps (numpy arrays in meters, float32)
        trajectory: List of 4x4 camera poses (numpy arrays)
        projection_matrix: 3x3 or 3x4 projection matrix for intrinsics
    
    Returns:
        bytes: PLY file content as bytes
    """
    # Debug prints to understand data structure
    print(f"DEBUG: rgbs type: {type(rgbs)}")
    print(f"       shape: {getattr(rgbs, 'shape', 'N/A')}")
    print(f"DEBUG: depths type: {type(depths)}")
    print(f"       shape: {getattr(depths, 'shape', 'N/A')}")
    print(f"DEBUG: trajectory type: {type(trajectory)}")
    print(f"       shape: {getattr(trajectory, 'shape', 'N/A')}")
    print(f"DEBUG: projection_matrix type: {type(projection_matrix)}")
    print(f"       shape: {getattr(projection_matrix, 'shape', 'N/A')}")
    
    # Create TSDF volume
    # volume = o3d.pipelines.integration.UniformTSDFVolume(
    #     length=4./64,
    #     resolution=64,  # Adjust resolution as needed
    #     sdf_trunc=0.1,
    #     color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    # )
    volume = o3d.t.geometry.VoxelBlockGrid(attr_names = ["tsdf", "weight"],
        attr_dtypes     = [o3d.core.Dtype.Float32, o3d.core.Dtype.UInt16],
        attr_channels   = [o3d.core.SizeVector([1]),   o3d.core.SizeVector([1])],
        voxel_size      = .10 / 200,      # 0.005 m per voxel
        block_resolution= 16,             # each block is 16×16×16 voxels
        block_count     = 5000,           # max number of blocks to allocate
        device          = o3d.core.Device("CUDA:0")
    )
    # Create a TSDF voxel grid on GPU
    # volume = o3d.t.geometry.VoxelBlockGrid(
    #     # {'tsdf': o3d.core.Dtype.Float32,
    #     # 'weight': o3d.core.Dtype.UInt16,
    #     # 'color':  o3d.core.Dtype.UInt16},
    #     voxel_size=0.005,
    #     sdf_trunc=0.04,
    #     block_resolution=16,
    #     block_count=10000,
    #     device=o3d.core.Device("CUDA:0")
    # )
    # volume = o3d.pipelines.integration.ScalableTSDFVolume(
    #     # voxel_length=0.4,
    #     # volume_unit_resolution=64,
    #     # sdf_trunc=0.1,
    #     color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    # )

    # Extract base intrinsics from projection matrix
    intrinsics_dict, _ = extract_camera_params_from_proj(projection_matrix)
    # print(projection_matrix)
    # Handle different depths data structures
    # Based on debug: depths shape is (50, 1, 336, 336)
    # Shape is (frames, 1, H, W) - squeeze out the singleton dimension
    if depths.shape[1] == 1:
        depth_data = depths[:, 0, :, :]  # Remove singleton dimension
        first_depth = depth_data[0]
    else:
        # Shape is (batch, frames, H, W)
        depth_data = depths[0]  # Take first batch
        first_depth = depth_data[0]
    
    # Get image dimensions from first depth frame
    height, width = first_depth.shape
    print(f"DEBUG: Using depth dimensions: {height}x{width}")
    print(f"DEBUG: RGB original shape: {rgbs[0].shape}")
    
    # The intrinsics might be for the original RGB size, we may need to
    # scale them to match the depth image size
    rgb_height, rgb_width = rgbs[0].shape[:2]
    scale_x = width / rgb_width
    scale_y = height / rgb_height
    
    # Scale intrinsics to match depth image size
    scaled_intrinsics = {
        'fx': intrinsics_dict['fx'] * scale_x,
        'fy': intrinsics_dict['fy'] * scale_y,
        'cx': intrinsics_dict['cx'] * scale_x,
        'cy': intrinsics_dict['cy'] * scale_y
    }
    
    print(f"DEBUG: Scaling factors: x={scale_x:.3f}, y={scale_y:.3f}")
    print("DEBUG: Scaled intrinsics:")
    print(f"       fx={scaled_intrinsics['fx']}, fy={scaled_intrinsics['fy']}")
    print(f"       cx={scaled_intrinsics['cx']}, cy={scaled_intrinsics['cy']}")
    
    # Create Open3D intrinsics with scaled values
    intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
        width, height,
        scaled_intrinsics['fx'], scaled_intrinsics['fy'],
        scaled_intrinsics['cx'], scaled_intrinsics['cy']
    )
    print(trajectory)
    # Process each frame
    print(f"DEBUG: Starting to process {len(rgbs)} frames")
    for i, (rgb, depth, pose) in enumerate(
        tqdm(zip(rgbs, depth_data, trajectory))
    ):
        if i < 3:  # Only debug first 3 frames to avoid spam
            print(f"DEBUG: Processing frame {i}")
            print(f"       rgb shape: {rgb.shape}, depth shape: {depth.shape}")

        # Resize RGB to match depth dimensions
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        if i < 3:
            print(f"DEBUG: Resized RGB to: {rgb.shape}")
        # Ensure RGB is uint8 and depth is float32
        rgb = rgb.astype(np.uint8)
        depth = depth.astype(np.float32)
        
        try:
            # Create Open3D images
            rgb_o3d = o3d.geometry.Image(rgb)
            depth_o3d = o3d.geometry.Image(depth)
            print(f"DEBUG: Created Open3D images for frame {i}")
            # Create RGBD image
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
            )
            print(f"DEBUG: Created rgbd image for frame {i}")
            # Integrate into TSDF volume
            # print(intrinsics_o3d)
            # print(pose)
            volume.integrate(rgbd, intrinsics_o3d, np.linalg.inv(pose))

        except (RuntimeError, ValueError, TypeError) as e:
            print(f"Error processing frame {i}: {e}")
            continue

    # Extract point cloud
    pcd = volume.to_legacy().extract_point_cloud()
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Convert to PLY bytes
    if len(colors) > 0 and len(points) > 0:
        colors_255 = (colors * 255).astype(np.uint8)
        point_cloud_data = np.hstack([points, colors_255])
        return get_ply_bytes(point_cloud_data, colors=True)
    elif len(points) > 0:
        return get_ply_bytes(points, colors=False)
    else:
        return b""
