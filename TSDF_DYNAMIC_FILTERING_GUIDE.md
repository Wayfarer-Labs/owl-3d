# TSDF Fusion for Dynamic Element Filtering

This guide explains how to use TSDF fusion to filter out dynamic parts of point clouds over time, keeping only the static structure of a scene.

## Core Concept

TSDF (Truncated Signed Distance Function) fusion naturally filters dynamic elements through **weighted temporal averaging**:

- **Static objects**: Appear consistently → High weights → Preserved in final reconstruction
- **Dynamic objects**: Appear inconsistently → Low weights → Filtered out

## Key Parameters for Dynamic Filtering

### 1. Weight Threshold (`min_weight_threshold`)
Controls how aggressively dynamic elements are filtered:

```python
# Conservative filtering - keeps more points, some dynamics may remain
points = tsdf.extract_pointcloud(min_weight_threshold=1.0)

# Balanced filtering - good for most applications  
points = tsdf.extract_pointcloud(min_weight_threshold=3.0)

# Aggressive filtering - only very stable static elements
points = tsdf.extract_pointcloud(min_weight_threshold=5.0)
```

### 2. Confidence Threshold (`confidence_threshold`)
Filters input data quality:

```python
tsdf_integration = CUT3RTSDFIntegration(
    confidence_threshold=0.5,  # Reject low-confidence depth measurements
    # ... other parameters
)
```

### 3. Progressive Weighting
Later frames get higher weights (more stable observations):

```python
reconstruction_info = tsdf_integration.process_cut3r_sequence(
    view_sequence,
    progressive_weights=True,  # Enable progressive weighting
    min_confidence=0.3
)
```

## Step-by-Step Usage

### 1. Initialize TSDF Integration

```python
from cut3r_tsdf_integration import CUT3RTSDFIntegration

tsdf_integration = CUT3RTSDFIntegration(
    voxel_size=0.02,           # Smaller = more detail, more memory
    volume_bounds=(-2, -2, -2, 2, 2, 2),  # Adjust to your scene size
    truncation_distance=0.05,   # Distance from surface to consider
    confidence_threshold=0.3,   # Minimum confidence for depth values
    device='cuda'
)
```

### 2. Process Multiple Frames

```python
# For each frame in your sequence
for frame_data in your_frames:
    # frame_data should contain:
    # - 'pts3d': 3D points (H, W, 3) in camera coordinates  
    # - 'camera_pose': Camera pose matrix (4, 4)
    # - 'camera_intrinsics': Camera intrinsics (3, 3)
    # - 'conf': Confidence map (H, W) - optional but recommended
    
    tsdf_integration.update_with_cut3r_output(
        frame_data,
        frame_weight=1.0,  # Can adjust per frame
        apply_confidence_filtering=True
    )
```

### 3. Extract Filtered Point Cloud

```python
# Extract point cloud with different filtering levels
filtered_points = tsdf_integration.extract_filtered_pointcloud(
    min_weight_threshold=3.0,  # Key parameter for dynamic filtering
    spatial_filtering=True,    # Remove isolated points
    cluster_threshold=0.1      # Distance for spatial clustering
)
```

## Advanced Filtering Strategies

### 1. Multi-Level Filtering

```python
# Extract multiple versions with different aggressiveness
conservative = tsdf.extract_pointcloud(min_weight_threshold=1.0)
balanced = tsdf.extract_pointcloud(min_weight_threshold=3.0) 
aggressive = tsdf.extract_pointcloud(min_weight_threshold=5.0)

print(f"Conservative: {len(conservative)} points")
print(f"Balanced: {len(balanced)} points") 
print(f"Aggressive: {len(aggressive)} points")
```

### 2. Confidence Analysis

```python
# Analyze which areas are reliable
confidence_mask = tsdf.tsdf.get_confidence_mask(min_weight_threshold=3.0)
reliable_percentage = (confidence_mask.sum() / confidence_mask.numel()) * 100
print(f"Reliable voxels: {reliable_percentage:.1f}%")
```

### 3. Temporal Consistency Check

```python
# Check frame history for processing quality
frame_history = tsdf_integration.frame_history
avg_confidence = np.mean([f['confidence'] for f in frame_history])
print(f"Average frame confidence: {avg_confidence:.2f}")
```

## Practical Tips

### For Better Dynamic Filtering:

1. **Use more frames**: More observations improve static/dynamic discrimination
2. **Ensure camera motion**: Static camera can't distinguish static vs dynamic well
3. **Adjust confidence thresholds**: Lower for noisy data, higher for clean data
4. **Progressive weighting**: Later frames in sequence are often more stable

### Parameter Tuning Guidelines:

- **Voxel size**: Start with 0.02-0.05m, adjust based on scene scale
- **Weight threshold**: Start with 3.0, increase for more aggressive filtering
- **Confidence threshold**: Start with 0.3-0.5, adjust based on your data quality
- **Volume bounds**: Set to tightly bound your scene for efficiency

### Common Issues:

- **Over-filtering**: If losing too much geometry, lower weight threshold
- **Under-filtering**: If dynamics remain, increase weight threshold or improve confidence maps
- **Memory issues**: Reduce voxel grid size or volume bounds
- **Poor filtering**: Ensure sufficient camera motion and frame count

## Example Output

After processing a sequence with both static structure and moving objects:

```
Frames processed: 8/10
Volume info: {'num_occupied_voxels': 45231, 'mean_weight': 4.2}

Point cloud extraction:
Low threshold (1.0): 12,450 points
Medium threshold (3.0): 8,320 points  
High threshold (5.0): 6,180 points

Reliable voxels (weight >= 3.0): 73.2%
```

The higher weight thresholds successfully filter out dynamic elements while preserving the static scene structure.
