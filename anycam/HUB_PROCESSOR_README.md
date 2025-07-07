# AnyCam Hub Processor

A simplified AnyCam processor using the `torch.hub` interface. This approach is much simpler than the full AnyCam Docker setup and uses the official torch.hub model.

## Files

- `anycam_hub_processor.py` - Main Python script using torch.hub interface
- `process_video_hub_docker.sh` - Bash script to run the processor inside Docker

## Features

- Uses `torch.hub.load('Brummi/anycam', 'AnyCam', ...)` for easy model loading
- Extracts frames from video using ffmpeg
- Processes frames as (H,W,3) arrays in [0,1] range
- Supports bundle adjustment refinement
- Saves trajectory, depths, uncertainties, and projection matrix as PyTorch tensors

## Usage

### Basic Usage
```bash
./process_video_hub_docker.sh video.mp4 ./outputs
```

### With Bundle Adjustment
```bash
./process_video_hub_docker.sh video.mp4 ./outputs --ba_refinement
```

### Limit Frames and Resize
```bash
./process_video_hub_docker.sh video.mp4 ./outputs --max_frames 100 --resize_height 480
```

### Custom Container
```bash
./process_video_hub_docker.sh video.mp4 ./outputs --container my-anycam:latest
```

## Docker Setup

The script assumes you have a Docker container with:
- Python 3.x
- PyTorch with CUDA support (optional)
- ffmpeg
- Required Python packages: numpy, torch, PIL, tqdm

## Output Files

The processor generates:
- `{video_name}_trajectory.pt` - Camera poses
- `{video_name}_depths.pt` - Depth maps  
- `{video_name}_uncertainties.pt` - Uncertainty maps
- `{video_name}_projection_matrix.pt` - Camera intrinsics
- `{video_name}_metadata.json` - Processing metadata

## Requirements

- Docker with GPU support (recommended)
- Input video file
- Sufficient disk space for output tensors

## Examples

```bash
# Process short video with bundle adjustment
./process_video_hub_docker.sh sample.mp4 ./results --ba_refinement --max_frames 50

# Process and resize for faster processing  
./process_video_hub_docker.sh large_video.mp4 ./results --resize_height 360

# Full processing with all options
./process_video_hub_docker.sh video.mp4 ./results \
    --ba_refinement \
    --max_frames 200 \
    --resize_height 480 \
    --container anycam:v2
```

## Advantages over Full AnyCam Pipeline

1. **Simpler**: Uses torch.hub interface instead of complex Docker setup
2. **Direct**: No intermediate video creation/processing steps
3. **Flexible**: Easy to modify for different input/output formats
4. **Efficient**: Processes frames directly in memory
5. **Portable**: Works with any Docker container that has PyTorch

## Script Arguments

### Python Script (`anycam_hub_processor.py`)
- `--input_video` - Path to input video file (required)
- `--output_dir` - Output directory for results (required)  
- `--ba_refinement` - Enable bundle adjustment refinement
- `--max_frames` - Maximum number of frames to process
- `--resize_height` - Resize frames to this height (maintains aspect ratio)

### Bash Script (`process_video_hub_docker.sh`)
- `INPUT_VIDEO` - Path to input video file (positional, required)
- `OUTPUT_DIR` - Directory for output files (positional, required)
- `--ba_refinement` - Enable bundle adjustment refinement
- `--max_frames N` - Process only first N frames
- `--resize_height N` - Resize frames to height N
- `--container NAME` - Docker container name (default: anycam:latest)
- `--help` - Show help message
