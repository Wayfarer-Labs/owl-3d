# AnyCam Docker Setup

This Docker setup provides a containerized environment for running [AnyCam](https://github.com/Brummi/anycam) - a system for learning to recover camera poses and intrinsics from casual videos.

## Prerequisites

- Docker installed on your system
- NVIDIA Docker runtime (for GPU support)
- NVIDIA GPU with CUDA support (recommended)

## Quick Start

### 1. Build the Docker Image

```bash
./build_docker.sh
```

Or manually:
```bash
docker build -t anycam:latest .
```

### 2. Run the Container

```bash
./run_docker.sh
```

Or manually:
```bash
docker run --gpus all -it --rm \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    -p 9090:9090 \
    anycam:latest
```

## Usage

### Basic Demo

Once inside the container, you can run the AnyCam demo:

```bash
# Process a video file
python anycam/scripts/anycam_demo.py \
    ++input_path=/workspace/data/your_video.mp4 \
    ++model_path=pretrained_models/anycam_seq8 \
    ++output_path=/workspace/outputs \
    ++visualize=true
```

### Feed-forward Only (Faster)

```bash
python anycam/scripts/anycam_demo.py \
    ++input_path=/workspace/data/your_video.mp4 \
    ++model_path=pretrained_models/anycam_seq8 \
    ++ba_refinement=false \
    ++output_path=/workspace/outputs \
    ++visualize=true
```

### Export to COLMAP Format

```bash
python anycam/scripts/anycam_demo.py \
    ++input_path=/workspace/data/your_video.mp4 \
    ++model_path=pretrained_models/anycam_seq8 \
    ++export_colmap=true \
    ++output_path=/workspace/outputs
```

## Directory Structure

- `/workspace/data/` - Mount point for input videos (maps to `./data/` on host)
- `/workspace/outputs/` - Mount point for output results (maps to `./outputs/` on host)
- `/workspace/anycam/` - AnyCam source code
- `/workspace/anycam/pretrained_models/` - Pre-downloaded model weights

## Visualization

### Local Visualization

If you're running on a local machine with a display, the visualization will work directly.

### Remote Server Setup

If you're running on a remote server:

1. Start the rerun web viewer inside the container:
```bash
rerun --serve-web
```

2. Forward port 9090 from your remote server to your local machine:
```bash
ssh -L 9090:localhost:9090 your-server
```

3. Run the demo with web connection:
```bash
python anycam/scripts/anycam_demo.py \
    ++input_path=/workspace/data/your_video.mp4 \
    ++model_path=pretrained_models/anycam_seq8 \
    ++rerun_mode=connect \
    ++visualize=true
```

4. Open your browser to: `http://localhost:9090/?url=ws://localhost:9877`

## PyTorch Hub Usage

You can also use AnyCam programmatically with PyTorch Hub:

```python
import torch
import numpy as np

# Load the model
anycam = torch.hub.load('Brummi/anycam', 'AnyCam', 
                        version="1.0", 
                        training_variant="seq8", 
                        pretrained=True)

# Process frames (list of numpy arrays with shape (H,W,3) and values in [0,1])
results = anycam.process_video(frames, ba_refinement=True)

# Access results
trajectory = results["trajectory"]              # Camera poses
depths = results["depths"]                      # Depth maps
uncertainties = results["uncertainties"]        # Uncertainty maps
projection_matrix = results["projection_matrix"] # Camera intrinsics
```

## Troubleshooting

### GPU Issues

If you encounter GPU-related errors:

1. Check if NVIDIA Docker runtime is installed:
```bash
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

2. If no GPU is available, the container will run on CPU (much slower).

### Memory Issues

- Large videos may require significant GPU memory
- Consider reducing video resolution or frame rate
- Use `++fps=10` to subsample high frame rate videos

### Build Issues

If the Docker build fails:

1. Check your internet connection (downloads pretrained models)
2. Ensure you have sufficient disk space (image is ~10GB)
3. Try building with `--no-cache` flag

## What's Included

- Ubuntu 22.04 with CUDA 12.4 support
- Python 3.11 environment
- PyTorch 2.5.1 with CUDA support
- All required dependencies from requirements.txt
- AnyCam source code (cloned from GitHub)
- Pretrained model weights (anycam_seq8)
- Rerun.io for visualization

## Citation

If you use this Docker setup with AnyCam, please cite the original paper:

```bibtex
@inproceedings{wimbauer2025anycam,
  title={AnyCam: Learning to Recover Camera Poses and Intrinsics from Casual Videos},
  author={Wimbauer, Felix and Chen, Weirong and Muhle, Dominik and Rupprecht, Christian and Cremers, Daniel},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
