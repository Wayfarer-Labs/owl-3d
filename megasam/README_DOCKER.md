# MegaSAM Docker Setup

This Docker image provides a complete environment for running MegaSAM (Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos).

## Building the Docker Image

```bash
docker build -t megasam .
```

## Running the Container

```bash
# Run with GPU support
docker run --gpus all -it --rm megasam

# Run with mounted volume for data persistence
docker run --gpus all -it --rm -v /path/to/your/data:/mega-sam/data megasam

# Run with X11 forwarding for visualization (Linux)
docker run --gpus all -it --rm \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    --network host \
    megasam
```

## Environment Details

- Base: CUDA 11.8 with Ubuntu 22.04
- Python 3.10
- PyTorch 2.0.1 with CUDA 11.8 support
- Conda environment named `mega_sam`
- All dependencies from `environment.yml` installed
- XFormers for UniDepth model
- Camera tracking extensions compiled

## Required Checkpoints

After running the container, you'll need to download the pretrained checkpoints:

1. **DepthAnything checkpoint**: Download to `Depth-Anything/checkpoints/depth_anything_vitl14.pth`
   ```bash
   wget -O Depth-Anything/checkpoints/depth_anything_vitl14.pth \
       https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth
   ```

2. **RAFT checkpoint**: Download to `cvd_opt/raft-things.pth`
   - Download from: https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT

3. **MegaSAM checkpoint**: Download to `checkpoints/megasam_final.pth`
   - This should be available from the MegaSAM authors

## Usage

Once inside the container with checkpoints downloaded:

### For Sintel dataset:
```bash
# Precompute mono-depth
./mono_depth_scripts/run_mono-depth_sintel.sh

# Run camera tracking
./tools/evaluate_sintel.sh

# Run consistent video depth optimization
./cvd_opt/cvd_opt_sintel.sh

# Evaluate results
python ./evaluations_poses/evaluate_sintel.py
python ./evaluations_depth/evaluate_depth_ours_sintel.py
```

### For DyCheck dataset:
```bash
# Precompute mono-depth
./mono_depth_scripts/run_mono-depth_dycheck.sh

# Run camera tracking
./tools/evaluate_dycheck.sh

# Run consistent video depth optimization
./cvd_opt/cvd_opt_dycheck.sh

# Evaluate results
python ./evaluations_poses/evaluate_dycheck.py
python ./evaluations_depth/evaluate_depth_ours_dycheck.py
```

### For demo videos (DAVIS):
```bash
# Precompute mono-depth
./mono_depth_scripts/run_mono-depth_demo.sh

# Run camera tracking
./tools/evaluate_demo.sh

# Run consistent video depth optimization
./cvd_opt/cvd_opt_demo.sh
```

## Notes

- Make sure to modify the data paths in the shell scripts to match your mounted volumes
- The conda environment `mega_sam` is automatically activated in the container
- GPU support is required for training and inference
- For visualization, you may need X11 forwarding (Linux) or other display solutions

## Troubleshooting

- If you encounter CUDA memory issues, try reducing batch sizes in the scripts
- For compilation issues with camera tracking extensions, ensure CUDA toolkit is properly installed
- XFormers installation may take some time during the build process

## Original Repository

MegaSAM: https://github.com/mega-sam/mega-sam
