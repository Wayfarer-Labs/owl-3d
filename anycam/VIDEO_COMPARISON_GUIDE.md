# AnyCam Video Comparison Guide

This guide shows you how to create side-by-side comparison videos showing original frames and depth maps from AnyCam processing.

## 🎬 What You'll Get

A video showing:
- **Left side**: Original video frames
- **Right side**: Depth maps with color coding (closer = warmer colors)

## 🚀 Quick Start

### Option 1: Run from Host (Recommended)

```bash
# Basic usage
./create_comparison_video_docker.sh data/your_video.mp4 outputs/

# Custom output filename
./create_comparison_video_docker.sh -o my_comparison.mp4 data/video.mp4 outputs/

# Different colormap (try: viridis, plasma, inferno, magma, jet)
./create_comparison_video_docker.sh --colormap plasma data/video.mp4 outputs/
```

### Option 2: Run Inside Container

```bash
# Enter container
./run_docker.sh

# Run inside container
./create_comparison_video.sh /workspace/data/video.mp4 /workspace/outputs/
```

## 🛠️ Usage Examples

### Basic Comparison Video
```bash
./create_comparison_video_docker.sh data/my_video.mp4 outputs/
```
Creates `comparison_video.mp4` with default settings.

### Custom Output and Colormap
```bash
./create_comparison_video_docker.sh \
    -o depth_analysis.mp4 \
    --colormap inferno \
    --fps 24 \
    data/my_video.mp4 outputs/
```

### Debug File Matching
If the script can't find depth files, use this to see what it found:
```bash
./create_comparison_video_docker.sh --list-files data/video.mp4 outputs/
```

## 🎨 Available Colormaps

- **viridis** (default): Blue → Green → Yellow (perceptually uniform)
- **plasma**: Purple → Pink → Yellow (perceptually uniform) 
- **inferno**: Black → Red → Yellow (dramatic)
- **magma**: Black → Purple → White (monochromatic feel)
- **jet**: Blue → Green → Red (classic, high contrast)

## 📁 Expected Directory Structure

After running AnyCam, your output directory should contain depth files:

```
outputs/
├── depths/
│   ├── depth_000.png
│   ├── depth_001.png
│   └── ...
├── trajectory.txt
└── ...
```

Or alternatively:
```
outputs/
├── depth_000.png
├── depth_001.png
├── ...
```

## 🔧 Troubleshooting

### No Depth Files Found
```bash
# Check what files exist
./create_comparison_video_docker.sh --list-files data/video.mp4 outputs/
```

The script looks for these patterns:
- `depth_*.png/jpg/npy`
- `depths/depth_*.png/jpg/npy`  
- `*_depth.png/jpg/npy`

### Frame/Depth Mismatch
The script automatically matches frame numbers with depth file numbers. If you see "No matching frame and depth indices", the numbering might not align.

### Python Dependencies
If running outside Docker, ensure you have:
```bash
pip install opencv-python matplotlib numpy tqdm
```

## 📋 Command Reference

### create_comparison_video_docker.sh
```bash
./create_comparison_video_docker.sh [OPTIONS] <input_video> <anycam_output_dir>

Options:
  -o, --output FILE     Output video filename
  --fps FPS            Output video FPS (default: 30)
  --colormap NAME      Depth colormap 
  --list-files         Debug: list found files
  --container NAME     Docker image name
  -h, --help           Show help
```

## 🎯 Tips

1. **Performance**: Higher FPS videos take longer to process
2. **Quality**: The output video quality depends on your input video resolution
3. **Colormaps**: Try different colormaps to see which works best for your data
4. **Debugging**: Always use `--list-files` first if you have issues

## 📝 Examples of Good Results

- **Indoor scenes**: `viridis` or `plasma` work well
- **Outdoor scenes**: `inferno` can highlight distant objects nicely  
- **High contrast needs**: `jet` provides maximum color range
- **Scientific visualization**: `viridis` is perceptually uniform

## 🔄 Complete Workflow

1. **Process video with AnyCam**:
   ```bash
   ./process_video_docker.sh data/video.mp4
   ```

2. **Create comparison video**:
   ```bash
   ./create_comparison_video_docker.sh data/video.mp4 outputs/
   ```

3. **View results**: Open `comparison_video.mp4`

That's it! You now have a side-by-side comparison showing how AnyCam perceived the depth in your video. 🎉
