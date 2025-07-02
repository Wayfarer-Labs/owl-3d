# Video Comparison Tools - Usage Guide

This directory contains multiple tools for creating side-by-side comparison videos of original frames and AnyCam depth maps.

## Tools Available

### 1. `create_comparison_video.py` (Primary Tool)
**Python-based with automatic fallbacks**

```bash
python3 create_comparison_video.py <input_video_or_frames> <anycam_output_dir> [options]

# Examples:
python3 create_comparison_video.py input.mp4 /path/to/anycam/output -o comparison.mp4
python3 create_comparison_video.py /path/to/frames /path/to/anycam/output --fps 25 --colormap plasma
```

**Features:**
- Automatically tries multiple video codecs (MP4V, MJPG, XVID)
- Falls back to ffmpeg if OpenCV VideoWriter fails
- Final fallback to individual PNG frames + ffmpeg command
- Supports both individual depth files and single `depths.npy` file
- Handles various depth array shapes automatically

### 2. `create_comparison_ffmpeg.sh` (ffmpeg-only)
**Pure ffmpeg approach for maximum compatibility**

```bash
./create_comparison_ffmpeg.sh <input_video_or_frames> <anycam_output_dir> [output_video] [fps]

# Examples:
./create_comparison_ffmpeg.sh input.mp4 /path/to/anycam/output
./create_comparison_ffmpeg.sh input.mp4 /path/to/anycam/output comparison.mp4 25
```

**Features:**
- Uses ffmpeg directly (more reliable in Docker environments)
- Processes frames with Python, creates video with ffmpeg
- Always creates high-quality H.264 videos

### 3. Wrapper Scripts for Docker

#### `create_comparison_video.sh` (In-container)
```bash
# Use inside the Docker container
./create_comparison_video.sh input.mp4 /workspace/anycam_output
```

#### `create_comparison_video_docker.sh` (Host)
```bash
# Use from host system to run in Docker
./create_comparison_video_docker.sh input.mp4 /path/to/anycam_output
```

## Testing Tools

### `test_video_creation.py`
Test OpenCV video writing capabilities:
```bash
python3 test_video_creation.py
```

### `test_comparison_workflow.sh`
End-to-end test of the video comparison workflow:
```bash
./test_comparison_workflow.sh
```

## Troubleshooting

### Video Creation Issues

1. **OpenCV VideoWriter fails:**
   - The primary tool automatically falls back to ffmpeg
   - Use `test_video_creation.py` to diagnose codec issues

2. **No output video created:**
   - Check that ffmpeg is installed: `ffmpeg -version`
   - Verify input files exist and are readable
   - Use `--list-files` option to debug file detection

3. **Docker display issues:**
   - The tools work without display (headless)
   - X11 forwarding not required for video creation

### File Format Issues

1. **Depth files not found:**
   - Check the AnyCam output directory contains `.npy` files
   - Supports both individual files (`depth_0000.npy`) and combined (`depths.npy`)

2. **Frame/depth count mismatch:**
   - The tools automatically find common indices
   - Use `--list-files` to see what files were detected

## Recommended Workflow

1. **First attempt:** Use the primary Python tool
   ```bash
   python3 create_comparison_video.py input.mp4 anycam_output/ -o result.mp4
   ```

2. **If that fails:** Use the ffmpeg-only version
   ```bash
   ./create_comparison_ffmpeg.sh input.mp4 anycam_output/ result.mp4
   ```

3. **For testing:** Run the diagnostic tools
   ```bash
   python3 test_video_creation.py
   ./test_comparison_workflow.sh
   ```

## Output

All tools create side-by-side videos with:
- Left side: Original frames
- Right side: Depth maps (colorized)
- Frame numbers and labels
- Consistent timing and resolution

The output videos are compatible with standard video players and suitable for presentations or analysis.
