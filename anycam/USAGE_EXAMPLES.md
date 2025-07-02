# AnyCam Video Processing Examples

This file contains common usage examples for processing videos with AnyCam.

## Quick Start

Place your video in the `data/` directory and run one of these commands:

### Basic Processing
```bash
# Process with full refinement and visualization
./process_video.sh /workspace/data/your_video.mp4
```

### Fast Processing (No Refinement)
```bash
# Skip bundle adjustment for faster processing
./process_video.sh -r false /workspace/data/your_video.mp4
```

### High Frame Rate Videos
```bash
# Subsample to 10 FPS for better performance
./process_video.sh -f 10 /workspace/data/your_video.mp4
```

### Export to COLMAP
```bash
# Export camera poses in COLMAP format
./process_video.sh -c /workspace/data/your_video.mp4
```

### Custom Output Directory
```bash
# Specify custom output location
./process_video.sh -o /workspace/my_results /workspace/data/your_video.mp4
```

## Manual Commands

If you prefer to run the commands manually:

### Full Processing with Visualization
```bash
cd /workspace/anycam
python anycam/scripts/anycam_demo.py \
    ++input_path=/workspace/data/your_video.mp4 \
    ++model_path=pretrained_models/anycam_seq8 \
    ++output_path=/workspace/outputs \
    ++visualize=true
```

### Fast Processing (Feed-forward only)
```bash
cd /workspace/anycam
python anycam/scripts/anycam_demo.py \
    ++input_path=/workspace/data/your_video.mp4 \
    ++model_path=pretrained_models/anycam_seq8 \
    ++ba_refinement=false \
    ++output_path=/workspace/outputs \
    ++visualize=true
```

### Export to COLMAP Format
```bash
cd /workspace/anycam
python anycam/scripts/anycam_demo.py \
    ++input_path=/workspace/data/your_video.mp4 \
    ++model_path=pretrained_models/anycam_seq8 \
    ++export_colmap=true \
    ++output_path=/workspace/outputs
```

### PyTorch Hub Usage (Programmatic)
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

## Remote Server Setup

If running on a remote server without display:

1. **Start rerun web viewer:**
```bash
rerun --serve-web
```

2. **Forward port 9090 to your local machine:**
```bash
# On your local machine
ssh -L 9090:localhost:9090 your-server
```

3. **Run processing with connect mode:**
```bash
./process_video.sh --rerun-mode connect /workspace/data/your_video.mp4
```

4. **Open browser to:** `http://localhost:9090/?url=ws://localhost:9877`

## Output Files

After processing, you'll find these files in your output directory:

- **trajectory.txt** - Camera poses for each frame
- **depths/** - Depth maps as images or numpy arrays
- **uncertainties/** - Uncertainty maps
- **projection_matrix.txt** - Camera intrinsics
- **pointcloud.ply** - 3D point cloud (if generated)
- **colmap/** - COLMAP reconstruction files (if exported)

## Tips

- **Video Format**: MP4, AVI, MOV are supported
- **Resolution**: Higher resolution videos may require more GPU memory
- **Frame Rate**: For videos >30fps, consider using `-f 10` or `-f 15`
- **Memory**: Large videos may need 8GB+ GPU memory
- **Speed**: Disable refinement (`-r false`) for 2-3x faster processing

## Troubleshooting

- **Out of Memory**: Reduce video resolution or use `-f` to subsample frames
- **Slow Processing**: Disable refinement with `-r false`
- **No Visualization**: Check if display is available or use remote setup
- **Missing Libraries**: Run `./test_installation.sh` to verify setup
