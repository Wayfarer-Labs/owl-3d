# AnyCam Development Container Setup

This directory contains the VS Code dev container configuration for the AnyCam project.

## Quick Start

1. **Prerequisites**:
   - VS Code with the "Dev Containers" extension installed
   - Docker with GPU support (nvidia-container-toolkit)
   - NVIDIA GPU drivers

2. **Open in Dev Container**:
   - Open VS Code in the `/anycam` directory
   - VS Code should detect the dev container configuration
   - Click "Reopen in Container" when prompted, or:
   - Open Command Palette (`Ctrl+Shift+P`) → "Dev Containers: Reopen in Container"

3. **Wait for Build**:
   - First time will take ~10-15 minutes to build
   - Subsequent starts will be much faster

## What's Included

### Development Environment
- CUDA 12.4 with cuDNN support
- Python 3.11 via Miniconda
- AnyCam conda environment pre-configured
- PyTorch with GPU support

### Development Tools
- Python development extensions (pylint, black, debugpy)
- Jupyter Lab support
- C++ tools for any native extensions
- Git configuration
- Common CLI tools (vim, htop, tmux)

### Pre-installed Python Packages
- All AnyCam requirements
- Development tools: ipython, ipdb, pytest
- Code formatting: black, flake8, pylint
- Jupyter: jupyter, jupyterlab
- Computer vision: opencv-python-headless, open3d

### Port Forwarding
- Port 9090: Rerun.io web viewer
- Port 8888: Jupyter Lab (if started)

## Usage Tips

### Terminal
The integrated terminal automatically activates the `anycam` conda environment.

### Data Directories
The following directories are mounted from your host:
- `./data` → `/workspace/anycam/data`
- `./outputs` → `/workspace/anycam/outputs` 
- `./output_frames` → `/workspace/anycam/output_frames`

### Running Scripts
```bash
# The conda environment is already activated
python test_s3_stream_vis.py

# Or explicitly activate if needed
source activate anycam
python your_script.py
```

### Jupyter Lab
```bash
# Start Jupyter Lab (accessible at http://localhost:8888)
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Debugging
The Python debugger is pre-configured. Set breakpoints in VS Code and use the debug panel.

## Customization

### Adding Python Packages
Edit `.devcontainer/Dockerfile.dev` and add pip install commands in the development dependencies section.

### Adding VS Code Extensions
Edit `.devcontainer/devcontainer.json` in the `customizations.vscode.extensions` array.

### Environment Variables
Add them to the `containerEnv` section in `devcontainer.json`.

## Troubleshooting

### GPU Access
If GPU is not available, check:
```bash
nvidia-smi  # Should show GPU info
```

### Conda Environment
If the environment isn't activated:
```bash
source activate anycam
```

### Rebuilding Container
If you need to rebuild after changes:
- Command Palette → "Dev Containers: Rebuild Container"

## Files

- `devcontainer.json`: Main configuration file
- `Dockerfile.dev`: Development-optimized Docker image
- `.dockerignore`: Files to exclude from Docker build context
- `README.md`: This file
