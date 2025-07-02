# Quick Start Guide - Running AnyCam from Host

This guide shows how to run AnyCam processing from outside the Docker container for convenience.

## Setup

1. **Build the container** (one time):
```bash
./build_docker.sh
```

2. **Create data directory** and place your videos:
```bash
mkdir -p data
# Copy your video files to the data/ directory
```

## Usage

### Basic Processing
```bash
./process_video_docker.sh data/your_video.mp4
```

### Fast Processing (No Refinement)
```bash
./process_video_docker.sh -r false data/your_video.mp4
```

### High Frame Rate Videos
```bash
./process_video_docker.sh -f 10 data/your_video.mp4
```

### Export to COLMAP
```bash
./process_video_docker.sh -c data/your_video.mp4
```

### Custom Output Directory
```bash
./process_video_docker.sh -o ./my_results data/your_video.mp4
```

## What Happens

The script will:
1. ✅ Check if your video file exists
2. ✅ Verify the Docker image is built
3. ✅ Show you the configuration
4. ✅ Ask for confirmation
5. ✅ Run the container with proper volume mounts
6. ✅ Process your video
7. ✅ Save results to your specified output directory

## Directory Structure

```
anycam/
├── process_video_docker.sh  # Main script (run from host)
├── build_docker.sh          # Build container
├── Dockerfile               # Container definition
├── data/                    # Put your videos here
│   └── your_video.mp4
└── outputs/                 # Results appear here
    ├── trajectory.txt
    ├── depths/
    └── ...
```

## Benefits of This Approach

- 🎯 **No need to enter container** - Run directly from host
- 📁 **Automatic volume mounting** - Handles file paths automatically  
- 🔧 **All options available** - Full AnyCam functionality
- ✨ **User-friendly** - Clear prompts and error checking
- 🚀 **Fast iterations** - No container startup overhead between runs

## Help

```bash
./process_video_docker.sh --help
```

## Troubleshooting

- **Container not found**: Run `./build_docker.sh` first
- **Permission errors**: Make sure the script is executable: `chmod +x process_video_docker.sh`
- **GPU issues**: Script auto-detects GPU availability
- **File not found**: Use relative paths from the anycam directory
