#!/bin/bash

# AnyCam Docker Video Processing Script
# Run AnyCam video processing from outside the container

set -e  # Exit on any error

# Default values
INPUT_PATH=""
OUTPUT_PATH="$(pwd)/outputs"
MODEL_PATH="pretrained_models/anycam_seq8"
VISUALIZE="true"
BA_REFINEMENT="true"
EXPORT_COLMAP="false"
FPS="30"
RERUN_MODE="spawn"
CONTAINER_NAME="anycam:latest"
BATCH_SIZE=""
BATCH_OVERLAP="0"
BATCH_MODE="false"

# Function to display help
show_help() {
    echo "AnyCam Docker Video Processing Script"
    echo "===================================="
    echo ""
    echo "Usage: $0 [OPTIONS] <input_video>"
    echo ""
    echo "Arguments:"
    echo "  <input_video>     Path to input video file (required)"
    echo ""
    echo "Options:"
    echo "  -o, --output DIR        Output directory (default: ./outputs)"
    echo "  -m, --model PATH        Model path (default: pretrained_models/anycam_seq8)"
    echo "  -v, --visualize BOOL    Enable visualization (default: true)"
    echo "  -r, --refinement BOOL   Enable bundle adjustment refinement (default: true)"
    echo "  -c, --colmap           Export to COLMAP format"
    echo "  -f, --fps FPS          Subsample video to specified FPS"
    echo "  -b, --batch-size SIZE  Process video in batches of N frames"
    echo "  --batch-overlap N      Overlap between batches in frames (default: 0)"
    echo "  --rerun-mode MODE      Rerun mode: spawn|connect (default: spawn)"
    echo "  --container NAME       Docker container name (default: anycam:latest)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  $0 ./data/video.mp4"
    echo ""
    echo "  # Fast processing without refinement"
    echo "  $0 -r false ./data/video.mp4"
    echo ""
    echo "  # Export to COLMAP"
    echo "  $0 -c ./data/video.mp4"
    echo ""
    echo "  # Subsample high framerate video"
    echo "  $0 -f 10 ./data/video.mp4"
    echo ""
    echo "  # Custom output directory"
    echo "  $0 -o ./my_results ./data/video.mp4"
    echo ""
    echo "  # Batch processing with sliding window"
    echo "  $0 -b 100 --batch-overlap 20 ./data/video.mp4"
    echo ""
    echo "  # Process 50 frames at a time with 10 frame overlap"
    echo "  $0 -b 50 --batch-overlap 10 -o ./batch_results ./data/video.mp4"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -v|--visualize)
            VISUALIZE="$2"
            shift 2
            ;;
        -r|--refinement)
            BA_REFINEMENT="$2"
            shift 2
            ;;
        -c|--colmap)
            EXPORT_COLMAP="true"
            shift
            ;;
        -f|--fps)
            FPS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            BATCH_MODE="true"
            shift 2
            ;;
        --batch-overlap)
            BATCH_OVERLAP="$2"
            shift 2
            ;;
        --rerun-mode)
            RERUN_MODE="$2"
            shift 2
            ;;
        --container)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            if [ -z "$INPUT_PATH" ]; then
                INPUT_PATH="$1"
            else
                echo "Multiple input files specified. Only one is supported."
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if input path is provided
if [ -z "$INPUT_PATH" ]; then
    echo "Error: Input video path is required."
    echo ""
    show_help
    exit 1
fi

# Convert to absolute paths
INPUT_PATH=$(realpath "$INPUT_PATH")
OUTPUT_PATH=$(realpath "$OUTPUT_PATH")

# Check if input file exists
if [ ! -f "$INPUT_PATH" ]; then
    echo "Error: Input file does not exist: $INPUT_PATH"
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if the container image exists
if ! docker image inspect "$CONTAINER_NAME" &> /dev/null; then
    echo "Error: Docker image '$CONTAINER_NAME' not found."
    echo "Please build it first with: ./build_docker.sh"
    exit 1
fi

# Check if GPU support is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Running without GPU support."
    GPU_FLAGS=""
else
    GPU_FLAGS="--gpus all"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Function to get video frame count
get_frame_count() {
    local video_path="$1"
    
    # Method 1: Try to get frame count directly
    local frame_count=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$video_path" 2>/dev/null)
    
    if [ -n "$frame_count" ] && [ "$frame_count" -gt 0 ] 2>/dev/null; then
        echo "$frame_count"
        return 0
    fi
    
    # Method 2: Count frames by decoding (more accurate but slower)
    echo "Direct frame count failed, counting frames..." >&2
    frame_count=$(ffprobe -v quiet -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$video_path" 2>/dev/null)
    
    if [ -n "$frame_count" ] && [ "$frame_count" -gt 0 ] 2>/dev/null; then
        echo "$frame_count"
        return 0
    fi
    
    # Method 3: Calculate from duration and frame rate
    echo "Frame counting failed, calculating from duration..." >&2
    local duration=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$video_path" 2>/dev/null)
    local fps=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$video_path" 2>/dev/null)
    
    if [ -n "$duration" ] && [ -n "$fps" ]; then
        if [[ "$fps" == *"/"* ]]; then
            # Convert fraction to decimal
            local fps_decimal=$(awk "BEGIN {printf \"%.6f\", $fps}")
        else
            local fps_decimal="$fps"
        fi
        
        # Calculate frame count
        local calculated_frames=$(awk "BEGIN {printf \"%.0f\", $duration * $fps_decimal}")
        
        if [ "$calculated_frames" -gt 0 ] 2>/dev/null; then
            echo "$calculated_frames"
            return 0
        fi
    fi
    
    # Method 4: Simple fallback - just count by seeking through the video
    echo "All methods failed, using fallback frame counting..." >&2
    frame_count=$(ffmpeg -i "$video_path" -map 0:v:0 -c copy -f null - 2>&1 | grep -o "frame=[[:space:]]*[0-9]*" | tail -1 | grep -o "[0-9]*")
    
    if [ -n "$frame_count" ] && [ "$frame_count" -gt 0 ] 2>/dev/null; then
        echo "$frame_count"
        return 0
    fi
    
    # If all methods fail, return 0
    echo "0"
}

# Function to extract frames from video batch
extract_video_batch() {
    local input_video="$1"
    local start_frame="$2"
    local frame_count="$3"
    local output_dir="$4"
    local batch_id="$5"
    
    echo "Extracting batch $batch_id: frames $start_frame to $((start_frame + frame_count - 1))" >&2
    
    # Create batch-specific temporary directory
    local batch_temp_dir="$output_dir/temp_batch_$batch_id"
    mkdir -p "$batch_temp_dir"
    
    # Calculate start time and duration instead of frame selection
    # Get video frame rate first
    local fps=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$input_video" 2>/dev/null)
    
    if [ -z "$fps" ]; then
        echo "Error: Could not get frame rate from video" >&2
        return 1
    fi
    
    if [[ "$fps" == *"/"* ]]; then
        local fps_decimal=$(awk "BEGIN {printf \"%.6f\", $fps}")
    else
        local fps_decimal="$fps"
    fi
    
    # Validate FPS
    if (( $(awk "BEGIN {print ($fps_decimal <= 0)}") )); then
        echo "Error: Invalid frame rate: $fps_decimal" >&2
        return 1
    fi
    
    # Calculate start time and duration
    local start_time=$(awk "BEGIN {printf \"%.6f\", $start_frame / $fps_decimal}")
    local duration=$(awk "BEGIN {printf \"%.6f\", $frame_count / $fps_decimal}")
    
    echo "Debug: fps=$fps, fps_decimal=$fps_decimal" >&2
    echo "Debug: start_frame=$start_frame, frame_count=$frame_count" >&2
    echo "Debug: Calculated start_time=$start_time, duration=$duration seconds" >&2
    
    # Get video duration to validate we're not going past the end
    local video_duration=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$input_video" 2>/dev/null)
    
    # Check if start time is already beyond video duration
    if [ -n "$video_duration" ] && (( $(awk "BEGIN {print ($start_time >= $video_duration)}") )); then
        echo "Error: Start time $start_time is beyond video duration $video_duration" >&2
        return 1
    fi
    
    local max_end_time=$(awk "BEGIN {printf \"%.6f\", $start_time + $duration}")
    
    if [ -n "$video_duration" ] && (( $(awk "BEGIN {print ($max_end_time > $video_duration)}") )); then
        # Adjust duration to not go past end of video
        duration=$(awk "BEGIN {printf \"%.6f\", $video_duration - $start_time}")
        echo "Adjusted duration to $duration to stay within video bounds" >&2
        
        if (( $(awk "BEGIN {print ($duration <= 0)}") )); then
            echo "Start time $start_time is beyond video duration $video_duration" >&2
            return 1
        fi
    fi
    
    # Create a video directly from the time range
    local batch_video="$batch_temp_dir/batch_$batch_id.mp4"
    
    echo "Debug: Extracting from time $start_time for duration $duration seconds" >&2
    
    # Use time-based extraction instead of frame-based  
    echo "Running: ffmpeg -ss $start_time -i \"$input_video\" -t $duration -c copy \"$batch_video\" -y" >&2
    ffmpeg -ss "$start_time" -i "$input_video" -t "$duration" -c copy "$batch_video" -y 2>&1 | tee /tmp/ffmpeg_batch_${batch_id}.log >&2
    local ffmpeg_exit_code=${PIPESTATUS[0]}
    
    if [ $ffmpeg_exit_code -ne 0 ] || [ ! -f "$batch_video" ] || [ ! -s "$batch_video" ]; then
        echo "Copy method failed (exit code: $ffmpeg_exit_code), trying re-encoding..." >&2
        echo "FFmpeg log:" >&2
        cat /tmp/ffmpeg_batch_${batch_id}.log >&2
        
        ffmpeg -ss "$start_time" -i "$input_video" -t "$duration" -c:v libx264 -preset fast "$batch_video" -y 2>&1 | tee /tmp/ffmpeg_reenc_${batch_id}.log >&2
        local ffmpeg_reenc_exit_code=${PIPESTATUS[0]}
        
        if [ $ffmpeg_reenc_exit_code -ne 0 ]; then
            echo "Re-encoding also failed (exit code: $ffmpeg_reenc_exit_code)" >&2
            echo "Re-encoding log:" >&2
            cat /tmp/ffmpeg_reenc_${batch_id}.log >&2
        fi
    fi
    
    # Verify the output video exists and has content
    if [ -f "$batch_video" ]; then
        local video_size=$(stat -c%s "$batch_video" 2>/dev/null || echo "0")
        if [ "$video_size" -gt 1000 ]; then  # At least 1KB
            # Clean up log files
            rm -f /tmp/ffmpeg_batch_${batch_id}.log /tmp/ffmpeg_reenc_${batch_id}.log
            echo "$batch_video"
            return 0
        else
            echo "Generated video is too small (${video_size} bytes), likely empty" >&2
            echo "Check FFmpeg logs for details:" >&2
            cat /tmp/ffmpeg_batch_${batch_id}.log 2>/dev/null >&2 || echo "No log file found" >&2
            cat /tmp/ffmpeg_reenc_${batch_id}.log 2>/dev/null >&2 || echo "No re-encoding log file found" >&2
            rm -f "$batch_video"
            rm -f /tmp/ffmpeg_batch_${batch_id}.log /tmp/ffmpeg_reenc_${batch_id}.log
            return 1
        fi
    else
        echo "No output video generated" >&2
        echo "Check FFmpeg logs for details:" >&2
        cat /tmp/ffmpeg_batch_${batch_id}.log 2>/dev/null >&2 || echo "No log file found" >&2
        cat /tmp/ffmpeg_reenc_${batch_id}.log 2>/dev/null >&2 || echo "No re-encoding log file found" >&2
        rm -f /tmp/ffmpeg_batch_${batch_id}.log /tmp/ffmpeg_reenc_${batch_id}.log
        return 1
    fi
}

# Function to process a single batch
process_batch() {
    local batch_video="$1"
    local batch_output_dir="$2"
    local batch_id="$3"
    
    echo "Processing batch $batch_id..."
    
    # Get container paths for this batch
    local container_batch_input="/workspace/data/$(basename "$batch_video")"
    local container_batch_output="/workspace/outputs"
    
    # Build AnyCam command for this batch
    local batch_cmd="cd /workspace/anycam && python anycam/scripts/anycam_demo.py"
    batch_cmd="$batch_cmd ++input_path=$container_batch_input"
    batch_cmd="$batch_cmd ++model_path=$MODEL_PATH"
    batch_cmd="$batch_cmd ++output_path=$container_batch_output"
    batch_cmd="$batch_cmd ++visualize=$VISUALIZE"
    batch_cmd="$batch_cmd ++ba_refinement=$BA_REFINEMENT"
    
    if [ "$EXPORT_COLMAP" = "true" ]; then
        batch_cmd="$batch_cmd ++export_colmap=true"
    fi
    
    if [ -n "$FPS" ]; then
        batch_cmd="$batch_cmd ++fps=$FPS"
    fi
    
    if [ "$RERUN_MODE" = "connect" ]; then
        batch_cmd="$batch_cmd ++rerun_mode=connect"
    fi
    
    # Run Docker container for this batch
    # Use non-interactive mode for batch processing
    docker run $GPU_FLAGS --rm \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -v "$(dirname "$batch_video"):/workspace/data" \
        -v "$batch_output_dir:/workspace/outputs" \
        -p 9090:9090 \
        "$CONTAINER_NAME" \
        bash -c "$batch_cmd"
    
    echo "Batch $batch_id processing complete"
}

# Function to run batch processing
run_batch_processing() {
    echo "Starting batch processing mode..."
    
    # Check if ffmpeg is available for video processing
    if ! command -v ffmpeg &> /dev/null; then
        echo "Error: ffmpeg is required for batch processing but not found in PATH"
        exit 1
    fi
    
    if ! command -v ffprobe &> /dev/null; then
        echo "Error: ffprobe is required for batch processing but not found in PATH"
        exit 1
    fi
    

    
    # Get total frame count
    echo "Analyzing video..."
    echo "Video path: $INPUT_PATH"
    
    # Get video properties for debugging
    echo "Video information:"
    ffprobe -v quiet -show_format -show_streams "$INPUT_PATH" | grep -E "(duration|nb_frames|r_frame_rate)" || echo "Could not get video info"
    
    local total_frames=$(get_frame_count "$INPUT_PATH")
    echo "Total frames detected: $total_frames"
    echo "Batch size: $BATCH_SIZE"
    echo "Batch overlap: $BATCH_OVERLAP"
    
    # Get video duration and FPS for debugging
    local video_duration=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$INPUT_PATH" 2>/dev/null)
    local video_fps=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$INPUT_PATH" 2>/dev/null)
    echo "Video duration: ${video_duration:-unknown} seconds"
    echo "Video FPS: ${video_fps:-unknown}"
    
    # Validate frame count
    if [ -z "$total_frames" ] || [ "$total_frames" -le 0 ]; then
        echo "Error: Could not determine video frame count or video has no frames"
        echo "Debug: Let's check what ffprobe can tell us about this video:"
        ffprobe -v quiet -show_format -show_streams "$INPUT_PATH" 2>/dev/null || echo "ffprobe failed completely"
        exit 1
    fi
    
    # Calculate batch parameters
    local step_size=$((BATCH_SIZE - BATCH_OVERLAP))
    local current_frame=0
    local batch_id=1
    
    # Calculate total batches more carefully
    if [ "$step_size" -gt 0 ]; then
        local total_batches=$(( (total_frames + step_size - 1) / step_size ))
    else
        echo "Error: Step size must be positive (batch_size > batch_overlap)"
        exit 1
    fi
    
    echo "Processing $total_batches batches with step size of $step_size frames"
    echo ""
    
    # Create main batch output directory
    local batch_base_output="$OUTPUT_PATH/batches"
    mkdir -p "$batch_base_output"
    
    # Process each batch
    while [ $current_frame -lt $total_frames ]; do
        # Calculate frames for this batch
        local remaining_frames=$((total_frames - current_frame))
        local frames_to_process=$BATCH_SIZE
        
        if [ $remaining_frames -lt $BATCH_SIZE ]; then
            frames_to_process=$remaining_frames
        fi
        
        echo "=== Processing Batch $batch_id/$total_batches ==="
        echo "Frame range: $current_frame to $((current_frame + frames_to_process - 1))"
        
        # Create batch-specific output directory
        local batch_output="$batch_base_output/batch_$(printf "%03d" $batch_id)"
        mkdir -p "$batch_output"
        
        # Extract and process this batch
        echo "DEBUG: About to call extract_video_batch with:"
        echo "  INPUT_PATH: $INPUT_PATH"
        echo "  current_frame: $current_frame"
        echo "  frames_to_process: $frames_to_process"
        echo "  batch_output: $batch_output"
        echo "  batch_id: $batch_id"
        
        local batch_video=$(extract_video_batch "$INPUT_PATH" "$current_frame" "$frames_to_process" "$batch_output" "$batch_id")
        local extract_status=$?
        
        echo "DEBUG: extract_video_batch returned:"
        echo "  extract_status: $extract_status"
        echo "  batch_video: '$batch_video'"
        echo "  batch_video exists: $([ -f "$batch_video" ] && echo "YES" || echo "NO")"
        
        if [ $extract_status -eq 0 ] && [ -f "$batch_video" ]; then
            echo "DEBUG: Calling process_batch function"
            process_batch "$batch_video" "$batch_output" "$batch_id"
            
            # Clean up temporary batch video
            rm -f "$batch_video"
            rmdir "$(dirname "$batch_video")" 2>/dev/null || true
        else
            echo "Error: Failed to create batch video for batch $batch_id"
            echo "  Extract status: $extract_status"
            echo "  Batch video path: '$batch_video'"
            echo "  File exists: $([ -f "$batch_video" ] && echo "YES" || echo "NO")"
            echo "Skipping this batch..."
        fi
        
        # Move to next batch
        current_frame=$((current_frame + step_size))
        batch_id=$((batch_id + 1))
        
        echo ""
    done
    
    echo "All batches processed!"
    echo "Results saved in: $batch_base_output"
}

# Check if batch processing is enabled
if [ "$BATCH_MODE" = "true" ]; then
    if [ -z "$BATCH_SIZE" ] || [ "$BATCH_SIZE" -le 0 ]; then
        echo "Error: Invalid batch size. Must be a positive integer."
        exit 1
    fi
    
    if [ "$BATCH_OVERLAP" -lt 0 ] || [ "$BATCH_OVERLAP" -ge "$BATCH_SIZE" ]; then
        echo "Error: Batch overlap must be between 0 and batch size - 1."
        exit 1
    fi
fi

# Display configuration
echo "AnyCam Docker Video Processing"
echo "=============================="
echo "Input:        $INPUT_PATH"
echo "Output:       $OUTPUT_PATH"
echo "Model:        $MODEL_PATH"
echo "Visualize:    $VISUALIZE"
echo "Refinement:   $BA_REFINEMENT"
echo "Export COLMAP: $EXPORT_COLMAP"
if [ -n "$FPS" ]; then
    echo "FPS:          $FPS"
fi
echo "Rerun mode:   $RERUN_MODE"
echo "Container:    $CONTAINER_NAME"
if [ "$BATCH_MODE" = "true" ]; then
    echo ""
    echo "BATCH PROCESSING MODE:"
    echo "Batch size:   $BATCH_SIZE frames"
    echo "Batch overlap: $BATCH_OVERLAP frames"
    echo "Step size:    $((BATCH_SIZE - BATCH_OVERLAP)) frames"
fi
echo ""

# Ask for confirmation
read -p "Proceed with processing? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo "Starting Docker container and processing..."
echo ""

# Check if batch processing is enabled
if [ "$BATCH_MODE" = "true" ]; then
    run_batch_processing
else
    # Original single-video processing
    # Get container paths
    CONTAINER_INPUT="/workspace/data/$(basename "$INPUT_PATH")"
    CONTAINER_OUTPUT="/workspace/outputs"

    # Build the AnyCam command
    ANYCAM_CMD="cd /workspace/anycam && python anycam/scripts/anycam_demo.py"
    ANYCAM_CMD="$ANYCAM_CMD ++input_path=$CONTAINER_INPUT"
    ANYCAM_CMD="$ANYCAM_CMD ++model_path=$MODEL_PATH"
    ANYCAM_CMD="$ANYCAM_CMD ++output_path=$CONTAINER_OUTPUT"
    ANYCAM_CMD="$ANYCAM_CMD ++visualize=$VISUALIZE"
    ANYCAM_CMD="$ANYCAM_CMD ++ba_refinement=$BA_REFINEMENT"

    if [ "$EXPORT_COLMAP" = "true" ]; then
        ANYCAM_CMD="$ANYCAM_CMD ++export_colmap=true"
    fi

    if [ -n "$FPS" ]; then
        ANYCAM_CMD="$ANYCAM_CMD ++fps=$FPS"
    fi

    if [ "$RERUN_MODE" = "connect" ]; then
        ANYCAM_CMD="$ANYCAM_CMD ++rerun_mode=connect"
    fi

    # Determine if we should run Docker in interactive mode
    if [ -t 0 ] && [ -t 1 ]; then
        # Running interactively
        DOCKER_FLAGS="$GPU_FLAGS -it --rm"
    else
        # Running non-interactively (piped input/output)
        DOCKER_FLAGS="$GPU_FLAGS --rm"
    fi

    # Run the Docker container with the AnyCam command
    docker run $DOCKER_FLAGS \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -v "$(dirname "$INPUT_PATH"):/workspace/data" \
        -v "$OUTPUT_PATH:/workspace/outputs" \
        -p 9091:9090 \
        "$CONTAINER_NAME" \
        bash -c "$ANYCAM_CMD"
fi

echo ""
echo "Processing complete!"
echo "Results saved to: $OUTPUT_PATH"
