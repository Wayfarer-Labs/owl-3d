#!/bin/bash

# process_side_by_side_docker.sh
# Run side_by_side_video.py inside Docker to generate an RGB+depth side-by-side video
#
# Usage:
#   ./process_side_by_side_docker.sh \
#     --input_video /path/to/video.mp4 \
#     --depth_pt /path/to/depths.pt \
#     --output_video /path/to/combined.mp4 [--fps N] [--container NAME]

set -e

# Default container name
CONTAINER_NAME="docker.io/library/anycam:latest"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
  cat << EOF
Usage: $0 \
  --input_video PATH \
  --depth_pt PATH \
  --output_video PATH [--fps N] [--container NAME]
EOF
}

# parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --input_video) INPUT_VIDEO="$2"; shift 2;;
    --depth_pt)    DEPTH_PT="$2";    shift 2;;
    --output_video) OUTPUT_VIDEO="$2"; shift 2;;
    --fps)         FPS="$2";         shift 2;;
    --container)   CONTAINER_NAME="$2"; shift 2;;
    -h|--help)     usage; exit 0;;
    *) print_error "Unknown arg: $1"; usage; exit 1;;
  esac
done

# validate
if [[ -z "$INPUT_VIDEO" || -z "$DEPTH_PT" || -z "$OUTPUT_VIDEO" ]]; then
  print_error "input_video, depth_pt and output_video are required"
  usage; exit 1
fi

# resolve absolute paths
audio_input=$(realpath "$INPUT_VIDEO")
depth_file=$(realpath "$DEPTH_PT")
output_file=$(realpath "$OUTPUT_VIDEO")

input_dir=$(dirname "$audio_input")
depth_dir=$(dirname "$depth_file")
output_dir=$(dirname "$output_file")
script_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

# create output directory
mkdir -p "$output_dir"

print_info "Using container: $CONTAINER_NAME"
print_info "Input video: $audio_input"
print_info "Depth file: $depth_file"
print_info "Output video: $output_file"
[[ -n "$FPS" ]] && print_info "FPS: $FPS"

# check docker
if ! command -v docker &>/dev/null; then
  print_error "Docker not found"; exit 1
fi

# run docker
print_info "Running side_by_side_video.py inside container"
docker run --rm \
  -v "$input_dir:/workspace/data_video:ro" \
  -v "$depth_dir:/workspace/data_depth:ro" \
  -v "$output_dir:/workspace/output" \
  -v "$script_dir:/workspace/scripts:ro" \
  -w /workspace/scripts \
  $CONTAINER_NAME \
  python side_by_side_video.py \
    --input_video /workspace/data_video/$(basename "$audio_input") \
    --depth_pt /workspace/data_depth/$(basename "$depth_file") \
    --output_video /workspace/output/$(basename "$output_file") \
    ${FPS:+--fps $FPS}

print_success "Combined video generated at $output_file"
