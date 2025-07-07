#!/usr/bin/env bash
# Wrapper script to run create_side_by_side.py inside AnyCam Docker container
# Usage: create_side_by_side.sh INPUT_DIR BASE_NAME OUTPUT_VIDEO [FPS] [COLORMAP]

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 INPUT_DIR BASE_NAME OUTPUT_VIDEO [FPS] [COLORMAP]"
  exit 1
fi

INPUT_DIR="$1"
BASE_NAME="$2"
OUTPUT_VIDEO="$3"
FPS="${4:-10.0}"
COLORMAP="${5:-JET}"

# After parsing input arguments, add script directory resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the conda environment (entrypoint already activates)
# Run the Python script
python3 "$SCRIPT_DIR/create_side_by_side.py" \
  --input_dir "$INPUT_DIR" \
  --base_name "$BASE_NAME" \
  --output_video "$OUTPUT_VIDEO" \
  --fps "$FPS" \
  --colormap "$COLORMAP"
