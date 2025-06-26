#!/bin/bash
# Script to run MEGA-SAM pipeline on CoD sequence data

DATA_DIR=../cod-output/
seq=CoD-sequence

# Activate the conda environment
echo "Activating mega_sam environment..."
eval "$(micromamba shell hook --shell=zsh)"
micromamba activate mega_sam

# Path to model checkpoint
CKPT_PATH=checkpoints/megasam_final.pth

echo "Processing CoD sequence at: $DATA_DIR"
ls $DATA_DIR | head -10

# First, run DepthAnything to generate mono depth files
echo "Checking DepthAnything progress..."
processed_count=$(ls Depth-Anything/video_visualization/$seq/*.npy 2>/dev/null | wc -l)
total_images=$(ls $DATA_DIR/*.{jpg,png} 2>/dev/null | wc -l)
echo "Found $processed_count depth files out of $total_images total images"

if [ $processed_count -lt $total_images ]; then
    echo "Resuming DepthAnything processing from frame $(($processed_count + 1))..."
    # Create temporary directory with remaining images
    temp_dir="/tmp/cod_remaining_$(date +%s)"
    mkdir -p $temp_dir
    
    # Get list of all images and copy only the unprocessed ones
    all_images=($(ls $DATA_DIR/*.{jpg,png} 2>/dev/null | sort))
    for ((i=$processed_count; i<${#all_images[@]}; i++)); do
        cp "${all_images[$i]}" "$temp_dir/"
    done
    
    echo "Processing remaining $(ls $temp_dir | wc -l) images..."
    CUDA_VISIBLE_DEVICES=0 python Depth-Anything/run_videos.py --encoder vitl \
    --load-from Depth-Anything/checkpoints/depth_anything_vitl14.pth \
    --img-path $temp_dir \
    --outdir Depth-Anything/video_visualization/$seq
    
    # Clean up temporary directory
    rm -rf $temp_dir
else
    echo "DepthAnything processing already complete, skipping..."
fi

# Second, run UniDepth to generate metric depth files
echo "Checking UniDepth progress..."
unidepth_count=$(ls UniDepth/outputs/$seq/*.npz 2>/dev/null | wc -l)
echo "Found $unidepth_count UniDepth files out of $total_images total images"

if [ $unidepth_count -lt $total_images ]; then
    echo "Resuming UniDepth processing from frame $(($unidepth_count + 1))..."
    # Create temporary directory with remaining images for UniDepth
    temp_unidepth_dir="/tmp/cod_unidepth_remaining_$(date +%s)"
    mkdir -p $temp_unidepth_dir
    
    # Get list of all images and copy only the unprocessed ones
    all_images=($(ls $DATA_DIR/*.{jpg,png} 2>/dev/null | sort))
    for ((i=$unidepth_count; i<${#all_images[@]}; i++)); do
        cp "${all_images[$i]}" "$temp_unidepth_dir/"
    done
    
    echo "Processing remaining $(ls $temp_unidepth_dir | wc -l) images with UniDepth..."
    CUDA_VISIBLE_DEVICES=0 python UniDepth/scripts/demo_mega-sam.py \
    --img-path $temp_unidepth_dir \
    --outdir UniDepth/outputs \
    --scene-name $seq
    
    # Clean up temporary directory
    rm -rf $temp_unidepth_dir
else
    echo "UniDepth processing already complete, skipping..."
fi

# Finally, run camera tracking with DROID-SLAM
echo "Starting camera tracking with DROID-SLAM..."
export PYTHONPATH=$(pwd)/base/build/lib.linux-x86_64-cpython-310:$(pwd)/base/droid_slam:$PYTHONPATH
CUDA_VISIBLE_DEVICE=0 micromamba run -n mega_sam python camera_tracking_scripts/test_demo.py \
--datapath=$DATA_DIR \
--weights=$CKPT_PATH \
--scene_name $seq \
--mono_depth_path $(pwd)/Depth-Anything/video_visualization \
--metric_depth_path $(pwd)/UniDepth/outputs \
$@

echo "Generated outputs:"
echo "  - outputs/${seq}_droid.npz"
echo "  - reconstructions/${seq}/"

echo "Camera tracking completed for CoD sequence"
