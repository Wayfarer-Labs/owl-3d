"""
Simplified AnyCam processor: load model and run inference on frames directly.
"""

import torch
from typing import List, Dict, Any
import numpy as np

# restore I/O imports for saving results
import os
import json
from typing import Optional


def load_anycam_model():
    """Clone AnyCam repo with submodules and load model locally."""
    import os, subprocess
    print("Cloning or updating AnyCam repository with submodules...")
    cache_dir = torch.hub.get_dir()
    repo_dir = os.path.join(cache_dir, 'Brummi_anycam')
    if not os.path.isdir(repo_dir):
        subprocess.check_call([
            'git', 'clone', '--recursive',
            'https://github.com/Brummi/anycam.git', repo_dir
        ])
    else:
        subprocess.check_call([
            'git', 'submodule', 'update', '--init', '--recursive'
        ], cwd=repo_dir)

    print("Loading AnyCam model from local repository...")
    try:
        anycam = torch.hub.load(
            repo_dir,
            'AnyCam',
            source='local',
            version="1.0",
            training_variant="seq8",
            pretrained=True
        )
        print("AnyCam model loaded successfully from local repo")
        return anycam.cuda() if torch.cuda.is_available() else anycam.cpu()
    except Exception as e:
        raise RuntimeError(f"Failed to load AnyCam model from local repo: {e}")


def process_frames_with_anycam(anycam, frames: List[np.ndarray], ba_refinement: bool = False) -> Dict[str, Any]:
    """
    Process frames through AnyCam and return results.
    """
    print(f"Processing {len(frames)} frames through AnyCam...")
    print(f"Bundle adjustment refinement: {ba_refinement}")
    print(f"Frame shape: {frames[0].shape}")
    
    # convert frames to float32 [0,1] to avoid Byte dtype errors during upsampling
    frames = [frame.astype(np.float32) / 255.0 for frame in frames]

    try:
        # Process frames through AnyCam
        results = anycam.process_video(frames, ba_refinement=ba_refinement)
        
        print("AnyCam processing completed successfully")
        print(f"Results keys: {list(results.keys())}")
        
        # Print result shapes/info
        if "trajectory" in results:
            print(f"Trajectory shape: {results['trajectory'].shape if hasattr(results['trajectory'], 'shape') else type(results['trajectory'])}")
        if "depths" in results:
            print(f"Depths shape: {results['depths'].shape if hasattr(results['depths'], 'shape') else type(results['depths'])}")
        if "uncertainties" in results:
            print(f"Uncertainties shape: {results['uncertainties'].shape if hasattr(results['uncertainties'], 'shape') else type(results['uncertainties'])}")
        if "projection_matrix" in results:
            print(f"Projection matrix shape: {results['projection_matrix'].shape if hasattr(results['projection_matrix'], 'shape') else type(results['projection_matrix'])}")
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"AnyCam processing failed: {e}")

# restore save_results function for output serialization
def save_results(results: Dict[str, Any], output_dir: str, input_video_name: str, frames: Optional[List[np.ndarray]] = None) -> List[str]:
    """
    Save AnyCam results to output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_video_name))[0]
    saved_files: List[str] = []

    # Save trajectory, depths, uncertainties, projection matrix
    if "trajectory" in results:
        path = os.path.join(output_dir, f"{base_name}_trajectory.pt")
        torch.save(results["trajectory"], path)
        saved_files.append(path)

    if "depths" in results:
        path = os.path.join(output_dir, f"{base_name}_depths.pt")
        torch.save(results["depths"], path)
        saved_files.append(path)

    if "uncertainties" in results:
        path = os.path.join(output_dir, f"{base_name}_uncertainties.pt")
        torch.save(results["uncertainties"], path)
        saved_files.append(path)

    if "projection_matrix" in results:
        path = os.path.join(output_dir, f"{base_name}_projection_matrix.pt")
        torch.save(results["projection_matrix"], path)
        saved_files.append(path)

    # Save video tensor and resized version if frames provided
    if frames is not None:
        video_tensor = torch.stack([torch.from_numpy(f).permute(2,0,1) for f in frames])
        path = os.path.join(output_dir, f"{base_name}_video.pt")
        torch.save(video_tensor, path)
        saved_files.append(path)

        if "depths" in results:
            _, _, Hd, Wd = results["depths"].shape
            resized = torch.nn.functional.interpolate(video_tensor, size=(Hd, Wd), mode="bilinear", align_corners=False)
            path = os.path.join(output_dir, f"{base_name}_video_resized.pt")
            torch.save(resized, path)
            saved_files.append(path)

    # Save metadata
    metadata = {
        "input_video": input_video_name,
        "num_frames": len(results.get("depths", [])) if "depths" in results else None,
        "saved_files": saved_files,
        "results_keys": list(results.keys())
    }
    if frames is not None:
        metadata["video_tensor"] = path

    metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return saved_files
