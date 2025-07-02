#!/usr/bin/env python3
"""
Test script to debug video creation issues
"""

import cv2
import numpy as np
import os
import tempfile

def test_opencv_video_writing():
    """Test basic OpenCV video writing capabilities"""
    print("Testing OpenCV video writing...")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Create test frames
    width, height = 640, 480
    num_frames = 10
    fps = 30
    
    # Test different codecs
    codecs_to_test = [
        ('MP4V', cv2.VideoWriter_fourcc(*'MP4V'), '.mp4'),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG'), '.avi'),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID'), '.avi'),
    ]
    
    results = {}
    
    for codec_name, fourcc, ext in codecs_to_test:
        print(f"\nTesting codec: {codec_name}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, f"test_video{ext}")
            
            try:
                # Create VideoWriter
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    print(f"  ❌ Failed to open VideoWriter")
                    results[codec_name] = False
                    continue
                
                print(f"  ✅ VideoWriter opened successfully")
                
                # Write test frames
                frames_written = 0
                for i in range(num_frames):
                    # Create a simple test frame with frame number
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    frame[:, :, 1] = 50  # Green background
                    
                    # Add frame number
                    cv2.putText(frame, f"Frame {i}", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                    
                    # Write frame
                    success = out.write(frame)
                    if success:
                        frames_written += 1
                    else:
                        print(f"  ⚠️  Failed to write frame {i}")
                
                out.release()
                
                # Check if file was created and has content
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    print(f"  ✅ Video file created: {file_size} bytes")
                    print(f"  ✅ Frames written: {frames_written}/{num_frames}")
                    
                    if file_size > 0 and frames_written > 0:
                        results[codec_name] = True
                        print(f"  ✅ {codec_name} codec working!")
                    else:
                        results[codec_name] = False
                        print(f"  ❌ {codec_name} codec failed (empty file or no frames written)")
                else:
                    print(f"  ❌ Video file not created")
                    results[codec_name] = False
                    
            except Exception as e:
                print(f"  ❌ Exception: {e}")
                results[codec_name] = False
    
    print("\n" + "="*50)
    print("SUMMARY:")
    working_codecs = [codec for codec, works in results.items() if works]
    if working_codecs:
        print(f"✅ Working codecs: {', '.join(working_codecs)}")
    else:
        print("❌ No codecs working!")
    
    failed_codecs = [codec for codec, works in results.items() if not works]
    if failed_codecs:
        print(f"❌ Failed codecs: {', '.join(failed_codecs)}")
    
    return results

def test_environment():
    """Test the environment for video creation"""
    print("Testing environment...")
    
    # Check if we're in a Docker container
    if os.path.exists('/.dockerenv'):
        print("✅ Running in Docker container")
    else:
        print("ℹ️  Running on host system")
    
    # Check display environment
    display = os.environ.get('DISPLAY')
    if display:
        print(f"✅ DISPLAY set: {display}")
    else:
        print("⚠️  DISPLAY not set (may affect some codecs)")
    
    # Check for video-related libraries
    try:
        import matplotlib
        print(f"✅ matplotlib available: {matplotlib.__version__}")
    except ImportError:
        print("⚠️  matplotlib not available")
    
    # Check OpenCV build info
    print(f"\nOpenCV build info:")
    build_info = cv2.getBuildInformation()
    
    # Look for important video-related info
    for line in build_info.split('\n'):
        line = line.strip()
        if any(keyword in line.lower() for keyword in ['ffmpeg', 'gstreamer', 'video', 'codec']):
            print(f"  {line}")

if __name__ == "__main__":
    print("🔬 Video Creation Test Script")
    print("="*50)
    
    test_environment()
    print("\n" + "="*50)
    test_opencv_video_writing()
    print("\n" + "="*50)
    print("Test complete!")
