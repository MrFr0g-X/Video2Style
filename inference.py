"""
Video stylization inference module for Video2Style.

This module handles the main inference logic for transforming videos
into stylized versions using different artistic styles.
"""
import torch
import numpy as np
from utils import extract_frames, create_video, compute_optical_flow
from stylizers.anime import AnimeStylizer
from stylizers.sketch import SketchStylizer
from stylizers.cinematic import CinematicStylizer
import cv2
import gc
import os


def run_inference(input_path, output_path, style, prompt, device="cpu", frame_skip=5, resolution=None):
    """
    Main inference logic for stylizing a video.
    
    Processes a video by extracting frames, stylizing key frames,
    interpolating between stylized keyframes, and creating the output video.
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to save stylized video
        style (str): Desired style (anime, sketch, cinematic)
        prompt (str): Optional text prompt for stylization
        device (str): Device to run inference on ("cpu" or "cuda")
        frame_skip (int): Process 1 frame every N frames to speed up
        resolution (tuple): Optional (width, height) to resize frames
    """
    # Extract frames from the input video
    frames, fps = extract_frames(input_path, return_fps=True)
    stylizer = get_stylizer(style, device)
    stylized_frames = []
    
    # Create key frames list (only process every Nth frame)
    keyframe_indices = list(range(0, len(frames), frame_skip))
    if keyframe_indices[-1] != len(frames) - 1:
        keyframe_indices.append(len(frames) - 1)  # Always include last frame
    
    # Create temp folder for intermediate results to reduce memory usage
    temp_folder = "temp_frames"
    os.makedirs(temp_folder, exist_ok=True)
    
    print(f"Processing {len(keyframe_indices)} keyframes with {style} style...")
    
    # Process each keyframe
    for i, idx in enumerate(keyframe_indices):
        frame = frames[idx]
        
        # Resize frame if resolution is specified
        if resolution:
            frame = cv2.resize(frame, resolution)
            
        print(f"Processing keyframe {i+1}/{len(keyframe_indices)} (frame {idx+1}/{len(frames)})")
        
        # Generate control image and stylize frame
        try:
            # Force garbage collection before heavy operations
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                
            # Generate control image
            control_image = stylizer.generate_control_image(frame)
            
            # Stylize the current frame
            stylized_frame = stylizer.stylize(frame, control_image, prompt)
            
            # Save to temp file to reduce memory usage
            temp_path = os.path.join(temp_folder, f"frame_{idx:05d}.jpg")
            cv2.imwrite(temp_path, cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR))
            
            # Add to stylized frames list
            stylized_frames.append(stylized_frame)
                
        except Exception as e:
            print(f"Error processing frame {idx}: {e}")
            # If there's an error, still use the stylized output if possible
            # Otherwise use original frame as fallback
            if i > 0 and len(stylized_frames) > 0:
                print(f"Using previous stylized frame as fallback")
                stylized_frames.append(stylized_frames[-1])
            else:
                print(f"Using original frame as fallback")
                stylized_frames.append(frame)
    
    # Interpolate intermediate frames if frame_skip > 1
    if frame_skip > 1:
        print("Interpolating skipped frames...")
        full_frames = interpolate_frames(frames, stylized_frames, keyframe_indices)
    else:
        full_frames = stylized_frames
    
    # Create output video
    print(f"Saving video to {output_path}...")
    create_video(full_frames, output_path, fps)
    
    # Clean up temp files
    for file in os.listdir(temp_folder):
        os.remove(os.path.join(temp_folder, file))
    os.rmdir(temp_folder)
    
    print("Done!")


def get_stylizer(style, device="cpu"):
    """
    Factory function to return the appropriate stylizer.
    
    Args:
        style (str): The style name ("anime", "sketch", or "cinematic")
        device (str): Device to run on ("cpu" or "cuda")
        
    Returns:
        object: An initialized stylizer object
        
    Raises:
        ValueError: If the requested style is not supported
    """
    stylizers = {
        "anime": AnimeStylizer,
        "sketch": SketchStylizer,
        "cinematic": CinematicStylizer
    }
    if style not in stylizers:
        raise ValueError(f"Unsupported style: {style}")
    return stylizers[style](device=device)


def blend_frames(prev_frame, current_frame, alpha=0.3):
    """
    Blend two frames for temporal smoothing.
    
    Args:
        prev_frame (np.ndarray): Previous frame
        current_frame (np.ndarray): Current frame
        alpha (float): Blending factor (0.0-1.0)
        
    Returns:
        np.ndarray: Blended frame
    """
    return (1 - alpha) * prev_frame + alpha * current_frame


def interpolate_frames(original_frames, stylized_keyframes, keyframe_indices):
    """
    Interpolate frames between keyframes using linear blending.
    
    Args:
        original_frames (list): List of all original frames
        stylized_keyframes (list): List of stylized keyframes
        keyframe_indices (list): Indices of keyframes in original_frames
        
    Returns:
        list: Complete list of frames with interpolated frames between keyframes
    """
    full_frames = []
    
    # Initialize with the correct total number of frames
    for _ in range(len(original_frames)):
        full_frames.append(None)
    
    # Set the known keyframes
    for i, idx in enumerate(keyframe_indices):
        full_frames[idx] = stylized_keyframes[i]
    
    # Interpolate intermediate frames
    for i in range(len(keyframe_indices) - 1):
        start_idx = keyframe_indices[i]
        end_idx = keyframe_indices[i + 1]
        start_frame = stylized_keyframes[i]
        end_frame = stylized_keyframes[i + 1]
        
        # Linear interpolation between keyframes
        for j in range(start_idx + 1, end_idx):
            alpha = (j - start_idx) / (end_idx - start_idx)
            interpolated = start_frame * (1 - alpha) + end_frame * alpha
            full_frames[j] = interpolated.astype(np.uint8)
    
    return full_frames