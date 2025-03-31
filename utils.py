"""
Utility functions for video processing in Video2Style.

This module contains helper functions for extracting frames from videos,
creating videos from frames, and computing optical flow for temporal consistency.
"""
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip

def extract_frames(video_path, return_fps=False):
    """
    Extract frames from a video file.
    
    Args:
        video_path (str): Path to the input video file
        return_fps (bool): If True, return tuple of (frames, fps)
        
    Returns:
        list: List of frames as numpy arrays, or (frames, fps) if return_fps=True
    """
    video = VideoFileClip(video_path)
    frames = [frame for frame in video.iter_frames()]
    if return_fps:
        return frames, video.fps
    return frames

def create_video(frames, output_path, fps=24):
    """
    Create a video from a list of frames.
    
    Args:
        frames (list): List of numpy arrays (frames)
        output_path (str): Path to save the generated video
        fps (float): Frames per second for the output video
    """
    clip = ImageSequenceClip([np.uint8(frame) for frame in frames], fps=fps)
    clip.write_videofile(output_path, codec="libx264", audio=False)

def compute_optical_flow(prev_frame, curr_frame):
    """
    Compute optical flow between two frames.
    
    Handles frames of different sizes and color formats safely, returning
    a zero flow matrix as fallback if computation fails.
    
    Args:
        prev_frame (np.ndarray): Previous frame
        curr_frame (np.ndarray): Current frame
        
    Returns:
        np.ndarray: Optical flow matrix or zero flow matrix if computation fails
    """
    try:
        # Ensure both frames have the same size
        if prev_frame.shape[:2] != curr_frame.shape[:2]:
            h, w = min(prev_frame.shape[0], curr_frame.shape[0]), min(prev_frame.shape[1], curr_frame.shape[1])
            prev_frame = cv2.resize(prev_frame, (w, h))
            curr_frame = cv2.resize(curr_frame, (w, h))
        
        # Convert frames to grayscale
        if len(prev_frame.shape) == 3 and prev_frame.shape[2] == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        else:
            prev_gray = prev_frame.astype(np.uint8)
            
        if len(curr_frame.shape) == 3 and curr_frame.shape[2] == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        else:
            curr_gray = curr_frame.astype(np.uint8)
        
        # Ensure grayscale images are in correct format
        prev_gray = prev_gray.astype(np.uint8)
        curr_gray = curr_gray.astype(np.uint8)
        
        # Compute optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, 
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        return flow
    
    except Exception as e:
        print(f"Error in optical flow calculation: {e}")
        # Return zero flow as fallback
        h, w = prev_frame.shape[:2]
        return np.zeros((h, w, 2), dtype=np.float32)