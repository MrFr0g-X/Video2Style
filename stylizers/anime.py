"""
Anime stylization module.

This module provides functionality for transforming regular video frames
into anime-style artistic renderings using Stable Diffusion and ControlNet.
"""
import torch
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline
from PIL import Image
from models.model_loader import load_controlnet
from controlnet_aux import MidasDetector

class AnimeStylizer:
    """
    Stylizer for anime-style video frame generation.
    
    This class uses MiDaS depth estimation as a conditioning method
    for the ControlNet model to generate anime-styled frames.
    """
    def __init__(self, device="cpu"):
        """
        Initialize the AnimeStylizer.
        
        Args:
            device (str): Device to run inference on ("cpu" or "cuda")
        """
        # Use MiDaS depth estimation for anime style conditioning
        self.depth_estimator = MidasDetector.from_pretrained("lllyasviel/Annotators")
        
        # Load ControlNet for depth-based control
        controlnet = load_controlnet("lllyasviel/sd-controlnet-depth")
        
        # Load Stable Diffusion pipeline
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            controlnet=controlnet,
            safety_checker=None
        ).to(device)
        
        # Enable memory optimization
        self.pipe.enable_attention_slicing(1)
        
        # Configure inference steps based on device
        if device == "cpu":
            # Use fewer steps for CPU to improve processing time
            self.num_inference_steps = 10
            print("Running in CPU mode with optimized settings")
        else:
            self.num_inference_steps = 20

    def generate_control_image(self, frame):
        """
        Generate a depth map as control image for anime style.
        
        Args:
            frame (np.ndarray or PIL.Image): Input frame
            
        Returns:
            PIL.Image: Depth map for ControlNet conditioning
        """
        # Convert frame to PIL Image if needed
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(np.uint8(frame))
        
        # Generate depth map using MiDaS
        depth_map = self.depth_estimator(frame)
        return depth_map

    def stylize(self, frame, control_image, prompt):
        """
        Stylize a frame into anime style.
        
        Args:
            frame (np.ndarray or PIL.Image): Input frame
            control_image (PIL.Image): Control image (depth map)
            prompt (str): Text prompt to guide the generation
            
        Returns:
            np.ndarray: Stylized frame as a NumPy array
        """
        if not prompt:
            prompt = "anime style, Studio Ghibli, detailed, vibrant colors"
            
        # Convert frame to PIL Image if needed
        if isinstance(frame, np.ndarray):
            frame_pil = Image.fromarray(np.uint8(frame))
        else:
            frame_pil = frame
            
        # Run the pipeline with depth conditioning
        stylized = self.pipe(
            prompt,
            image=control_image,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=7.5,
            negative_prompt="low quality, bad anatomy, worst quality, low quality"
        ).images[0]
        
        return np.array(stylized)