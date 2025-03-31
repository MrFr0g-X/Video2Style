"""
Cinematic stylization module.

This module provides functionality for transforming regular video frames
into cinematic-style artistic renderings using Stable Diffusion and ControlNet.
"""
import torch
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline
from PIL import Image
from models.model_loader import load_controlnet
from controlnet_aux import HEDdetector

class CinematicStylizer:
    """
    Stylizer for cinematic-style video frame generation.
    
    This class uses HED edge detection as a conditioning method
    for the ControlNet model to generate cinematic-styled frames.
    """
    def __init__(self, device="cpu"):
        """
        Initialize the CinematicStylizer.
        
        Args:
            device (str): Device to run inference on ("cpu" or "cuda")
        """
        # Initialize HED edge detector for cinematic style conditioning
        self.hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
        
        # Load ControlNet for HED edge control
        controlnet = load_controlnet("lllyasviel/sd-controlnet-hed")
        
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
        Generate an HED edge map as control image for cinematic style.
        
        Args:
            frame (np.ndarray or PIL.Image): Input frame
            
        Returns:
            PIL.Image: HED edge map for ControlNet conditioning
        """
        # Convert frame to PIL Image if needed
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(np.uint8(frame))
        
        # Generate HED edge map
        hed_map = self.hed_detector(frame)
        return hed_map

    def stylize(self, frame, control_image, prompt):
        """
        Stylize a frame into cinematic style.
        
        Args:
            frame (np.ndarray or PIL.Image): Input frame
            control_image (PIL.Image): Control image (HED edge map)
            prompt (str): Text prompt to guide the generation
            
        Returns:
            np.ndarray: Stylized frame as a NumPy array
        """
        if not prompt:
            prompt = "cinematic scene, professional cinematography, movie still, high quality"
            
        # Convert frame to PIL Image if needed
        if isinstance(frame, np.ndarray):
            frame_pil = Image.fromarray(np.uint8(frame))
        else:
            frame_pil = frame
            
        # Run the pipeline with HED edge conditioning
        stylized = self.pipe(
            prompt,
            image=control_image,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=7.5,
            negative_prompt="low quality, blurry, worst quality"
        ).images[0]
        
        return np.array(stylized)