"""
Sketch stylization module.

This module provides functionality for transforming regular video frames
into sketch-style artistic renderings using Stable Diffusion and ControlNet.
"""
import torch
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline
from PIL import Image
from models.model_loader import load_controlnet
from controlnet_aux import CannyDetector

class SketchStylizer:
    """
    Stylizer for sketch-style video frame generation.
    
    This class uses Canny edge detection as a conditioning method
    for the ControlNet model to generate sketch-styled frames.
    """
    def __init__(self, device="cpu"):
        """
        Initialize the SketchStylizer.
        
        Args:
            device (str): Device to run inference on ("cpu" or "cuda")
        """
        # Initialize Canny edge detector for sketch style conditioning
        self.canny_detector = CannyDetector.from_pretrained("lllyasviel/Annotators")
        
        # Load ControlNet for Canny edge control
        controlnet = load_controlnet("lllyasviel/sd-controlnet-canny")
        
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
        Generate an edge map as control image for sketch style.
        
        Args:
            frame (np.ndarray or PIL.Image): Input frame
            
        Returns:
            PIL.Image: Edge map for ControlNet conditioning
        """
        # Convert frame to PIL Image if needed
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(np.uint8(frame))
        
        # Generate Canny edge map with appropriate thresholds
        edge_map = self.canny_detector(frame, low_threshold=100, high_threshold=200)
        return edge_map

    def stylize(self, frame, control_image, prompt):
        """
        Stylize a frame into sketch style.
        
        Args:
            frame (np.ndarray or PIL.Image): Input frame
            control_image (PIL.Image): Control image (edge map)
            prompt (str): Text prompt to guide the generation
            
        Returns:
            np.ndarray: Stylized frame as a NumPy array
        """
        if not prompt:
            prompt = "pencil sketch, detailed shading, professional drawing"
            
        # Convert frame to PIL Image if needed
        if isinstance(frame, np.ndarray):
            frame_pil = Image.fromarray(np.uint8(frame))
        else:
            frame_pil = frame
            
        # Run the pipeline with edge conditioning
        stylized = self.pipe(
            prompt,
            image=control_image,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=7.5,
            negative_prompt="low quality, blurry, worst quality"
        ).images[0]
        
        return np.array(stylized)