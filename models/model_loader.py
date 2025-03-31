"""
Model loading utilities for Video2Style.

This module provides functions to load pre-trained ControlNet and Stable Diffusion models
from Hugging Face Hub.
"""
from diffusers import ControlNetModel, StableDiffusionPipeline

def load_controlnet(model_id):
    """
    Load a ControlNet model from the Hugging Face Hub.
    
    Args:
        model_id (str): The model identifier on Hugging Face Hub.
        
    Returns:
        ControlNetModel: The loaded ControlNet model.
    """
    return ControlNetModel.from_pretrained(model_id)

def load_stable_diffusion(model_id):
    """
    Load a Stable Diffusion pipeline from the Hugging Face Hub.
    
    Args:
        model_id (str): The model identifier on Hugging Face Hub.
        
    Returns:
        StableDiffusionPipeline: The loaded Stable Diffusion pipeline.
    """
    return StableDiffusionPipeline.from_pretrained(model_id)