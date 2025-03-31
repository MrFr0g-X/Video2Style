#!/usr/bin/env python3
"""
Video2Style - Stylized Video-to-Video Generation.

This module is the main entry point for the Video2Style application,
which transforms regular videos into artistic styles using
ControlNet and Stable Diffusion.

Examples:
    Basic usage with anime style:
        $ python main.py --input sample_video.mp4 --output anime_video.mp4 --style anime

    Adding a custom prompt:
        $ python main.py --input sample_video.mp4 --output anime_video.mp4 --style anime --prompt "Studio Ghibli style"

    CPU-optimized processing with frame skipping and resolution control:
        $ python main.py --input sample_video.mp4 --output anime_video.mp4 --style anime --frame_skip 10 --resolution 512,384
"""

import argparse
from inference import run_inference

def main():
    """Entry point for the Video2Style pipeline."""
    parser = argparse.ArgumentParser(description="Video2Style: Stylized Video-to-Video Generation")
    parser.add_argument("--input", required=True, help="Path to input video file (.mp4 or .avi)")
    parser.add_argument("--output", required=True, help="Path to output stylized video file")
    parser.add_argument("--style", choices=["anime", "sketch", "cinematic"], default="anime", 
                        help="Style to apply (default: anime)")
    parser.add_argument("--prompt", default="", help="Optional textual prompt for stylization")
    parser.add_argument("--frame_skip", type=int, default=5, 
                        help="Process 1 frame every N frames to speed up (default: 5)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", 
                        help="Device to run on (default: cpu)")
    parser.add_argument("--resolution", type=str, default=None, 
                        help="Optional output resolution as 'width,height', e.g. '512,384'")
    
    args = parser.parse_args()
    
    # Parse resolution if provided
    resolution = None
    if args.resolution:
        try:
            width, height = map(int, args.resolution.split(','))
            resolution = (width, height)
        except ValueError:
            print("Invalid resolution format. Using original resolution.")
    
    print(f"Starting video stylization with {args.style} style on {args.device}...")
    if args.device == "cpu":
        print("CPU mode detected. Using optimized settings for CPU processing.")
        print(f"Processing 1 frame every {args.frame_skip} frames for efficiency.")
    
    run_inference(
        args.input, 
        args.output, 
        args.style, 
        args.prompt, 
        device=args.device,
        frame_skip=args.frame_skip,
        resolution=resolution
    )

if __name__ == "__main__":
    main()