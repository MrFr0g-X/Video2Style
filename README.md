# Video2Style

![Video2Style Banner](https://i.imgur.com/G3vx3ed.jpeg)

Video2Style is an AI-powered video transformation tool that converts regular videos into artistic styles like anime, sketch, or cinematic using Stable Diffusion and ControlNet.

## 🎨 Features

- **Multiple Artistic Styles**: Transform videos into anime, sketch, or cinematic styles
- **Custom Prompts**: Guide the style generation with custom text prompts
- **CPU and GPU Support**: Optimized for both CPU and CUDA-enabled GPU systems
- **Frame Interpolation**: Smart interpolation between keyframes for smooth transitions
- **Memory Optimization**: Efficient processing that works on systems with limited resources
- **Temporal Consistency**: Methods to ensure consistent style across frames

## 📋 Requirements

- Python 3.8+
- PyTorch
- Diffusers
- ControlNet
- OpenCV
- MoviePy
- transformers

See [requirements.txt](requirements.txt) for full dependencies.

## 🚀 Installation

1. Clone the repository:
   ```
   git clone https://github.com/MrFr0g-X/Video2Style.git
   cd Video2Style
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) If you have a GPU, make sure to install the CUDA-enabled version of PyTorch.

## 🎮 Usage

### Basic Usage

```bash
python main.py --input your_video.mp4 --output stylized_video.mp4 --style anime
```

### With Custom Prompt

```bash
python main.py --input your_video.mp4 --output stylized_video.mp4 --style anime --prompt "Studio Ghibli fantasy landscape with magical elements"
```

### CPU Optimization

For slower systems or when running on CPU:

```bash
python main.py --input your_video.mp4 --output stylized_video.mp4 --style anime --frame_skip 10 --resolution 512,384
```

### All Options

```
usage: main.py [-h] --input INPUT --output OUTPUT [--style {anime,sketch,cinematic}] [--prompt PROMPT] [--frame_skip FRAME_SKIP] [--device {cpu,cuda}] [--resolution RESOLUTION]

Video2Style: Stylized Video-to-Video Generation

required arguments:
  --input INPUT         Path to input video file (.mp4 or .avi)
  --output OUTPUT       Path to output stylized video file

optional arguments:
  --style {anime,sketch,cinematic}
                        Style to apply (default: anime)
  --prompt PROMPT       Optional textual prompt for stylization
  --frame_skip FRAME_SKIP
                        Process 1 frame every N frames to speed up (default: 5)
  --device {cpu,cuda}   Device to run on (default: cpu)
  --resolution RESOLUTION
                        Optional output resolution as 'width,height', e.g. '512,384'
```

## 📊 Examples

| Original | Anime Style | Sketch Style | Cinematic Style |
|----------|-------------|--------------|-----------------|
| ![Original](https://i.imgur.com/Rgy5sbA.png) | ![Anime](https://i.imgur.com/NH3rWiA.png) | ![Sketch](https://i.imgur.com/EJyMd1F.png) | ![Cinematic](https://i.imgur.com/lUe2QH8.png) |
| Original video | *"anime style, Studio Ghibli, detailed, vibrant colors"* | *"pencil sketch, detailed shading, professional drawing"* | *"cinematic scene, professional cinematography, movie still"* |

## 📝 How It Works

Video2Style uses different ControlNet models to transform videos:

1. **Anime Style**: Uses MiDaS depth estimation for anime-style conditioning
2. **Sketch Style**: Uses Canny edge detection for sketch-style conditioning
3. **Cinematic Style**: Uses HED edge detection for cinematic-style conditioning

The process works by:
1. Extracting frames from the input video
2. Processing key frames (every N frames for efficiency)
3. Generating control images (depth maps, edge maps) for each keyframe
4. Applying Stable Diffusion with ControlNet conditioning
5. Interpolating between stylized keyframes
6. Assembling the final video

## 🛠️ Performance Tips

- **Frame Skip**: Higher values are faster but may reduce temporal consistency
- **Resolution**: Lower resolutions process faster but with less detail
- **Prompts**: Be specific in your prompts for better style control
- **Memory Usage**: For low memory systems, increase frame skip and lower resolution

## 👨‍💻 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) by CompVis
- [ControlNet](https://github.com/lllyasviel/ControlNet) by lllyasviel
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
