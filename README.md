# ComfyUI-LatentSync-Node - High-Resolution Lip-Sync

## Inspired by Innovation:

`ComfyUI-LatentSync-Node` builds upon the groundwork laid by  [ComfyUI-LatentSyncWrapper](https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper), itself a brilliant implementation of the groundbreaking [LatentSync](https://github.com/bytedance/LatentSync) code, adapted for the world of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on Windows.  We've taken that inspiration and pushed it further to give you the power you crave with high resolution output.

## 👄 Unleash the Power of Speech: Your Dreams, Now in Sync with Reality!

Tired of boring A.I. characters that can't speak and lifeless A.I. videos? Prepare to witness a revolution in digital expression!  **ComfyUI-LatentSync-Node** is here to empower *you* to bring your wildest creative visions to life, right on your local machine.

This isn't just another node; it's a gateway to **perfectly synchronized lip movements**, breathing life into any video you create within the powerful ComfyUI environment.  Using ByteDance's cutting-edge LatentSync model, you can effortlessly make anyone say anything, with uncanny accuracy. 

## What Awaits You:

*   **Effortless Lip-Sync Magic:** Seamlessly match the lip movements of your videos to any audio input. 
*   **High-Resolution Brilliance:**  Experience jaw-dropping, high-resolution lip-sync results that will elevate your projects to a new level.
*   **Unleash Your Inner Director:**  Craft scenarios where any character can deliver your dialogue with lifelike precision.
*   **Voice Your Vision:** Use your own voice for personalized narratives, or explore the endless possibilities of voice cloning with [F4-TTS](https://github.com/SWivid/F5-TTS).  We highly recommend using [Pinokio](https://pinokio.computer/) to set up F5-TTS with ease.

## Why Choose ComfyUI-LatentSync-Node?

Imagine creating:

*   **Dynamic characters** that express the full range of human emotions.
*   **Personalized videos** where your own voice is brought to life in stunning visuals.
*   **Storytelling experiences** that push the boundaries of what's possible.
*   **High Resolution Output** that give you the power to create clear lip-Sync videos.

This isn't just about syncing lips; it's about **unlocking a new dimension of creative expression**. Stop dreaming about what could be and start creating the impossible.  

**Ready to transform your projects? Dive into `ComfyUI-LatentSync-Node` today and let your voice be heard!**

![Screenshot 2025-01-02 210507](https://github.com/user-attachments/assets/df4c83a9-d170-4eb2-b406-38fb7a93c6aa)


https://github.com/user-attachments/assets/49c40cf4-5db1-46c5-99a4-7fbb2031c907


## Prerequisites

# You must be running ComfyUI in Python 3.8-3.11 to use ComfyUI-LatentSync-Node as it uses mediapipe 

Before installing this node, you must install the following in order:

**What Your Computer Needs**: 
1. **NVIDIA Graphics Card**: At least 8GB memory (newer models work best).  
2. **[CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)** (fancy software to make it run faster).  
3.**Windows 10 or 11**.  
4.**16GB RAM** (for best results).  
5.  [visual studio 2022 runtimes (windows)](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
6. [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed and working with Python 3.8-3.11
7. Python 3.8-3.11 (mediapipe is not yet compatible with Python 3.12)
8. FFmpeg installed on your system:
   - Windows: Download from [here](https://github.com/BtbN/FFmpeg-Builds/releases) and add to system PATH

9. If you get PYTHONPATH errors:
   - Make sure Python is in your system PATH
   - Try running ComfyUI as administrator
     
## Installation

Only proceed with installation after confirming all prerequisites are installed and working.

1. Clone this repository into your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper.git
cd ComfyUI-LatentSyncWrapper
pip install -r requirements.txt
```

## Required Dependencies
```
diffusers
transformers
huggingface-hub
omegaconf
einops
opencv-python
mediapipe>=0.10.8
face-alignment
decord
ffmpeg-python
safetensors
soundfile
```
## Model Setup

The models can be obtained in two ways:

### Option 1: Automatic Download (First Run)
The node will attempt to automatically download required model files from HuggingFace on first use.
If automatic download fails, use Option 2.

### Option 2: Manual Download
1. Visit the HuggingFace repo: https://huggingface.co/chunyu-li/LatentSync
2. Download these files:
   - `latentsync_unet.pt`
   - `whisper/tiny.pt`
3. Place them in the following structure:
```bash
ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints/
├── latentsync_unet.pt
└── whisper/
    └── tiny.pt
```
## Usage

1. Select an input video file
2. Load an audio file using ComfyUI audio loader
3. (Optional) Set a seed value for reproducible results
4. Connect to the LatentSync node
5. Run the workflow

The processed video will be saved in ComfyUI's output directory.

### Node Parameters:
- `video_path`: Path to input video file
- `audio`: Audio input from AceNodes audio loader
- `seed`: Random seed for reproducible results (default: 1247)


## Known Limitations

- Works best with clear, frontal face videos
- Currently does not support anime/cartoon faces
- Video should be at 25 FPS (will be automatically converted)
- Face should be visible throughout the video

## How A.I. Face Swap Works and Key to Understanding A.I. face Rotations and its Limits (click image to watch video)

[![Roop Face Tracking](https://i.ytimg.com/vi/BzTqrIm69Ws/maxresdefault.jpg)](https://youtu.be/BzTqrIm69Ws?si=C4t7jL6CJ9JvdgX0)

### click image to watch video [Roop Deep Fake Course](https://youtu.be/BzTqrIm69Ws?si=C4t7jL6CJ9JvdgX0)

### NEW - Video Length Adjuster Node
A complementary node that helps manage video length and synchronization with audio.

#### Features:
- Displays video and audio duration information
- Three modes of operation:
  - `normal`: Passes through video frames with added padding to prevent frame loss
  - `pingpong`: Creates a forward-backward loop of the video sequence
  - `loop_to_audio`: Extends video by repeating frames to match audio duration

#### Usage:
1. Place the Video Length Adjuster between your video input and the LatentSync node
2. Connect audio to both the Video Length Adjuster and Video Combine nodes
3. Select desired mode based on your needs:
   - Use `normal` for standard lip-sync
   - Use `pingpong` for back-and-forth animation
   - Use `loop_to_audio` to match longer audio durations

#### Example Workflow:
1. Load Video (Upload) → Video frames output
2. Load Audio → Audio output
3. Connect both to Video Length Adjuster
4. Video Length Adjuster → LatentSync Node
5. LatentSync Node + Original Audio → Video Combine

## Troubleshooting

### mediapipe Installation Issues
If you encounter mediapipe installation errors:
1. Ensure you're using Python 3.8-3.11 (Check with `python --version`)
2. If using Python 3.12, you'll need to downgrade to a compatible version
3. Try installing mediapipe separately first:
   ```bash
   pip install mediapipe>=0.10.8

## Credits

This is an unofficial implementation based on:
- [LatentSync](https://github.com/bytedance/LatentSync) by ByteDance Research
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
