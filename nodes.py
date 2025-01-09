import os
import torch
import random
import torchaudio
import folder_paths
import numpy as np
import platform
import subprocess
import sys
import importlib.util
import importlib.machinery
import argparse
from omegaconf import OmegaConf
from PIL import Image
import shutil
import cv2
from typing import Optional, Dict, Any

def get_video_bitrate(video_path):
    """Get bitrate of video file using ffprobe"""
    cmd = [
        'ffprobe', 
        '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=bit_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    try:
        output = subprocess.check_output(cmd).decode().strip()
        return int(output) // 1000 if output else 12000
    except:
        return 12000

def write_video_high_quality(
    filename: str,
    video_array: torch.Tensor,
    fps: float,
    video_codec: str = "libx264",
    options: Optional[Dict[str, Any]] = None,
    audio_array: Optional[torch.Tensor] = None,
    audio_fps: Optional[float] = None,
    audio_codec: Optional[str] = None,
    audio_options: Optional[Dict[str, Any]] = None,
) -> None:
    temp_dir = os.path.dirname(filename)
    temp_frames_dir = os.path.join(temp_dir, "temp_frames")
    os.makedirs(temp_frames_dir, exist_ok=True)

    try:
        # Ensure video_array is on CPU and in the right format
        if isinstance(video_array, torch.Tensor):
            video_array = video_array.cpu()
            if video_array.shape[-1] != 3:  # If not in HWC format
                video_array = video_array.permute(0, 2, 3, 1)  # BCHW -> BHWC

        for i, frame in enumerate(video_array):
            # Convert to numpy and ensure in range [0, 255]
            frame_np = frame.numpy()
            if frame_np.dtype != np.uint8:
                frame_np = (frame_np * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(temp_frames_dir, f"frame_{i:04d}.png"), frame_bgr)

        temp_frames_dir = os.path.normpath(temp_frames_dir).replace('\\', '/')
        filename = os.path.normpath(filename).replace('\\', '/')

        command = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", os.path.join(temp_frames_dir, "frame_%04d.png"),
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "5",
            "-pix_fmt", "yuv420p",
            filename
        ]
        
        subprocess.run(command, check=True)
    finally:
        if os.path.exists(temp_frames_dir):
            shutil.rmtree(temp_frames_dir)

def import_inference_script(script_path):
    """Import a Python file as a module using its file path."""
    if not os.path.exists(script_path):
        raise ImportError(f"Script not found: {script_path}")

    module_name = "latentsync_inference"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None:
        raise ImportError(f"Failed to create module spec for {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise ImportError(f"Failed to execute module: {str(e)}")

    return module

def check_ffmpeg():
    try:
        if platform.system() == "Windows":
            # Check if ffmpeg exists in PATH
            ffmpeg_path = shutil.which("ffmpeg.exe")
            if ffmpeg_path is None:
                # Look for ffmpeg in common locations
                possible_paths = [
                    os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "ffmpeg", "bin"),
                    os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "ffmpeg", "bin"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "bin"),
                ]
                for path in possible_paths:
                    if os.path.exists(os.path.join(path, "ffmpeg.exe")):
                        # Add to PATH
                        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                        return True
                print("FFmpeg not found. Please install FFmpeg and add it to PATH")
                return False
            return True
        else:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg not found. Please install FFmpeg")
        return False

def check_and_install_dependencies():
    if not check_ffmpeg():
        raise RuntimeError("FFmpeg is required but not found")

    required_packages = [
        'omegaconf',
        'pytorch_lightning',
        'transformers',
        'accelerate',
        'huggingface_hub',
        'einops',
        'diffusers'
    ]

    def is_package_installed(package_name):
        return importlib.util.find_spec(package_name) is not None

    def install_package(package):
        python_exe = sys.executable
        try:
            subprocess.check_call([python_exe, '-m', 'pip', 'install', package],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {str(e)}")
            raise RuntimeError(f"Failed to install required package: {package}")

    for package in required_packages:
        if not is_package_installed(package):
            print(f"Installing required package: {package}")
            try:
                install_package(package)
            except Exception as e:
                print(f"Warning: Failed to install {package}: {str(e)}")
                raise

def normalize_path(path):
    """Normalize path to handle spaces and special characters"""
    return os.path.normpath(path).replace('\\', '/')

def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir
    
def save_and_reload_frames(frames, temp_dir):
    final_frames = []
    for frame in frames:
        # Convert to proper range (0-1)
        frame = frame.float() / max(frame.max(), 1.0)  
        # Ensure CHW format
        if frame.shape[0] != 3:
            frame = frame.permute(2, 0, 1)
        final_frames.append(frame)
    
    stacked = torch.stack(final_frames)
    print(f"Stacked min/max: {stacked.min()}, {stacked.max()}")
    return stacked.to(device='cpu', dtype=torch.float32)

def setup_models():
    cur_dir = get_ext_dir()
    ckpt_dir = os.path.join(cur_dir, "checkpoints")
    whisper_dir = os.path.join(ckpt_dir, "whisper")
    
    # Create directories if they don't exist
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(whisper_dir, exist_ok=True)
    
    unet_path = os.path.join(ckpt_dir, "latentsync_unet.pt")
    whisper_path = os.path.join(whisper_dir, "tiny.pt")
    
    if not (os.path.exists(unet_path) and os.path.exists(whisper_path)):
        print("Downloading required model checkpoints... This may take a while.")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="chunyu-li/LatentSync",
                            allow_patterns=["latentsync_unet.pt", "whisper/tiny.pt"],
                            local_dir=ckpt_dir, local_dir_use_symlinks=False)
            print("Model checkpoints downloaded successfully!")
        except Exception as e:
            print(f"Error downloading models: {str(e)}")
            print("\nPlease download models manually:")
            print("1. Visit: https://huggingface.co/chunyu-li/LatentSync")
            print("2. Download: latentsync_unet.pt and whisper/tiny.pt")
            print(f"3. Place them in: {ckpt_dir}")
            print(f"   with whisper/tiny.pt in: {whisper_dir}")
            raise RuntimeError("Model download failed. See instructions above.")

class LatentSyncNode:
    def __init__(self):
        check_and_install_dependencies()
        setup_models()


    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",),
                    "audio": ("AUDIO", ),
                    "seed": ("INT", {"default": 1247}),
                 },}

    CATEGORY = "LatentSyncNode"

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images", )
    FUNCTION = "inference"

    def inference(self, images, audio, seed):
        """Performs the main inference process to synchronize lip movements with the audio."""
        cur_dir = get_ext_dir() # Get the current directory
        ckpt_dir = os.path.join(cur_dir, "checkpoints") # Get the path for the checkpoints
        output_dir = folder_paths.get_output_directory() # Get the path for comfyUI output directory
        temp_dir = os.path.join(output_dir, "temp_frames") # Get the path for the temporary directory that is used to store frames for video creation
        os.makedirs(output_dir, exist_ok=True) # Create the output directory, if it does not exist
        os.makedirs(temp_dir, exist_ok=True)  # Create a temporary directory, if it does not exist

        # Create a temporary video file from the input frames
        output_name = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5)) # Generates random filename of 5 chars
        temp_video_path = os.path.join(output_dir, f"temp_{output_name}.mp4") # Path for temp video file
        output_video_path = os.path.join(output_dir, f"latentsync_{output_name}_highres_audio_out.mp4") # Path for final output video

        # Save frames as temporary video
        import torchvision.io as io # Import the module to deal with reading and writing videos
        # Save frames as temporary video
        if isinstance(images, list):  # If the input images are a list
            frames = torch.stack(images)  # Stack the list of frames to a single tensor
        else:  # If the images input is not a list but an image
            frames = images
            print(f"Initial frame count: {frames.shape[0]}")
            if frames.shape[0] == 0:
                raise RuntimeError("No frames could be processed due to face detection failure.")

        frames = (frames * 255).byte()  # Convert frames to byte format (0-255 range)
        if len(frames.shape) == 3: # If it is a 3-dimensional tensor then add an extra dimention
            frames = frames.unsqueeze(0)
        print(f"Frame count before writing video: {frames.shape[0]}")

        if isinstance(frames, torch.Tensor): # Move the data to cpu
            frames = frames.cpu()
        try:
            write_video_high_quality(temp_video_path, frames, fps=25)
            # io.write_video(temp_video_path, frames, fps=25, video_codec='h264') # Save frames as a video
        # Handles a type error when using a newer version of `torchvision`
        except TypeError:
            # Fallback for newer versions
            import av
            container = av.open(temp_video_path, mode='w')
            stream = container.add_stream('h264', rate=25)
            stream.width = frames.shape[2]
            stream.height = frames.shape[1]
            
            for frame in frames:
                frame = av.VideoFrame.from_ndarray(frame.numpy(), format='rgb24')
                packet = stream.encode(frame)
                container.mux(packet)
            
            # Flush stream
            packet = stream.encode(None)
            container.mux(packet)
            container.close()
        video_path = normalize_path(temp_video_path) # Normalize the path for the video file to work on all OSs

        if not os.path.exists(ckpt_dir): # Checks if the ckpt_dir folder exists, and if it does not then download all the necessary model files
            print("Downloading model checkpoints... This may take a while.")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="chunyu-li/LatentSync",
                                    allow_patterns=["latentsync_unet.pt", "whisper/tiny.pt"],
                                    local_dir=ckpt_dir, local_dir_use_symlinks=False)
            print("Model checkpoints downloaded successfully!")

        inference_script_path = os.path.join(cur_dir, "scripts", "inference.py") # Get the path for the inference.py file
        unet_config_path = normalize_path(os.path.join(cur_dir, "configs", "unet", "second_stage.yaml")) # Get the path to unet settings file
        scheduler_config_path = normalize_path(os.path.join(cur_dir, "configs")) # Get the path to the scheduler settings file
        ckpt_path = normalize_path(os.path.join(ckpt_dir, "latentsync_unet.pt")) # Path to the UNet model checkpoint
        whisper_ckpt_path = normalize_path(os.path.join(ckpt_dir, "whisper", "tiny.pt")) # Path to the whisper model checkpoint

        # resample audio to 16k hz and save to wav
        waveform = audio["waveform"] # Get the audio waveform data
        sample_rate = audio["sample_rate"] # Get the audio sample rate

        if waveform.dim() == 3: # Check if the audio is a tensor with dimension of 3, such as [channels, sample]
            waveform = waveform.squeeze(0) # Removes the channel dimension if the audio is stereo

        if sample_rate != 16000: # Check if the audio sample rate is different than 16000
            new_sample_rate = 16000 # Set the new sample rate to 16000
            waveform_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)(waveform) # Resample audio to 16000hz
            waveform, sample_rate = waveform_16k, new_sample_rate # Update the waveform and sample_rate variables

        audio_path = normalize_path(os.path.join(output_dir, f"latentsync_{output_name}_audio.wav")) # Path for the output audio file
        torchaudio.save(audio_path, waveform, sample_rate)  # Save audio to a .wav file

        print(f"Using video path: {video_path}") # Print out path to video being processed
        print(f"Video file exists: {os.path.exists(video_path)}") # Check if the video file exists
        print(f"Video file size: {os.path.getsize(video_path)} bytes")  # Print video file size

        assert os.path.exists(video_path), f"video_path not exists: {video_path}" # Exit if video is not found at the given path
        assert os.path.exists(audio_path), f"audio_path not exists: {audio_path}" # Exit if the audio file is not found at the given path

        try:
            # Add the package root to Python path
            package_root = os.path.dirname(cur_dir)
            if package_root not in sys.path:
                sys.path.insert(0, package_root)
               
            # Add the current directory to Python path
            if cur_dir not in sys.path:
                sys.path.insert(0, cur_dir)

            # Import the inference module
            inference_module = import_inference_script(inference_script_path)
           
            # Create a Namespace object with the arguments
            args = argparse.Namespace(
                unet_config_path=unet_config_path,
                inference_ckpt_path=ckpt_path,
                video_path=video_path,
                audio_path=audio_path,
                video_out_path=output_video_path,
                seed=seed,
                scheduler_config_path=scheduler_config_path,
                whisper_ckpt_path=whisper_ckpt_path
            )
            print(f"inference_module args: {args}")
            # Load the config
            config = OmegaConf.load(unet_config_path)
            
            print(f"Enter inference_module")
            # Call main with both config and args
            inference_module.main(config, args)
            
            # Load the processed video back as frames
            processed_frames = io.read_video(output_video_path, pts_unit='sec')[0]  # [T, H, W, C]
            print(f"Frame count after reading video: {processed_frames.shape[0]}")
            
            # Process frames following wav2lip.py pattern
            out_tensor_list = []
            for frame in processed_frames:
                # Convert to numpy and ensure correct format
                frame = frame.numpy()
                
                # Convert frame to float32 and normalize
                frame = frame.astype(np.float32) / 255.0
                
                # Convert back to tensor
                frame = torch.from_numpy(frame)
                
                # Ensure we have 3 channels
                if len(frame.shape) == 2:  # If grayscale
                    frame = frame.unsqueeze(2).repeat(1, 1, 3)
                elif frame.shape[2] == 4:  # If RGBA
                    frame = frame[:, :, :3]
                
                # Change to [C, H, W] format
                frame = frame.permute(2, 0, 1)
                
                out_tensor_list.append(frame)

            processed_frames = io.read_video(output_video_path, pts_unit='sec')[0]  # [T, H, W, C]
            processed_frames = processed_frames.float() / 255.0
            print(f"Frame count after normalization: {processed_frames.shape[0]}")

            # Fix dimensions for VideoCombine compatibility
            if len(processed_frames.shape) == 3:  
                processed_frames = processed_frames.unsqueeze(0)
            if processed_frames.shape[0] == 1 and len(processed_frames.shape) == 4:
                processed_frames = processed_frames.squeeze(0)
            if processed_frames.shape[0] == 3:  # If in CHW format
                processed_frames = processed_frames.permute(1, 2, 0)  # Convert to HWC
            if processed_frames.shape[-1] == 4:  # If RGBA
                processed_frames = processed_frames[..., :3]

            print(f"Final frame count: {processed_frames.shape[0]}")

            print(f"Final shape: {processed_frames.shape}")

            # Clean up
            # if os.path.exists(temp_video_path):
            #     os.remove(temp_video_path)
            # if os.path.exists(output_video_path):
            #     os.remove(output_video_path)
                
            shutil.rmtree(temp_dir, ignore_errors=True)
                
        # If any of the above steps failed, remove temporary files
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        return (processed_frames,) # Returns the processed image frames

class VideoLengthAdjuster:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",),
                    "audio": ("AUDIO", ),
                    "mode": (["normal", "pingpong", "loop_to_audio"], {"default": "normal"}),
                 },}

    CATEGORY = "LatentSyncNode"
    RETURN_TYPES = ("IMAGE", "AUDIO")  # Add AUDIO back
    RETURN_NAMES = ("images", "audio")  # Add audio back
    FUNCTION = "adjust"

    def adjust(self, images, audio, mode):
        if isinstance(images, list):
            frames = len(images)
        else:
            frames = images.shape[0]
            images = [images[i] for i in range(frames)]
        
        # Handle images as before
        video_duration = frames / 25
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
            
        # Add 1 second of silence to audio
        silence_samples = sample_rate  # 1 second worth of samples
        silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
        padded_waveform = torch.cat([waveform, silence], dim=1)
        
        # Create modified audio dict
        padded_audio = {
            "waveform": padded_waveform,
            "sample_rate": sample_rate
        }

        audio_samples = padded_waveform.shape[1]
        audio_duration = audio_samples / sample_rate

        print(f"Video Length Adjuster Info:")
        print(f"Video frames: {frames}")
        print(f"Video duration: {video_duration:.2f} seconds")
        print(f"Audio samples: {audio_samples}")
        print(f"Audio sample rate: {sample_rate}")
        print(f"Audio duration (with padding): {audio_duration:.2f} seconds")
        
        if mode == "pingpong":
            backward_frames = images[-2:0:-1]
            images = images + backward_frames
            
        if mode in ["pingpong", "loop_to_audio"] and audio_duration > video_duration:
            frames_needed = int(audio_duration * 25)
            complete_sequence = images
            while len(images) < frames_needed:
                images.extend(complete_sequence[:frames_needed-len(images)])

        return (torch.stack(images), padded_audio)

NODE_CLASS_MAPPINGS = {
    "D_LatentSyncNode": LatentSyncNode,
    "D_VideoLengthAdjuster": VideoLengthAdjuster,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "D_LatentSyncNode": "LatentSync Node",
    "D_VideoLengthAdjuster": "Video Length Adjuster",
}

