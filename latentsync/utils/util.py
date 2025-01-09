import os
import imageio
import numpy as np
import json
from typing import Union, Optional, Dict, Any
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.distributed as dist
from torchvision import transforms
import av
from tqdm import tqdm
from einops import rearrange
import cv2
from decord import AudioReader, VideoReader
import shutil
import subprocess


# Machine epsilon for a float32 (single precision)
eps = np.finfo(np.float32).eps


def read_json(filepath: str):
    with open(filepath) as f:
        json_dict = json.load(f)
    return json_dict


def read_video(video_path: str, change_fps=False, use_decord=True):
    if change_fps:
          # Create temp directory next to the video file
        temp_dir = os.path.join(os.path.dirname(video_path), "temp")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Normalize paths for Windows
        video_path = os.path.normpath(video_path).replace('\\', '/')
        temp_video = os.path.join(temp_dir, 'video.mp4').replace('\\', '/')
        
        # Use proper path quoting for FFmpeg
        command = f'ffmpeg -loglevel error -y -nostdin -i "{video_path}" -r 25 -crf 2 "{temp_video}"'
        
        print(f"Running FFmpeg command: {command}")
        subprocess.run(command, shell=True, check=True)
        target_video_path = temp_video
    else:
        target_video_path = video_path

    print(f"Reading video from: {target_video_path}")
    
    if use_decord:
        return read_video_decord(target_video_path)
    else:
        return read_video_cv2(target_video_path)


def read_video_decord(video_path: str):
    vr = VideoReader(video_path)
    video_frames = vr[:].asnumpy()
    vr.seek(0)
    return video_frames


def read_video_cv2(video_path: str):
    # Open the video file
    video_path = os.path.normpath(video_path)
    print(f"Opening video with CV2: {video_path}")
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video at path: {video_path}")
        return np.array([])

    frames = []
    frame_count = 0

    while True:
        # Read a frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_count += 1

    # Release the video capture object
    cap.release()
    
    print(f"Successfully read {frame_count} frames from video")
    return np.array(frames)

def read_audio(audio_path: str, audio_sample_rate: int = 16000):
    if audio_path is None:
        raise ValueError("Audio path is required.")
    ar = AudioReader(audio_path, sample_rate=audio_sample_rate, mono=True)

    # To access the audio samples
    audio_samples = torch.from_numpy(ar[:].asnumpy())
    audio_samples = audio_samples.squeeze(0)

    return audio_samples


def get_video_bitrate(video_path: str) -> int:
    """
    Extracts the video bitrate using FFmpeg.

    Args:
        video_path (str): Path to the video file.

    Returns:
         int: The video bitrate in kbps.
    """
    try:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=bit_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        bitrate = int(result.stdout.strip()) // 1000  # Convert bits/s to kbps
        return bitrate
    except (subprocess.CalledProcessError, ValueError):
        print(f"Failed to get bitrate for video: {video_path}, defaulting to 12000")
        return 12000


def write_video(
    filename: str,
    video_array: Union[torch.Tensor, np.ndarray],
    fps: float,
    original_video_path: str = None,
    video_codec: str = "libx264",
    options: Optional[Dict[str, Any]] = None,
    audio_array: Optional[torch.Tensor] = None,
    audio_fps: Optional[float] = None,
    audio_codec: Optional[str] = None,
    audio_options: Optional[Dict[str, Any]] = None,
    crf: int = 5,
) -> None:
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file using FFmpeg.

    Args:
        filename (str): path where the video will be saved
        video_array (Tensor[T, H, W, C]): tensor containing the individual frames,
            as a uint8 tensor in [T, H, W, C] format
        fps (Number): video frames per second
         original_video_path (str, optional): Path to the original video to extract original bitrate.
        video_codec (str): the name of the video codec, i.e. "libx264", "h264", etc.
        options (Dict): dictionary containing options to be passed into the PyAV video stream
        audio_array (Tensor[C, N]): tensor containing the audio, where C is the number of channels
            and N is the number of samples
        audio_fps (Number): audio sample rate, typically 44100 or 48000
        audio_codec (str): the name of the audio codec, i.e. "mp3", "aac", etc.
        audio_options (Dict): dictionary containing options to be passed into the PyAV audio stream
    """
    if isinstance(video_array, torch.Tensor):
        height, width = video_array.shape[1], video_array.shape[2]
    elif isinstance(video_array, np.ndarray):
        height, width = video_array[0].shape[:2]
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray but got {type(video_array)}")
        
    temp_dir = os.path.dirname(filename)
    temp_frames_dir = os.path.join(temp_dir, "temp_frames")
    os.makedirs(temp_frames_dir, exist_ok=True)

    compression_params = [
        cv2.IMWRITE_PNG_COMPRESSION, 0,  # No compression for max quality
        cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_DEFAULT
    ]

    if isinstance(video_array, torch.Tensor):
        for i, frame in enumerate(video_array):
                if frame.is_cuda:
                    frame = frame.cpu().numpy()
                else:
                    frame = frame.numpy()
                frame = frame.astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(temp_frames_dir, f"frame_{i:04d}.png"), frame, compression_params)
    elif isinstance(video_array, np.ndarray):
          for i, frame in enumerate(video_array):
               frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
               cv2.imwrite(os.path.join(temp_frames_dir, f"frame_{i:04d}.png"), frame, compression_params)

    bitrate = 12000 
    
    temp_frames_dir = os.path.normpath(temp_frames_dir).replace('\\', '/')
    filename = os.path.normpath(filename).replace('\\', '/')
        
    command = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", os.path.join(temp_frames_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-preset", "veryslow",  # Highest quality preset
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        # "-b:v", f"{bitrate}k",  # Set bitrate
        filename,
    ]
        
    print(f"Running FFmpeg command: {' '.join(command)}")
    subprocess.run(command, shell=False, check=True)
    shutil.rmtree(temp_frames_dir)

def mux_audio_video(video_path: str, audio_path: str, output_path: str) -> None:
    """
    Muxes an audio and a video file into one file. This will avoid recoding the video stream.
        
    Args:
        video_path (str): path to the video file.
        audio_path (str): Path to the audio file.
        output_path (str): Path for the final video with audio muxed.
    """
    # Normalize paths for Windows
    video_path = os.path.normpath(video_path).replace('\\', '/')
    audio_path = os.path.normpath(audio_path).replace('\\', '/')
    output_path = os.path.normpath(output_path).replace('\\', '/')

    command = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-q:a",
            "0",
            output_path,
            ]
    print(f"Running FFmpeg command: {' '.join(command)}")
    subprocess.run(command, shell=False, check=True)

def init_dist(backend="nccl", **kwargs):
    """Initializes distributed environment."""
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available for training.")
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, **kwargs)

    return local_rank


def zero_rank_print(s):
    if dist.is_initialized() and dist.get_rank() == 0:
        print("### " + s)


def zero_rank_log(logger, message: str):
    if dist.is_initialized() and dist.get_rank() == 0:
        logger.info(message)


def make_audio_window(audio_embeddings: torch.Tensor, window_size: int):
    audio_window = []
    end_idx = audio_embeddings.shape[1] - window_size + 1
    for i in range(end_idx):
        audio_window.append(audio_embeddings[:, i : i + window_size, :])
    audio_window = torch.stack(audio_window)
    audio_window = rearrange(audio_window, "f b w d -> b f w d")
    return audio_window


def check_video_fps(video_path: str):
    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    if fps != 25:
        raise ValueError(f"Video FPS is not 25, it is {fps}. Please convert the video to 25 FPS.")


def tailor_tensor_to_length(tensor: torch.Tensor, length: int):
    if len(tensor) == length:
        return tensor
    elif len(tensor) > length:
        return tensor[:length]
    else:
        return torch.cat([tensor, tensor[-1].repeat(length - len(tensor))])


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c f h w -> f b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def interpolate_features(features: torch.Tensor, output_len: int) -> torch.Tensor:
    features = features.cpu().numpy()
    input_len, num_features = features.shape

    input_timesteps = np.linspace(0, 10, input_len)
    output_timesteps = np.linspace(0, 10, output_len)
    output_features = np.zeros((output_len, num_features))
    for feat in range(num_features):
        output_features[:, feat] = np.interp(output_timesteps, input_timesteps, features[:, feat])
    return torch.from_numpy(output_features)


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def reversed_forward(ddim_scheduler, pred_noise, timesteps, x_t):
    # Compute alphas, betas
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timesteps]
    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if ddim_scheduler.config.prediction_type == "epsilon":
        beta_prod_t = beta_prod_t[:, None, None, None, None]
        alpha_prod_t = alpha_prod_t[:, None, None, None, None]
        pred_original_sample = (x_t - beta_prod_t ** (0.5) * pred_noise) / alpha_prod_t ** (0.5)
    else:
        raise NotImplementedError("This prediction type is not implemented yet")

    # Clip "predicted x_0"
    if ddim_scheduler.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
    return pred_original_sample


def next_step(
    model_output: Union[torch.FloatTensor, np.ndarray],
    timestep: int,
    sample: Union[torch.FloatTensor, np.ndarray],
    ddim_scheduler,
):
    timestep, next_timestep = (
        min(timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999),
        timestep,
    )
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
    return next_sample

def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred

@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent

@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents

def plot_loss_chart(save_path: str, *args):
    # Creating the plot
    plt.figure()
    for loss_line in args:
        plt.plot(loss_line[1], loss_line[2], label=loss_line[0])
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()

    # Save the figure to a file
    plt.savefig(save_path)

    # Close the figure to free memory
    plt.close()

CRED = "\033[91m"
CEND = "\033[0m"

def red_text(text: str):
    return f"{CRED}{text}{CEND}"

log_loss = nn.BCELoss(reduction="none")

def cosine_loss(vision_embeds, audio_embeds, y):
    sims = nn.functional.cosine_similarity(vision_embeds, audio_embeds)
    # sims[sims!=sims] = 0 # remove nan
    # sims = sims.clamp(0, 1)
    loss = log_loss(sims.unsqueeze(1), y).squeeze()
    return loss

def save_image(image, save_path):
    # input size (C, H, W)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image * 255).to(torch.uint8)
    image = transforms.ToPILImage()(image)
    # Save the image copy
    image.save(save_path)
    # Close the image file
    image.close()


def gather_loss(loss, device):
    # Sum the local loss across all processes
    local_loss = loss.item()
    global_loss = torch.tensor(local_loss, dtype=torch.float32).to(device)
    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)

    # Calculate the average loss across all processes
    global_average_loss = global_loss.item() / dist.get_world_size()
    return global_average_loss


def gather_video_paths_recursively(input_dir):
    print(f"Recursively gathering video paths of {input_dir} ...")
    paths = []
    gather_video_paths(input_dir, paths)
    return paths

def gather_video_paths(input_dir, paths):
    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".mp4"):
            filepath = os.path.join(input_dir, file)
            paths.append(filepath)
        elif os.path.isdir(os.path.join(input_dir, file)):
            gather_video_paths(os.path.join(input_dir, file), paths)

def count_video_time(video_path):
    video = cv2.VideoCapture(video_path)

    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    return frame_count / fps