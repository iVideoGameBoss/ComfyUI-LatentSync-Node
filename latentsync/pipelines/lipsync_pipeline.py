# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import os
import shutil
from typing import Callable, List, Optional, Union
from dataclasses import dataclass
import subprocess

import numpy as np
import torch
import torchvision

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import AutoProcessor, Wav2Vec2Model

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange

from ..models.unet import UNet3DConditionModel
from ..utils.image_processor import ImageProcessor
from ..utils.util import read_audio, write_video, read_video,mux_audio_video
import tqdm
import soundfile as sf

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_processor: AutoProcessor,
        audio_encoder: Wav2Vec2Model,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_processor=audio_processor,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.latent_space = vae is not None
        if self.latent_space:
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        if self.latent_space:
            latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
            latents = rearrange(latents, "b c f h w -> (b f) c h w")
            decoded_latents = self.vae.decode(latents).sample
        else:
            decoded_latents = rearrange(latents, "b c f h w -> (b f) c h w")
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        if self.latent_space:
            shape = (
                batch_size,
                num_channels_latents,
                1,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )
        else:
            shape = (
                batch_size,
                num_channels_latents,
                1,
                height,
                width,
            )
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        if self.latent_space:
            # resize the mask to latents shape as we concatenate the mask to the latents
            # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
            # and half precision
            mask = torch.nn.functional.interpolate(
                mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
            )
            masked_image = masked_image.to(device=device, dtype=dtype)

            # encode the mask image into latents space so we can concatenate it to the latents
            masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
            masked_image_latents = (
                masked_image_latents - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor
        else:
            masked_image_latents = masked_image.to(device=device, dtype=dtype)

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        if self.latent_space:
            image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
            image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            image_latents = images
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def recover_original_pixel_values(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Combine the pixel values
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def crop_audio_window(self, original_mel, start_index):
        start_idx = int(80.0 * (start_index / float(self.video_fps)))
        end_idx = start_idx + self.mel_window_length
        return original_mel[:, start_idx:end_idx].unsqueeze(0)

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, change_fps=False,use_decord=True)
        faces = []
        boxes = []
        affine_matrices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for i, frame in enumerate(tqdm.tqdm(video_frames)):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            if face is None:
                print(f"Warning: Skipping frame {i} due to face detection failure")
                continue
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        faces = torch.stack(faces)
        return faces, video_frames, boxes, affine_matrices

    def restore_video(self, faces, video_frames, boxes, affine_matrices):
        video_frames = video_frames[: faces.shape[0]]
        out_frames = []
        for index, face in enumerate(faces):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)
            face = rearrange(face, "c h w -> h w c")
            face = (face / 2 + 0.5).clamp(0, 1)
            face = (face * 255).to(torch.uint8).cpu().numpy()
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)

    @torch.no_grad()
    def __call__(
        self,
        output_name: str,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask: str = "fix_mask",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()

        # 0. Define call parameters
        batch_size = 1
        device = self._execution_device
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda")
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # Use decord to directly read video frames
        # original_video_frames = read_video(video_path, change_fps=False,use_decord=True)
        faces, original_video_frames, boxes, affine_matrices = self.affine_transform_video(video_path)
        
        print(f"Reading audio {audio_path}")
        audio_samples = read_audio(audio_path)

        # 1. Default height and width to unet
        if self.latent_space:
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        print(f"check_inputs height={height}, width={width}, callback_steps={callback_steps}") 
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        print(f"prepare extra step kwargs")
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        self.video_fps = video_fps

        if self.unet.add_audio_layer:
            whisper_feature = self.audio_encoder.audio2feat(audio_path)
            whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

            num_inferences = min(len(faces), len(whisper_chunks)) // num_frames
        else:
            num_inferences = len(faces) // num_frames

        synced_video_frames = []
        masked_video_frames = []

        save_affine_faces = False
        if save_affine_faces:
            pixel_values_faces = []
            masked_pixel_values_faces = []

        # Prepare latent variables
        if self.latent_space:
            num_channels_latents = self.vae.config.latent_channels
        else:
            num_channels_latents = 3
        
        print(f"prepare_latents")
        all_latents = self.prepare_latents(
            batch_size,
            num_frames * num_inferences,
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )

        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            if self.unet.add_audio_layer:
                # mel_overlap = torch.stack(mel_overlap_list[i * num_frames : (i + 1) * num_frames])
                # mel_overlap = mel_overlap.unsqueeze(0).to(device, dtype=weight_dtype)
                mel_overlap = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                mel_overlap = mel_overlap.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    empty_mel_overlap = torch.zeros_like(mel_overlap)
                    mel_overlap = torch.cat([empty_mel_overlap, mel_overlap])
            else:
                mel_overlap = None
            inference_video_frames = faces[i * num_frames : (i + 1) * num_frames]
            latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
            pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_video_frames, affine_transform=False
            )

            if save_affine_faces:
                pixel_values_faces.append(pixel_values)
                masked_pixel_values_faces.append(masked_pixel_values)

            # 7. Prepare mask latent variables
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # 8. Prepare image latents
            image_latents = self.prepare_image_latents(
                pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            # 9. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    # concat latents, mask, masked_image_latents in the channel dimension
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    latent_model_input = torch.cat(
                        [latent_model_input, mask_latents, masked_image_latents, image_latents], dim=1
                    )

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=mel_overlap).sample

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if j == len(timesteps) - 1 or (
                        (j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)

            # Recover the pixel values
            decoded_latents = self.decode_latents(latents)
            decoded_latents = self.recover_original_pixel_values(
                decoded_latents, pixel_values, 1 - masks, device, weight_dtype
            )
            synced_video_frames.append(decoded_latents)
            masked_video_frames.append(masked_pixel_values)
        
        synced_video_frames = self.restore_video(
            torch.cat(synced_video_frames), original_video_frames, boxes, affine_matrices
        )
        masked_video_frames = self.restore_video(
            torch.cat(masked_video_frames), original_video_frames, boxes, affine_matrices
        )

        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        temp_dir = "output"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        if save_affine_faces:
            pixel_values_faces = torch.cat(pixel_values_faces)
            masked_pixel_values_faces = torch.cat(masked_pixel_values_faces)

            pixel_values_faces = self.pixel_values_to_images(pixel_values_faces)
            masked_pixel_values_faces = self.pixel_values_to_images(masked_pixel_values_faces)

            write_video("affine_faces.mp4", pixel_values_faces, fps=25)
            write_video("masked_affine_faces.mp4", masked_pixel_values_faces, fps=25)

        temp_video_path = os.path.join(temp_dir, f"latentsync_{output_name}_high_video_sync_out.mp4")
        temp_audio_path = os.path.join(temp_dir, f"latentsync_{output_name}_high_audio_sync__out.wav")

        write_video(temp_video_path, synced_video_frames, fps=25, crf=2, original_video_path=video_path) # Pass original_video_path
        # write_video(video_mask_path, masked_video_frames, fps=25)

        sf.write(temp_audio_path, audio_samples, audio_sample_rate)

        mux_audio_video(temp_video_path, temp_audio_path, video_out_path)

        # Clean up the temp directory
        # if os.path.exists(temp_dir):
        #     shutil.rmtree(temp_dir)


        # command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v copy -c:a aac  -q:a 0 {video_out_path}"
        # subprocess.run(command, shell=True)