import inspect
from typing import List, Optional, Union

import numpy as np
import torch

import PIL
from transformers import CLIPTextModel, CLIPTokenizer

from ...models import AutoencoderKL, UNet2DConditionModel
from ...pipeline_utils import DiffusionPipeline
from ...schedulers import DDIMScheduler, PNDMScheduler
from . import StableDiffusionPipelineOutput


def preprocess_image(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def preprocess_mask(mask):
    mask = mask.convert("L")
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize((w // 8, h // 8), resample=PIL.Image.NEAREST)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)
    return mask


def predict_xstart(scheduler, sample, t, noise):
    sqrt_alpha_prod = scheduler.match_shape(torch.sqrt(scheduler.alphas_cumprod[t]), sample)
    sqrt_one_minus_alpha_prod = scheduler.match_shape(torch.sqrt(1 - scheduler.alphas_cumprod[t]), sample)
    return (sample - sqrt_one_minus_alpha_prod * noise) / sqrt_alpha_prod


def undo_step(scheduler, sample, from_t, to_t, noise):
    # undo sample from x_t-1 to x_t
    sqrt_alpha_prod = scheduler.match_shape(torch.sqrt(scheduler.alphas_cumprod[from_t]), sample)
    sqrt_alpha_prod_to = scheduler.match_shape(torch.sqrt(scheduler.alphas_cumprod[to_t]), sample)
    sqrt_one_minus_alpha_prod = scheduler.match_shape(torch.sqrt(1 - scheduler.alphas_cumprod[from_t]), sample)
    sqrt_one_minus_alpha_prod_to = scheduler.match_shape(torch.sqrt(1 - scheduler.alphas_cumprod[to_t]), sample)

    noise_coefficient = sqrt_one_minus_alpha_prod_to * sqrt_alpha_prod - sqrt_one_minus_alpha_prod * sqrt_alpha_prod_to
    # get x_t by diffusion forward
    return sample * sqrt_alpha_prod_to / sqrt_alpha_prod + noise * noise_coefficient / sqrt_alpha_prod


def get_ts_index(t_T, jump_len, jump_n_sample):
    jumps = {}
    for j in range(0, t_T - jump_len, jump_len):
        jumps[j] = jump_n_sample - 1

    t = t_T
    ts = []

    while t >= 1:
        t = t - 1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_len):
                t = t + 1
                ts.append(t)
    return ts


def get_resample_timesteps(timesteps, resample_count, jump_len, jump_n_sample):
    total_ts = len(timesteps)
    ts = get_ts_index(resample_count, jump_len, jump_n_sample)
    ts = [t + total_ts - resample_count for t in ts]
    ts.extend(list(range(total_ts - resample_count))[::-1])
    ts = [timesteps[total_ts - i - 1] for i in ts]
    ts.append(-1)
    return ts


class StableDiffusionInpaintPipeline(DiffusionPipeline):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: Union[DDIMScheduler, PNDMScheduler],
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            init_image: Union[torch.FloatTensor, PIL.Image.Image],
            mask_image: Union[torch.FloatTensor, PIL.Image.Image],
            num_inference_steps: Optional[int] = 50,
            resample_count: Optional[int] = 20,
            resample_jump_len: Optional[int] = 1,
            resample_times: Optional[int] = 1,
                guidance_scale: Optional[float] = 7.5,
            eta: Optional[float] = 0.0,
            generator: Optional[torch.Generator] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
    ):

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # preprocess mask
        mask = preprocess_mask(mask_image).to(self.device)
        mask = torch.cat([mask] * batch_size)

        # preprocess image
        init_image = preprocess_image(init_image).to(self.device)

        # encode the init image into latents and scale the latents
        init_latent_dist = self.vae.encode(init_image.to(self.device)).latent_dist
        init_latents = init_latent_dist.sample(generator=generator)
        init_latents = torch.cat([init_latents] * batch_size)

        # adding noise to the masked areas depending on strength
        init_latents = 0.18215 * init_latents
        init_latents_orig = init_latents

        # check sizes
        if not mask.shape == init_latents.shape:
            raise ValueError("The mask and init_image should be the same size!")

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
            extra_step_kwargs["generator"] = generator

        latents = torch.randn(init_latents.shape, generator=generator, device=self.device)
        timesteps = get_resample_timesteps(self.scheduler.timesteps, resample_count, resample_jump_len, resample_times)
        timestep_pairs = list(zip(timesteps[:-1], timesteps[1:]))
        print(f"Progress count:{len(timestep_pairs)}")

        for t, t_next in self.progress_bar(timestep_pairs):
            # reverse if x_t -> x_t-1
            if t > t_next:
                # add t noise to origin latents and masking
                noise = torch.randn(init_latents_orig.shape, generator=generator, device=self.device)
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, t)
                latents = (init_latents_proper * mask) + (latents * (1 - mask))

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            else:
                noise = torch.randn(latents.shape, generator=generator, device=self.device)
                latents = undo_step(self.scheduler, latents, t, t_next, noise)

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return image, None

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
