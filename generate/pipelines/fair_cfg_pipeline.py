# fair_cfg_pipeline.py

import torch
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipelineOutput,
)


class FairCFGStableDiffusionPipeline(StableDiffusionPipeline):
    r"""
    Stable Diffusion pipeline with Fairness-aware CFG:

        w * eps_pos
        - (w - w_fair) * eps_neg
        + (1 - w_fair) * eps_null

    NOTE (moai_h100, 2026-08-16): the eps_pos / eps_null coefficients were changed from
    (1+w) / -w_fair to w / (1-w_fair). The released form satisfies

        fair(w)|a=0 = (1+w) eps_pos - w eps_null = stock(w+1)

    i.e. its w is "how much CFG to add on top of the conditional model" (no-CFG at w=0),
    while diffusers' convention is "how far to interpolate toward conditional" (no-CFG at
    w=1). Both are self-consistent, and on moai1 nothing was wrong: the CFG and \method
    rows of a block were both produced by this pipeline, so the offset cancelled. It only
    matters here because the grid must mean the same thing in every block -- the vanilla
    rows come from the stock pipeline, so a w of 2.5..12.5 there is not the 2.5..12.5 of
    the released form (which is effectively 3.5..13.5).

    With the change,

        fair(w)|a=0 = w eps_pos + (1-w) eps_null = stock(w)

    so a single w axis covers every block, the already-generated stock runs serve as the
    a=0 rows without regeneration, and the intervention term is untouched:

        fair(w) - fair(w)|a=0 = -(w - w_fair) (eps_neg - eps_null)

    which is still exactly zero at w = w_fair.
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        *,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        w_fair: float = 0.0,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        **kwargs,
    ):
        device = self._execution_device

        # -------------------------
        # 1. Encode prompts
        # -------------------------
        do_cfg = True

        prompt_embeds, _ = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=False,
        )

        # Match batch size: if prompt is a list, neg/null must also be lists of same length
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        if negative_prompt_embeds is not None:
            neg_embeds = negative_prompt_embeds
        else:
            neg_prompt = negative_prompt if negative_prompt is not None else ""
            if isinstance(neg_prompt, str):
                neg_prompt = [neg_prompt] * batch_size
            neg_embeds, _ = self.encode_prompt(
                prompt=neg_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=False,
            )

        null_embeds, _ = self.encode_prompt(
            prompt=[""] * batch_size,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=False,
        )

        prompt_embeds = torch.cat(
            [prompt_embeds, neg_embeds, null_embeds], dim=0
        )

        # -------------------------
        # 2. Prepare latents
        # -------------------------
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        if latents is None:
            latents = self.prepare_latents(
                batch_size=prompt_embeds.shape[0] // 3,
                num_channels_latents=self.unet.config.in_channels,
                height=height or self.unet.config.sample_size * self.vae_scale_factor,
                width=width or self.unet.config.sample_size * self.vae_scale_factor,
                dtype=prompt_embeds.dtype,
                device=device,
                generator=generator,
            )

        # -------------------------
        # 3. Denoising loop
        # -------------------------
        for t in self.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 3)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t
            )

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample

            eps_pos, eps_neg, eps_null = noise_pred.chunk(3)

            # ---- Fair CFG (핵심) ----
            noise_pred = (
                guidance_scale * eps_pos
                - (guidance_scale - w_fair) * eps_neg
                + (1.0 - w_fair) * eps_null
            )

            latents = self.scheduler.step(
                noise_pred, t, latents
            ).prev_sample

        # -------------------------
        # 4. Decode
        # -------------------------
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor,
            return_dict=False,
        )[0]

        image = self.image_processor.postprocess(
            image, output_type=output_type
        )

        if not return_dict:
            return image

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
