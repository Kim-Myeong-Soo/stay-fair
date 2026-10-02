#!/usr/bin/env python
# coding=utf-8
"""
Fair CFG SD1.5 with LoRA text encoder (cfg / cfg_gd).
Uses FairCFGStableDiffusionPipeline from pipelines/fair_cfg_pipeline.py.
"""

import argparse
import hashlib
import json
import math
import os
import sys

import torch
from tqdm.auto import tqdm

from diffusers import DPMSolverMultistepScheduler

# Import custom pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.fair_cfg_pipeline import FairCFGStableDiffusionPipeline


def encode_prompt(pipe, text):
    tok = pipe.tokenizer(
        text,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(pipe.device)
    with torch.no_grad():
        emb = pipe.text_encoder(tok.input_ids)[0]
    return emb.squeeze(0)


# occupation -> the alpha it was generated with; read by the save path so that
# candidates for one occupation never land on each other
ALPHA_BY_OCC = {}


def _load_gd_map(args):
    """{occupation: alpha} from --gd_json, or None."""
    path = getattr(args, "gd_json", None)
    if not path:
        return None
    with open(path, "r") as f:
        loaded = json.load(f)
    m = loaded.get("alpha", loaded)
    print(f"gd_json: {path} ({len(m)} occupations)")
    return m


def compute_gender_direction_embeddings(pipe, test_occs, gd_scale, gd_map=None):
    """Compute gender direction-based negative embeddings (shared across occupations)."""
    negative_embedding_dict = {}
    male_emb = encode_prompt(pipe, "male").unsqueeze(0)
    female_emb = encode_prompt(pipe, "female").unsqueeze(0)

    gender_dir = female_emb - male_emb
    gender_norm = gender_dir.norm(dim=-1, keepdim=True)
    gender_unit = gender_dir / (gender_norm + 1e-6)

    base_emb = encode_prompt(pipe, "").unsqueeze(0)
    for occ in test_occs:
        a = gd_scale if gd_map is None else float(gd_map.get(occ, gd_scale))
        ALPHA_BY_OCC[occ] = a
        negative_embedding_dict[occ] = base_emb + a * gender_unit

    return negative_embedding_dict


def setup_negative_prompt_config(args, pipe, test_occs):
    """Configure negative prompt/embedding based on test_option."""
    negative_prompt = None
    use_negative_embedding = False
    negative_embedding_dict = {}

    if args.test_option == 'cfg':
        print("Test option: cfg (Fair CFG, no extra negative)")

    elif args.test_option == 'cfg_gd':
        use_negative_embedding = True
        print(f"Test option: cfg_gd (gender direction, gd_scale={args.gd_scale})")
        negative_embedding_dict = compute_gender_direction_embeddings(
            pipe, test_occs, args.gd_scale, gd_map=_load_gd_map(args))
        print(f"Computed gender direction-based negative embeddings for {len(negative_embedding_dict)} occupations")

    return use_negative_embedding, negative_embedding_dict, negative_prompt


def generate_noise_tensors(test_prompts, num_imgs_per_prompt, random_seed, device, weight_dtype):
    """Generate reproducible noise tensors for each prompt."""
    noise_all = []
    for prompt in test_prompts:
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        noise_per_prompt = []
        for i in range(num_imgs_per_prompt):
            torch.manual_seed(random_seed + prompt_hash + i)
            noise_single = torch.randn([1, 4, 64, 64], dtype=weight_dtype).to(device)
            noise_per_prompt.append(noise_single)
        noise_per_prompt = torch.cat(noise_per_prompt).unsqueeze(0)
        noise_all.append(noise_per_prompt)
    return torch.cat(noise_all)


def prepare_pipeline_args(prompt, noises, args, weight_dtype,
                          use_negative_embedding, negative_embedding_dict,
                          negative_prompt, occ_name):
    pipe_args = {
        "prompt": [prompt] * len(noises),
        "num_inference_steps": args.num_denoising_steps,
        "guidance_scale": args.guidance_scale,
        "latents": noises.to(weight_dtype),
    }
    if args.w_fair is not None:
        pipe_args["w_fair"] = args.w_fair
    if use_negative_embedding:
        neg_emb = negative_embedding_dict[occ_name]
        pipe_args["negative_prompt_embeds"] = neg_emb.repeat(len(noises), 1, 1).to(weight_dtype)
    elif negative_prompt is not None:
        pipe_args["negative_prompt"] = [negative_prompt] * len(noises)
    return pipe_args


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Fair CFG SD1.5 + LoRA (cfg / cfg_gd)")

    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to LoRA text encoder checkpoint (.pth)")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--prompts_path", type=str, default="./data/occupation_8_v1.json")
    parser.add_argument("--num_imgs_per_prompt", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./generated_sd15_fair")
    parser.add_argument("--occs", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=1997)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--num_denoising_steps', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--test_option', type=str, default='cfg',
                        choices=['cfg', 'cfg_gd'])
    parser.add_argument('--gd_json', type=str, default=None,
                        help='JSON mapping occupation -> alpha; overrides --gd_scale per occupation')
    parser.add_argument('--gd_scale', type=float, default=1.0)
    parser.add_argument('--w_fair', type=float, default=None,
                        help="Fair CFG fairness weight (w_fair). If None, uses standard CFG.")
    parser.add_argument('--lora_rank', type=int, default=50)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def main(args):
    args.device = f"cuda:{args.gpu_id}"

    weight_dtype_high_precision = torch.float32
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    print(f"Loading Fair CFG pipeline from {args.pretrained_model_name_or_path}...")
    pipe = FairCFGStableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Load LoRA text encoder.
    # Injection is LoraLoaderMixin._modify_text_encoder(patch_mlp=True) reproduced inline:
    # that classmethod is gone from diffusers 0.36, but the parts it uses are not, and the
    # module names it produces are what our exported checkpoints are keyed on. The peft
    # route the bundle used matches none of them and silently loads nothing.
    from diffusers.models.lora import (
        PatchedLoraProjection,
        text_encoder_attn_modules,
        text_encoder_mlp_modules,
    )

    text_encoder = pipe.text_encoder
    for _, attn_module in text_encoder_attn_modules(text_encoder):
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(attn_module, proj, PatchedLoraProjection(
                getattr(attn_module, proj), 1, network_alpha=None,
                rank=args.lora_rank, dtype=torch.float32))
    for _, mlp_module in text_encoder_mlp_modules(text_encoder):
        for proj in ("fc1", "fc2"):
            setattr(mlp_module, proj, PatchedLoraProjection(
                getattr(mlp_module, proj), 1, network_alpha=None,
                rank=args.lora_rank, dtype=torch.float32))

    lora_dict = torch.load(args.lora_path, map_location=args.device)
    result = text_encoder.load_state_dict(lora_dict, strict=False)
    unexpected = list(result.unexpected_keys)
    print(f"LoRA — file tensors: {len(lora_dict)}, unexpected: {len(unexpected)}")
    if not lora_dict:
        sys.exit("FAIL: the LoRA checkpoint is empty")
    if unexpected:
        for k in unexpected[:5]:
            print(f"    {k}")
        sys.exit("FAIL: checkpoint keys are absent from the patched text encoder, so "
                 "strict=False dropped them. A zero-initialised LoRA is an identity, so "
                 "this would have produced baseline images under the FT label.")
    model_sd = text_encoder.state_dict()
    max_diff = max((model_sd[k].detach().float().cpu() - v.detach().float().cpu()).abs().max().item()
                   for k, v in lora_dict.items())
    n_nonzero = sum(1 for v in lora_dict.values() if v.abs().max().item() > 0)
    print(f"LoRA — loaded {len(lora_dict)} tensors (non-zero {n_nonzero}), "
          f"max |model-file| {max_diff:.3e}")
    if max_diff > 1e-6:
        sys.exit("FAIL: loaded weights differ from the checkpoint file")
    if n_nonzero == 0:
        sys.exit("FAIL: every LoRA tensor is zero, which is an identity")
    pipe.text_encoder = text_encoder

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)
    pipe.vae.enable_slicing()

    with open(args.prompts_path, 'r') as f:
        experiment_data = json.load(f)
    test_prompts = experiment_data["test_prompts"]
    test_occs = experiment_data["occupations_test_set"]

    use_negative_embedding, negative_embedding_dict, negative_prompt = setup_negative_prompt_config(
        args=args, pipe=pipe, test_occs=test_occs
    )

    print("Generating noise tensors...")
    noise_all = generate_noise_tensors(
        test_prompts=test_prompts,
        num_imgs_per_prompt=args.num_imgs_per_prompt,
        random_seed=args.random_seed,
        device=args.device,
        weight_dtype=weight_dtype_high_precision,
    )

    print(f"Starting image generation for {len(test_prompts)} prompts...")
    candidate_occs = args.occs.split(',') if args.occs is not None else test_occs

    for i, prompt_i in tqdm(enumerate(test_prompts), total=len(test_prompts), desc='Prompts', leave=True):
        if test_occs[i] not in candidate_occs:
            continue
        alpha_i = ALPHA_BY_OCC.get(test_occs[i], args.gd_scale)
        save_dir_prompt_i = os.path.join(
            args.save_dir, test_occs[i], f"a{alpha_i:g}", f"seed{args.random_seed}")
        os.makedirs(save_dir_prompt_i, exist_ok=True)

        noises_to_use = []
        img_save_paths_to_use = []
        for j in range(args.num_imgs_per_prompt):
            img_save_path = os.path.join(save_dir_prompt_i, f"img_{j}.jpg")
            if not os.path.exists(img_save_path):
                noises_to_use.append(noise_all[i, j].unsqueeze(0))
                img_save_paths_to_use.append(img_save_path)

        try:
            noises_to_use = torch.cat(noises_to_use)
        except RuntimeError:
            print(f"Prompt {i} already complete, skipping: {prompt_i}")
            continue

        N = math.ceil(noises_to_use.shape[0] / args.batch_size)
        for j in tqdm(range(N), desc='Images per prompt', leave=False):
            noises_ij = noises_to_use[args.batch_size * j : args.batch_size * (j + 1)]
            img_save_paths_ij = img_save_paths_to_use[args.batch_size * j : args.batch_size * (j + 1)]

            with torch.no_grad():
                pipe_args = prepare_pipeline_args(
                    prompt=prompt_i,
                    noises=noises_ij,
                    args=args,
                    weight_dtype=weight_dtype,
                    use_negative_embedding=use_negative_embedding,
                    negative_embedding_dict=negative_embedding_dict,
                    negative_prompt=negative_prompt,
                    occ_name=test_occs[i],
                )
                output = pipe(**pipe_args)
                images_ij = output.images

            for img_pil, img_save_path in zip(images_ij, img_save_paths_ij):
                img_pil.save(img_save_path)

    print(f"Image generation complete! Saved to {args.save_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)


"""
Example usage:

python generate/gen_image_sd15_fair.py \
    --prompts_path ./data/occupation_8_v1.json \
    --lora_path ./pretrained/text_encoder_lora.pth \
    --save_dir ./generated/fair_cfg \
    --test_option cfg \
    --w_fair 2.0

python generate/gen_image_sd15_fair.py \
    --prompts_path ./data/occupation_8_v1.json \
    --lora_path ./pretrained/text_encoder_lora.pth \
    --save_dir ./generated/fair_cfg_gd \
    --test_option cfg_gd \
    --gd_scale 1.0 \
    --w_fair 2.0
"""
