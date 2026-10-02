#!/usr/bin/env python3
"""Generate SD1.5 images with UCE weights using FairCFG with a reference
guidance scale.

Pipeline:
    noise_pred = w * eps_pos
               - (w - w_fair) * eps_neg
               + (1 - w_fair) * eps_null

We set `w_fair = w_ref` (fixed reference), so:
    - at w == w_ref: noise_pred = w_ref eps_pos + (1-w_ref) eps_null  (== stock CFG at w_ref)
    - at w  > w_ref: eps_neg is weighted by (w - w_ref) → intervention grows with the gap

The `gd_scale` argument controls the *direction* that `eps_neg` encodes:
    neg_emb = enc("") + gd_scale * gender_unit
where `gender_unit` = normalize(enc("female") - enc("male")).

UCE weights are loaded into the UNet the same way as `gen_image_sd15_uce.py`.
No LoRA.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import sys

import torch
from diffusers import DPMSolverMultistepScheduler
from safetensors.torch import load_file
from tqdm import tqdm

# Import custom pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.fair_cfg_pipeline import FairCFGStableDiffusionPipeline


# ---------------------------------------------------------------------------
# Embeddings / gender direction
# ---------------------------------------------------------------------------
def encode_prompt(pipe, text):
    tok = pipe.tokenizer(
        text,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(pipe.device)
    with torch.no_grad():
        emb = pipe.text_encoder(tok.input_ids)[0]  # (1, 77, 768)
    return emb.squeeze(0)  # (77, 768)


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
    """Return {occ: neg_emb (1,77,768)} where neg_emb = "" + gd_scale * gender_unit.

    Matches the gd_scale handling used in gen_image_sd15_uce.py so that runs
    with gd_scale=0 exactly reproduce the empty-string negative prompt.
    """
    if gd_scale == 0.0 and not gd_map:
        base = encode_prompt(pipe, "").unsqueeze(0)  # (1,77,768)
        for occ in test_occs:
            ALPHA_BY_OCC[occ] = 0.0
        return {occ: base for occ in test_occs}

    male_emb = encode_prompt(pipe, "male").unsqueeze(0)
    female_emb = encode_prompt(pipe, "female").unsqueeze(0)
    gender_dir = female_emb - male_emb
    gender_unit = gender_dir / (gender_dir.norm(dim=-1, keepdim=True) + 1e-6)
    base = encode_prompt(pipe, "").unsqueeze(0)

    out = {}
    for occ in test_occs:
        a = gd_scale if gd_map is None else float(gd_map.get(occ, gd_scale))
        ALPHA_BY_OCC[occ] = a
        out[occ] = base + a * gender_unit
    return out


def generate_noise_tensors(test_prompts, num_imgs_per_prompt, random_seed, device, weight_dtype):
    """Reproducible noise tensor with shape (num_prompts, num_imgs_per_prompt, 4, 64, 64).

    seed = random_seed + int(md5(prompt)[:8], 16) + i, the same scheme as the other
    generators, so the noise depends on the prompt text and not on its position in the JSON.
    """
    noise_all = []
    for prompt in test_prompts:
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        per_prompt = []
        for i in range(num_imgs_per_prompt):
            torch.manual_seed(random_seed + prompt_hash + i)
            per_prompt.append(torch.randn([1, 4, 64, 64], dtype=weight_dtype).to(device))
        noise_all.append(torch.cat(per_prompt).unsqueeze(0))
    return torch.cat(noise_all)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate images with UCE-debiased SD1.5 + FairCFG(w_ref)")
    p.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--uce_model_path", type=str, required=True,
                   help="Path to UCE .safetensors file (e.g. uce_models/sd15_gender_debias.safetensors)")
    p.add_argument("--prompts_path", type=str, required=True,
                   help="JSON with 'test_prompts' and 'occupations_test_set' keys")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--num_imgs_per_prompt", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--random_seed", type=int, default=1997)
    p.add_argument("--guidance_scale", type=float, default=7.5,
                   help="Total CFG scale w used for generation")
    p.add_argument("--w_ref", type=float, default=7.5,
                   help="Reference guidance scale at which the pipeline reduces to plain CFG. "
                        "w_fair is set to this value; eps_neg is weighted by (w - w_ref).")
    p.add_argument("--num_denoising_steps", type=int, default=30)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gd_json", type=str, default=None,
        help="JSON mapping occupation -> alpha; overrides --gd_scale per occupation")
    p.add_argument("--gd_scale", type=float, default=0.0,
                   help="Scale for gender direction in neg_emb: '' + gd_scale * gender_unit")
    p.add_argument("--occs", type=str, default=None,
                   help="Comma-separated subset of occupations to generate. "
                        "Noise indices are preserved from the full JSON order for reproducibility.")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))
    device = f"cuda:{args.gpu_id}"

    weight_dtype_high_precision = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # ----- Pipeline -----
    print(f"Loading SD1.5 from {args.model_id}...")
    pipe = FairCFGStableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=weight_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.vae.enable_slicing()

    # ----- UCE weights into UNet (no LoRA on text encoder) -----
    print(f"Loading UCE weights from {args.uce_model_path}...")
    uce_weights = load_file(args.uce_model_path)
    pipe.unet.load_state_dict(uce_weights, strict=False)
    print(f"Loaded {len(uce_weights)} UCE weight tensors into UNet")

    # ----- Prompts -----
    with open(args.prompts_path) as f:
        experiment_data = json.load(f)
    test_prompts = experiment_data["test_prompts"]
    test_occs = experiment_data["occupations_test_set"]

    # ----- Negative embeddings -----
    neg_emb_dict = compute_gender_direction_embeddings(
            pipe, test_occs, args.gd_scale, gd_map=_load_gd_map(args))
    print(f"Computed neg embeddings for {len(neg_emb_dict)} occupations "
          f"(gd_scale={args.gd_scale})")

    # ----- Noise -----
    print("Generating noise tensors...")
    noise_all = generate_noise_tensors(
        test_prompts=test_prompts,
        num_imgs_per_prompt=args.num_imgs_per_prompt,
        random_seed=args.random_seed,
        device=device,
        weight_dtype=weight_dtype_high_precision,
    )
    print(f"Noise tensor shape: {noise_all.shape}")
    print(f"Generating for {len(test_prompts)} occupations, "
          f"{args.num_imgs_per_prompt} images each  "
          f"(w={args.guidance_scale}, w_ref={args.w_ref}, gd_scale={args.gd_scale})")

    # ----- Generate -----
    candidate_occs = args.occs.split(',') if args.occs is not None else test_occs
    for i, prompt_i in tqdm(enumerate(test_prompts), total=len(test_prompts), desc='Prompts'):
        occ = test_occs[i]
        if occ not in candidate_occs:
            print(f"Skipping occupation: {occ}")
            continue
        alpha_i = ALPHA_BY_OCC.get(occ, args.gd_scale)
        save_dir_occ = os.path.join(
            args.save_dir, occ, f"a{alpha_i:g}", f"seed{args.random_seed}")
        os.makedirs(save_dir_occ, exist_ok=True)

        noises_to_use = []
        img_save_paths = []
        for j in range(args.num_imgs_per_prompt):
            img_path = os.path.join(save_dir_occ, f"img_{j}.jpg")
            if not os.path.exists(img_path):
                noises_to_use.append(noise_all[i, j].unsqueeze(0))
                img_save_paths.append(img_path)

        if not noises_to_use:
            print(f"Skipping {occ} (already complete)")
            continue

        noises_to_use = torch.cat(noises_to_use)
        N = math.ceil(len(noises_to_use) / args.batch_size)
        neg_emb = neg_emb_dict[occ]

        for j in tqdm(range(N), desc=occ, leave=False):
            batch_noises = noises_to_use[args.batch_size * j: args.batch_size * (j + 1)]
            batch_paths = img_save_paths[args.batch_size * j: args.batch_size * (j + 1)]
            bsz = len(batch_noises)

            neg_batch = neg_emb.repeat(bsz, 1, 1).to(weight_dtype)

            with torch.no_grad():
                output = pipe(
                    prompt=[prompt_i] * bsz,
                    num_inference_steps=args.num_denoising_steps,
                    guidance_scale=args.guidance_scale,
                    w_fair=args.w_ref,
                    latents=batch_noises.to(weight_dtype),
                    negative_prompt_embeds=neg_batch,
                )
            for img_pil, path in zip(output.images, batch_paths):
                img_pil.save(path)

    # ----- Metadata -----
    metadata = {
        "method": "UCE+FairCFG(w_ref)",
        "model": args.model_id,
        "uce_model_path": args.uce_model_path,
        "prompts_path": args.prompts_path,
        "num_imgs_per_prompt": args.num_imgs_per_prompt,
        "guidance_scale": args.guidance_scale,
        "w_ref": args.w_ref,
        "gd_scale": args.gd_scale,
        "random_seed": args.random_seed,
        "num_denoising_steps": args.num_denoising_steps,
        "mixed_precision": args.mixed_precision,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, f"metadata_seed{args.random_seed}.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done! Images saved to {args.save_dir}")


if __name__ == "__main__":
    main()
