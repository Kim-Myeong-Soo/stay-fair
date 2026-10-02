#!/usr/bin/env python
# coding=utf-8
"""
FLUX.2 [klein] base-9B 이미지 생성 스크립트.

SD1.5 레퍼런스(gen_image_sd15.py)의 규약을 그대로 보존한 포트:
- 입력 JSON: test_prompts / occupations_test_set (index-aligned 1:1)
- 출력 레이아웃: {save_dir}/{occupation}/seed{random_seed}/img_{j}.jpg
- 재현용 시드 공식: seed = random_seed + int(md5(prompt)[:8], 16) + j
  (SD1.5에서는 latent를 시딩했지만, FLUX는 latent packing이 달라서
   대신 per-image torch.Generator에 동일한 시드를 넣어 pipeline에 전달)
- skip-existing / resumable
- batch_size 만큼 pipeline 호출 (per-image generator 리스트로 배칭)

test_option:
- cfg    : 기본 negative prompt를 쓰는 CFG
- cfg_gd : StayFair null. base = enc(" ") (13 tokens, direction과 길이 정렬),
           u = (enc("female") - enc("male")) / ||.||_F (block normalisation),
           negative = base + gd * u.  gd는 --gd_json의 직업별 값.
           --gd_json에 없는 직업은 cfg로 생성 (논문과 동일). --gd_json이 없으면 모든 직업에 --gd_scale.
"""

import argparse
import hashlib
import json
import math
import os

# --- GPU 고정: torch import 전에 CUDA_VISIBLE_DEVICES 설정 ---
_parser_gpu = argparse.ArgumentParser(add_help=False)
_parser_gpu.add_argument("--gpu_id", type=int, default=0)
_known, _ = _parser_gpu.parse_known_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(_known.gpu_id))

import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from diffusers import DiffusionPipeline  # noqa: E402

GD_NULL_TOKEN = " "


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Generate images with FLUX.2 [klein] base-9B (gender-bias T2I study)."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.2-klein-base-9B",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--prompts_path",
        type=str,
        default="./data/occupation_8_v1.json",
        help="Path to JSON file containing prompts",
    )
    parser.add_argument("--num_imgs_per_prompt", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./generated_images_flux2")
    parser.add_argument(
        "--occs",
        type=str,
        default=None,
        help="Comma-separated subset of occupations to generate (filter).",
    )
    parser.add_argument("--random_seed", type=int, default=1997)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--guidance_scale",
        help="diffusion model text guidance scale (FLUX base wants ~4.0)",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--num_denoising_steps",
        help="num denoising steps (FLUX base wants ~28-50)",
        type=int,
        default=28,
    )
    parser.add_argument(
        "--batch_size", help="batch size for image generation", type=int, default=4
    )
    parser.add_argument(
        "--test_option",
        help="test option",
        type=str,
        default="cfg",
        choices=["cfg", "cfg_gd"],
    )
    parser.add_argument(
        "--gd_scale",
        help="gender direction scale for every occupation (test_option='cfg_gd' without --gd_json)",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--gd_json", type=str, default=None,
        help="JSON giving gd per occupation, either {occ: gd} or {'gd': {occ: gd}}. Occupations "
             "absent from the map are generated with plain cfg. Read after the pipeline is built and "
             "before any generator is seeded, so it touches no RNG.",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument(
        "--enable_cpu_offload",
        action="store_true",
        default=False,
        help="Enable pipe.enable_model_cpu_offload() (use if full-GPU load OOMs).",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def enc(pipe, text, device):
    """
    Flux2KleinPipeline의 자체 encode_prompt 경로 사용.
    encode_prompt(prompt) -> (prompt_embeds, text_ids)
      prompt_embeds: (1, 512, D)  <- main token-sequence 임베딩
      text_ids:      (1, 512, 4)  <- 여기선 사용 안 함 (negative 쪽은 pipeline이 shape로부터 재계산)
    """
    with torch.no_grad():
        prompt_embeds, _text_ids = pipe.encode_prompt(
            prompt=text,
            device=device,
            num_images_per_prompt=1,
        )
    return prompt_embeds  # (1, 512, D)


def gd_direction(pipe, device):
    """Returns (base, u). Neither depends on the occupation, so this is computed once."""
    d = enc(pipe, "female", device) - enc(pipe, "male", device)
    u = d / (d.norm() + 1e-6)
    return enc(pipe, GD_NULL_TOKEN, device), u


def gd_negative(base, u, gd_scale, weight_dtype):
    return (base + gd_scale * u).to(weight_dtype)


def prompt_seed(prompt, j, random_seed):
    """seed = random_seed + int(md5(prompt)[:8], 16) + j  (레퍼런스 공식 그대로)."""
    prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
    return random_seed + prompt_hash + j


def main(args):
    # CUDA_VISIBLE_DEVICES로 gpu를 고정했으므로 항상 cuda:0.
    device = "cuda:0"

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    print(f"Loading model from {args.pretrained_model_name_or_path} (dtype={weight_dtype})...")
    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
    )
    print(f"Loaded pipeline: {pipe.__class__.__name__}")

    if args.enable_cpu_offload:
        print("Enabling model CPU offload...")
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe = pipe.to(device)

    # 입력 로드
    with open(args.prompts_path, "r") as f:
        experiment_data = json.load(f)
    test_prompts = experiment_data["test_prompts"]
    test_occs = experiment_data["occupations_test_set"]
    assert len(test_prompts) == len(test_occs), "test_prompts와 occupations_test_set 길이 불일치"

    # negative embedding 설정 (test_option 기반)
    negative_prompt_embeds_single = None  # (1, 512, D) or None
    GD_BASE = GD_U = None
    GD_MAP = {}
    if args.test_option == "cfg":
        print("Test option: cfg (no negative prompt, empty-negative CFG baseline)")
    elif args.test_option == "cfg_gd":
        print(f"Test option: cfg_gd (gender direction, gd_scale={args.gd_scale})")
        GD_BASE, GD_U = gd_direction(pipe, device)
        negative_prompt_embeds_single = gd_negative(GD_BASE, GD_U, args.gd_scale, weight_dtype)
        print(f"  [gd] null={GD_NULL_TOKEN!r} ||u||_F={GD_U.norm().item():.3f}")
        if args.gd_json:
            with open(args.gd_json) as fh:
                loaded = json.load(fh)
            GD_MAP = {k: float(v) for k, v in (loaded.get("gd", loaded)).items()}
            shown = ", ".join(f"{o}={GD_MAP[o]:g}" for o in sorted(GD_MAP)[:5])
            print(f"  [gd] gd_json: {args.gd_json} ({len(GD_MAP)} occupations; {shown}, ...)")

    candidate_occs = args.occs.split(",") if args.occs is not None else test_occs

    print(f"Starting image generation for {len(test_prompts)} prompts "
          f"(guidance={args.guidance_scale}, steps={args.num_denoising_steps}, "
          f"{args.height}x{args.width})...")

    for i, prompt_i in tqdm(
        enumerate(test_prompts), total=len(test_prompts), desc="Prompts", leave=True
    ):
        occ_i = test_occs[i]
        if occ_i not in candidate_occs:
            continue

        if GD_MAP and occ_i in GD_MAP:
            # per-occupation gd; the map is data and is read after the pipeline exists, so the
            # noise a given (prompt, index) gets does not depend on which gd it is generated at
            negative_prompt_embeds_single = gd_negative(GD_BASE, GD_U, GD_MAP[occ_i], weight_dtype)
            print(f"  [gd] {occ_i}: gd={GD_MAP[occ_i]:g}")
        elif GD_MAP:
            negative_prompt_embeds_single = None
            print(f"  [gd] {occ_i}: not in gd_json -> cfg")

        save_dir_prompt_i = os.path.join(args.save_dir, f"{occ_i}", f"seed{args.random_seed}")
        os.makedirs(save_dir_prompt_i, exist_ok=True)

        # skip-existing: 이미 존재하는 img_j.jpg 는 건너뜀
        seeds_to_use = []
        img_save_paths_to_use = []
        for j in range(args.num_imgs_per_prompt):
            img_save_path = os.path.join(save_dir_prompt_i, f"img_{j}.jpg")
            if not os.path.exists(img_save_path):
                seeds_to_use.append(prompt_seed(prompt_i, j, args.random_seed))
                img_save_paths_to_use.append(img_save_path)

        if len(seeds_to_use) == 0:
            print(f"Prompt {i} already complete, skipping: {prompt_i}")
            continue

        # 배치 단위 생성
        N = math.ceil(len(seeds_to_use) / args.batch_size)
        for b in tqdm(range(N), desc="Images per prompt", leave=False):
            seeds_b = seeds_to_use[args.batch_size * b: args.batch_size * (b + 1)]
            paths_b = img_save_paths_to_use[args.batch_size * b: args.batch_size * (b + 1)]
            n = len(seeds_b)

            # per-image generator 리스트 (per-(prompt,index) 재현성 보존)
            generators = [torch.Generator(device=device).manual_seed(s) for s in seeds_b]

            pipe_args = {
                "prompt": [prompt_i] * n,
                "num_inference_steps": args.num_denoising_steps,
                "guidance_scale": args.guidance_scale,
                "generator": generators,
                "height": args.height,
                "width": args.width,
            }

            if negative_prompt_embeds_single is not None:
                # 단일 (1,512,D) 를 배치 n 으로 repeat -> (n,512,D)
                # pipeline이 negative 쪽 text_ids 등을 shape로부터 자동 재계산.
                neg_b = negative_prompt_embeds_single.repeat(n, 1, 1).to(weight_dtype)
                pipe_args["negative_prompt_embeds"] = neg_b

            with torch.no_grad():
                output = pipe(**pipe_args)
            images_b = output.images

            for img_pil, img_save_path in zip(images_b, paths_b):
                img_pil.save(img_save_path)

    print(f"Image generation complete! Saved to {args.save_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)


"""
Example usage (paper setting: 768px, 28 steps, batch 16):

# cfg (baseline)
python generate/vanilla/gen_image_flux2.py \
    --prompts_path ./data/occupation_8_v1.json \
    --save_dir ./outputs/flux2/cfg/g4.0 \
    --gpu_id 0 --batch_size 16 --height 768 --width 768 \
    --guidance_scale 4.0 --num_denoising_steps 28 \
    --test_option cfg

# cfg_gd (StayFair)
python generate/vanilla/gen_image_flux2.py \
    --prompts_path ./data/occupation_8_v1.json \
    --save_dir ./outputs/flux2/stayfair/g4.0 \
    --gpu_id 0 --batch_size 16 --height 768 --width 768 \
    --guidance_scale 4.0 --num_denoising_steps 28 \
    --test_option cfg_gd --gd_json ./data/alpha/flux2.json
"""
