#!/usr/bin/env python3
"""SD3-medium image generation (cfg / cfg_gd).

test_option:
- cfg    : CFG with the pipeline's default negative prompt ("")
- cfg_gd : StayFair null embedding
    base  : filler token ids in the slots "female"/"male" occupy
            CLIP-L / CLIP-G  [SOT, EOT*k, EOT, pad...]   (EOT = 49407)
            T5               [<pad>*k, </s>, <pad>...]   (<pad> = 0)
    shift : seq  = B  + gd * d / ||d||_F                  d  = E("female") - E("male")
            pool = Bp + (gd / sqrt(333)) * dp / ||dp||
    gd per occupation from --gd_json; occupations absent from it are generated with plain CFG
    (as in the paper). Without --gd_json every occupation uses --gd_scale. gd = 0 keeps the filler base.

Seeds: random_seed + int(md5(prompt)[:8], 16) + j, one generator per image. Image-level skip-existing.
Output: {save_dir}/{occupation}/seed{random_seed}/img_{j}.jpg
"""
import argparse, hashlib, json, math, os
import torch
from tqdm.auto import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--pretrained_model_name_or_path", type=str,
                default="stabilityai/stable-diffusion-3-medium-diffusers")
ap.add_argument("--prompts_path", type=str, default="./data/occupation_8_v1.json")
ap.add_argument("--save_dir", type=str, default="./generated_images_sd3")
ap.add_argument("--occs", type=str, default=None,
                help="Comma-separated subset of occupations to generate (default: all)")
ap.add_argument("--num_imgs_per_prompt", type=int, default=100)
ap.add_argument("--random_seed", type=int, default=1997)
ap.add_argument("--guidance_scale", type=float, default=7.0)
ap.add_argument("--num_denoising_steps", type=int, default=28)
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--height", type=int, default=512)
ap.add_argument("--width", type=int, default=512)
ap.add_argument("--test_option", type=str, default="cfg", choices=["cfg", "cfg_gd"])
ap.add_argument("--gd_json", type=str, default=None,
                help="JSON giving gd per occupation, either {occ: gd} or {'gd': {occ: gd}}")
ap.add_argument("--gd_scale", type=float, default=0.0,
                help="gd for every occupation when --gd_json is not given")
args = ap.parse_args()
from diffusers import StableDiffusion3Pipeline

CLIP_F, T5_F = 49407, 0   # eep: EOT for CLIP-L and CLIP-G, <pad> for T5
CFG_GD = args.test_option == "cfg_gd"

dev, dt = "cuda:0", torch.float16
torch.backends.cuda.matmul.allow_tf32 = CFG_GD   # the null embedding below is built with tf32 on
pipe = StableDiffusion3Pipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=dt).to(dev)
pipe.set_progress_bar_config(disable=True)
data = json.load(open(args.prompts_path))
test_prompts, test_occs = data["test_prompts"], data["occupations_test_set"]
candidate_occs = args.occs.split(",") if args.occs is not None else test_occs

gd_map = {}
if CFG_GD and args.gd_json:
    loaded = json.load(open(args.gd_json))
    gd_map = {k: float(v) for k, v in loaded.get("gd", loaded).items()}
    print(f"gd_json: {args.gd_json} ({len(gd_map)} occupations)", flush=True)


@torch.no_grad()
def E_text(t):
    s, _, p, _ = pipe.encode_prompt(prompt=t, prompt_2=None, prompt_3=None, device=dev,
                                     num_images_per_prompt=1, do_classifier_free_guidance=False)
    return s[0].float(), p[0].float()

def base_ids(tok, n):
    return tok("", padding="max_length", max_length=n, truncation=True, return_tensors="pt").input_ids[0]

@torch.no_grad()
def E_ids(idsL, idsG, idsT):
    oL = pipe.text_encoder(idsL[None].to(dev), output_hidden_states=True)
    oG = pipe.text_encoder_2(idsG[None].to(dev), output_hidden_states=True)
    t5 = pipe.text_encoder_3(idsT[None].to(dev))[0]
    clip = torch.cat([oL.hidden_states[-2], oG.hidden_states[-2]], -1)
    clip = torch.nn.functional.pad(clip, (0, t5.shape[-1] - clip.shape[-1]))
    return torch.cat([clip, t5], 1)[0].float(), torch.cat([oL[0], oG[0]], -1)[0].float()

def filler_ids(kc, kt):
    out = []
    for tok in (pipe.tokenizer, pipe.tokenizer_2):
        b = base_ids(tok, 77)
        out.append(torch.cat([b[:1], torch.full((kc,), CLIP_F, dtype=b.dtype), b[1:77 - kc]]))
    b = base_ids(pipe.tokenizer_3, 256)
    out.append(torch.cat([torch.full((kt,), T5_F, dtype=b.dtype), b[:256 - kt]]))
    return out

if CFG_GD:
    assert pipe.tokenizer.convert_ids_to_tokens([CLIP_F]) == pipe.tokenizer_2.convert_ids_to_tokens([CLIP_F]) == ["<|endoftext|>"]
    assert pipe.tokenizer_3.convert_ids_to_tokens([T5_F]) == ["<pad>"]
    # sanity: id path reproduces encode_prompt("")
    _s, _p = E_ids(base_ids(pipe.tokenizer, 77), base_ids(pipe.tokenizer_2, 77), base_ids(pipe.tokenizer_3, 256))
    _s0, _p0 = E_text("")
    err = max((_s - _s0).abs().max().item(), (_p - _p0).abs().max().item())
    print(f"id-path vs encode_prompt(''): max|diff| = {err:.2e}", flush=True)
    assert err < 1e-3, "id-level encoding does not reproduce encode_prompt"

    # gender direction and the filler base, shared by every occupation
    f, fp = E_text("female"); m, mp = E_text("male"); D, DP = f - m, fp - mp
    KC = len(pipe.tokenizer("female").input_ids) - 2
    KT = len(pipe.tokenizer_3("female").input_ids) - 1
    B, BP = E_ids(*filler_ids(KC, KT))

_null = {}
def null_for(gd):
    gd = float(gd)
    if gd not in _null:
        if gd == 0.0:
            seq, pool = B, BP
        else:
            seq = B + gd * D / (D.norm() + 1e-6)
            pool = BP + gd / math.sqrt(D.shape[0]) * DP / (DP.norm(dim=-1, keepdim=True) + 1e-6)
        assert torch.isfinite(seq).all() and torch.isfinite(pool).all(), gd
        assert seq.shape == (333, 4096) and pool.shape == (2048,), (seq.shape, pool.shape)
        _null[gd] = (seq.to(dt), pool.to(dt))
    return _null[gd]

def prompt_seed(prompt, j):
    return args.random_seed + int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16) + j


print(f"Starting generation ({args.test_option}, guidance={args.guidance_scale}, "
      f"steps={args.num_denoising_steps}, {args.height}x{args.width})...", flush=True)
for prompt, occ in tqdm(list(zip(test_prompts, test_occs)), desc="Prompts"):
    if occ not in candidate_occs:
        continue
    out_dir = os.path.join(args.save_dir, occ, f"seed{args.random_seed}")
    os.makedirs(out_dir, exist_ok=True)
    todo = [j for j in range(args.num_imgs_per_prompt) if not os.path.exists(os.path.join(out_dir, f"img_{j}.jpg"))]
    if not todo:
        print(f"{occ} already complete, skipping", flush=True)
        continue
    use_gd = CFG_GD and (not gd_map or occ in gd_map)
    torch.backends.cuda.matmul.allow_tf32 = use_gd   # off for CFG, as in the paper's CFG runs
    if use_gd:
        gd = gd_map.get(occ, args.gd_scale)
        seq, pool = null_for(gd)
        print(f"  {occ}: gd={gd:g}", flush=True)
    elif CFG_GD:
        print(f"  {occ}: not in gd_json -> cfg", flush=True)
    for k in tqdm(range(0, len(todo), args.batch_size), desc=occ, leave=False):
        js = todo[k:k + args.batch_size]; n = len(js)
        gens = [torch.Generator(device=dev).manual_seed(prompt_seed(prompt, j)) for j in js]
        with torch.no_grad():
            if use_gd:
                cs, _, cp, _ = pipe.encode_prompt(prompt=[prompt] * n, prompt_2=None, prompt_3=None, device=dev,
                                                  num_images_per_prompt=1, do_classifier_free_guidance=True,
                                                  negative_prompt=[""] * n)
                imgs = pipe(prompt_embeds=cs, pooled_prompt_embeds=cp,
                            negative_prompt_embeds=seq[None].repeat(n, 1, 1),
                            negative_pooled_prompt_embeds=pool[None].repeat(n, 1),
                            guidance_scale=args.guidance_scale, num_inference_steps=args.num_denoising_steps,
                            height=args.height, width=args.width, generator=gens).images
            else:
                imgs = pipe(prompt=[prompt] * n, negative_prompt=[""] * n,
                            num_inference_steps=args.num_denoising_steps, guidance_scale=args.guidance_scale,
                            generator=gens, height=args.height, width=args.width).images
        for j, im in zip(js, imgs):
            path = os.path.join(out_dir, f"img_{j}.jpg")
            if use_gd:
                im.save(path, quality=95)
            else:
                im.save(path)
print(f"Image generation complete! Saved to {args.save_dir}", flush=True)


"""
Example usage (paper setting: 512px, 28 steps):

# cfg (baseline)
python generate/vanilla/gen_image_sd3.py \\
    --prompts_path ./data/occupation_8_v1.json \\
    --save_dir ./outputs/sd3/cfg/w4.5 \\
    --guidance_scale 4.5 --batch_size 100 \\
    --test_option cfg

# cfg_gd (StayFair)
python generate/vanilla/gen_image_sd3.py \\
    --prompts_path ./data/occupation_8_v1.json \\
    --save_dir ./outputs/sd3/stayfair/w4.5 \\
    --guidance_scale 4.5 --batch_size 100 \\
    --test_option cfg_gd --gd_json ./data/alpha/sd3.json
"""
