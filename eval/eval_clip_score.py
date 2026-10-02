#!/usr/bin/env python3
"""Per-image CLIP score evaluation.

Saves clip_score.csv in each occupation folder:
    image_name, clip_score

Prompt is resolved per occupation from --prompts_path, or, if not given, from
data/occupation_8_v{N}.json picked by the v_N component of the path.
The occupation is the image folder name, or its parent when the folder is seed*.
    not found → exit
"""

import argparse
import json
import os
import re
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(__file__))
from utils import add_common_args, setup_gpu, find_occupation_dirs, load_images

CSV_NAME = "clip_score.csv"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root


def extract_version(occ_path: str) -> str:
    """Extract v_N component from path. Exits if not found."""
    for part in occ_path.replace("\\", "/").split("/"):
        if re.fullmatch(r"v_\d+", part):
            return part
    raise SystemExit(f"[ERROR] No v_* version found in path: {occ_path}")


def version_to_json_path(version: str, project_root: str) -> str:
    """Return data/occupation_8_v{N}.json for the given v_N version."""
    num = version.split("_")[1]
    return os.path.join(project_root, f"data/occupation_8_v{num}.json")


def load_json_prompts(path: str) -> dict:
    """Load {occupation: prompt} from a versioned JSON file."""
    with open(path) as f:
        data = json.load(f)
    return dict(zip(data["occupations_test_set"], data["test_prompts"]))


def get_prompt(occ: str, prompts: dict) -> str:
    """Look up prompt for occupation. Exits if not found."""
    if occ in prompts:
        return prompts[occ]
    raise SystemExit(f"[ERROR] Occupation '{occ}' not found in prompts JSON. Aborting.")


def main():
    parser = argparse.ArgumentParser(description="Per-image CLIP score")
    add_common_args(parser)
    parser.add_argument("--project_root", type=str, default=PROJECT_ROOT)
    parser.add_argument("--prompts_path", type=str, default=None,
                        help="prompts JSON for every folder (default: data/occupation_8_v{N}.json from v_N in the path)")
    parser.add_argument("--group_size", type=int, default=5,
                        help="Number of folders to batch together")
    args = parser.parse_args()
    use_fp16 = args.fp16 and not args.no_fp16

    setup_gpu(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if use_fp16 else torch.float32

    print(f"Loading CLIP model... (fp16={use_fp16})")
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", torch_dtype=dtype
    ).eval().to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    occ_dirs = list(find_occupation_dirs(args.root))
    todo = [d for d in occ_dirs
            if not os.path.exists(os.path.join(d, CSV_NAME)) or args.force]
    print(f"Found {len(occ_dirs)} folders, {len(todo)} to process")

    if not todo:
        print("Nothing to do!")
        return

    # Cache: prompts JSON path → {occupation: prompt}
    version_cache: dict[str, dict] = {}
    # Cache: prompt text → normalized text embedding tensor (1, D)
    text_emb_cache: dict[str, torch.Tensor] = {}

    def get_text_emb(prompt: str) -> torch.Tensor:
        if prompt not in text_emb_cache:
            inputs = processor(text=[prompt], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = model.get_text_features(**inputs).float()
                emb = F.normalize(emb, p=2, dim=-1).to(dtype)
            text_emb_cache[prompt] = emb
        return text_emb_cache[prompt]

    pbar = tqdm(total=len(todo), desc="Folders")
    for g in range(0, len(todo), args.group_size):
        group = todo[g:g + args.group_size]

        group_meta = []  # (occ_path, filenames, start, count, text_emb)
        all_imgs = []
        all_text_embs = []  # per-image text embedding (same prompt within a folder)

        for occ_path in group:
            occ = os.path.basename(occ_path)
            if re.fullmatch(r"seed\d+", occ):
                occ = os.path.basename(os.path.dirname(occ_path))
            json_path = args.prompts_path or version_to_json_path(extract_version(occ_path), args.project_root)
            if json_path not in version_cache:
                version_cache[json_path] = load_json_prompts(json_path)
            prompt = get_prompt(occ, version_cache[json_path])
            text_emb = get_text_emb(prompt)  # (1, D)

            images, filenames = load_images(occ_path)
            start = len(all_imgs)
            all_imgs.extend(images)
            # repeat text_emb for each image in this folder
            all_text_embs.extend([text_emb] * len(images))
            group_meta.append((occ_path, filenames, start, len(images)))
            del images

        if not all_imgs:
            pbar.update(len(group))
            continue

        all_scores = []
        for i in range(0, len(all_imgs), args.batch_size):
            batch_imgs = all_imgs[i:i + args.batch_size]
            batch_text_embs = torch.cat(all_text_embs[i:i + args.batch_size], dim=0)  # (B, D)

            pixel_values = processor(images=batch_imgs, return_tensors="pt")["pixel_values"]
            pixel_values = pixel_values.to(device=device, dtype=dtype)

            with torch.no_grad():
                img_emb = model.get_image_features(pixel_values=pixel_values).float()
                img_emb = F.normalize(img_emb, p=2, dim=-1).to(dtype)  # (B, D)

            batch_text_embs = batch_text_embs.to(device=device, dtype=dtype)
            scores = (img_emb * batch_text_embs).sum(dim=-1).float().cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)
            del batch_imgs, pixel_values

        for occ_path, filenames, start, count in group_meta:
            if count == 0:
                continue
            folder_scores = all_scores[start:start + count]
            rows = [{"image_name": fn, "clip_score": round(s, 6)}
                    for fn, s in zip(filenames, folder_scores)]
            pd.DataFrame(rows).to_csv(os.path.join(occ_path, CSV_NAME), index=False)

        del all_imgs, all_text_embs, all_scores, group_meta
        pbar.update(len(group))

    pbar.close()
    print("Done!")


if __name__ == "__main__":
    main()
