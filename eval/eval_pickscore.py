#!/usr/bin/env python3
"""Per-image PickScore evaluation.

Saves pickscore.csv in each occupation folder:
    image_name, pickscore

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
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, os.path.dirname(__file__))
from utils import add_common_args, setup_gpu, find_occupation_dirs, load_images

CSV_NAME = "pickscore.csv"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root


def extract_version(occ_path: str) -> str:
    for part in occ_path.replace("\\", "/").split("/"):
        if re.fullmatch(r"v_\d+", part):
            return part
    raise SystemExit(f"[ERROR] No v_* version found in path: {occ_path}")


def version_to_json_path(version: str, project_root: str) -> str:
    """Return data/occupation_8_v{N}.json for the given v_N version."""
    num = version.split("_")[1]
    return os.path.join(project_root, f"data/occupation_8_v{num}.json")


def load_json_prompts(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return dict(zip(data["occupations_test_set"], data["test_prompts"]))


def get_prompt(occ: str, prompts: dict) -> str:
    """Look up prompt for occupation. Exits if not found."""
    if occ in prompts:
        return prompts[occ]
    raise SystemExit(f"[ERROR] Occupation '{occ}' not found in prompts JSON. Aborting.")


def main():
    parser = argparse.ArgumentParser(description="Per-image PickScore")
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

    print(f"Loading PickScore model... (fp16={use_fp16})")
    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1",
                                       torch_dtype=dtype).eval().to(device)
    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    occ_dirs = list(find_occupation_dirs(args.root))
    todo = [d for d in occ_dirs if not os.path.exists(os.path.join(d, CSV_NAME)) or args.force]
    print(f"Found {len(occ_dirs)} folders, {len(todo)} to process")

    if not todo:
        print("Nothing to do!")
        return

    version_cache: dict[str, dict] = {}

    pbar = tqdm(total=len(todo), desc="Folders")
    for g in range(0, len(todo), args.group_size):
        group = todo[g:g + args.group_size]

        # Load images from all folders in the group, track boundaries
        group_meta = []  # (occ_path, start_idx, count)
        all_imgs = []
        all_prompts = []

        for occ_path in group:
            images, filenames = load_images(occ_path)
            occ = os.path.basename(occ_path)
            if re.fullmatch(r"seed\d+", occ):
                occ = os.path.basename(os.path.dirname(occ_path))
            json_path = args.prompts_path or version_to_json_path(extract_version(occ_path), args.project_root)
            if json_path not in version_cache:
                version_cache[json_path] = load_json_prompts(json_path)
            prompt = get_prompt(occ, version_cache[json_path])
            start = len(all_imgs)
            all_imgs.extend(images)
            all_prompts.extend([prompt] * len(images))
            group_meta.append((occ_path, filenames, start, len(images)))
            del images

        if not all_imgs:
            pbar.update(len(group))
            continue

        # Compute scores for all images in one pass
        all_scores = []
        for i in range(0, len(all_imgs), args.batch_size):
            batch_imgs = all_imgs[i:i + args.batch_size]
            batch_prompts = all_prompts[i:i + args.batch_size]

            inputs = processor(
                text=batch_prompts, images=batch_imgs,
                return_tensors="pt", padding=True
            )
            inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point()
                      else v.to(device) for k, v in inputs.items()}

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_fp16):
                image_embs = model.get_image_features(pixel_values=inputs["pixel_values"])
                image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
                text_embs = model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
                logit_scale = model.logit_scale.exp()
                scores = (logit_scale * (image_embs * text_embs).sum(dim=-1)).float().cpu().tolist()

            all_scores.extend(scores)
            del inputs, batch_imgs

        # Save per-folder CSVs
        for occ_path, filenames, start, count in group_meta:
            if count == 0:
                continue
            folder_scores = all_scores[start:start + count]
            rows = [{"image_name": fn, "pickscore": round(s, 6)}
                    for fn, s in zip(filenames, folder_scores)]
            pd.DataFrame(rows).to_csv(os.path.join(occ_path, CSV_NAME), index=False)

        del all_imgs, all_prompts, all_scores, group_meta
        pbar.update(len(group))

    pbar.close()
    print("Done!")


if __name__ == "__main__":
    main()
