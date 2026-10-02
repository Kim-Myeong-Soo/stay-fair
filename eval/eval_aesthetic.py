#!/usr/bin/env python3
"""Per-image aesthetic score evaluation.

Saves aesthetic_score.csv in each occupation folder:
    image_name, aesthetic_score
"""

import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModel
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(__file__))
from utils import add_common_args, setup_gpu, find_occupation_dirs, load_images

CSV_NAME = "aesthetic_score.csv"


class AestheticScorer(nn.Module):
    def __init__(self, backbone: CLIPVisionModel):
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        self.aesthetic_head = nn.Linear(hidden_size, 1)
        self.quality_head = nn.Linear(hidden_size, 1)
        self.composition_head = nn.Linear(hidden_size, 1)
        self.light_head = nn.Linear(hidden_size, 1)
        self.color_head = nn.Linear(hidden_size, 1)
        self.dof_head = nn.Linear(hidden_size, 1)
        self.content_head = nn.Linear(hidden_size, 1)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values)
        pooled = outputs.pooler_output
        return self.aesthetic_head(pooled)


def _remap_state_dict(state_dict):
    new = {}
    for k, v in state_dict.items():
        nk = k
        if k.startswith("backbone.") and not k.startswith("backbone.vision_model."):
            nk = k.replace("backbone.", "backbone.vision_model.", 1)
        nk = nk.replace(".0.weight", ".weight").replace(".0.bias", ".bias")
        new[nk] = v
    return new


def load_aesthetic_model(device):
    model_path = hf_hub_download(repo_id="rsinema/aesthetic-scorer", filename="model.pt")
    loaded = torch.load(model_path, map_location=device, weights_only=False)
    backbone = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    model = AestheticScorer(backbone)
    sd = loaded if isinstance(loaded, dict) else loaded.state_dict()
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(_remap_state_dict(sd), strict=False)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description="Per-image aesthetic score")
    add_common_args(parser)
    parser.add_argument("--group_size", type=int, default=5,
                        help="Number of folders to batch together")
    args = parser.parse_args()

    setup_gpu(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading aesthetic model...")
    model = load_aesthetic_model(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    occ_dirs = list(find_occupation_dirs(args.root))
    todo = [d for d in occ_dirs
            if not os.path.exists(os.path.join(d, CSV_NAME)) or args.force]
    print(f"Found {len(occ_dirs)} occupation folders, {len(todo)} to process")

    if not todo:
        print("Nothing to do!")
        return

    pbar = tqdm(total=len(todo), desc="Folders")
    for g in range(0, len(todo), args.group_size):
        group = todo[g:g + args.group_size]

        group_meta = []  # (occ_path, filenames, start, count)
        all_imgs = []

        for occ_path in group:
            images, filenames = load_images(occ_path)
            start = len(all_imgs)
            all_imgs.extend(images)
            group_meta.append((occ_path, filenames, start, len(images)))
            del images

        if not all_imgs:
            pbar.update(len(group))
            continue

        all_scores = []
        for i in range(0, len(all_imgs), args.batch_size):
            batch_imgs = all_imgs[i:i + args.batch_size]
            pixel_values = processor(images=batch_imgs, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                scores = model(pixel_values).squeeze(-1).cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)
            del batch_imgs, pixel_values

        for occ_path, filenames, start, count in group_meta:
            if count == 0:
                continue
            folder_scores = all_scores[start:start + count]
            rows = [{"image_name": fn, "aesthetic_score": round(s, 6)}
                    for fn, s in zip(filenames, folder_scores)]
            pd.DataFrame(rows).to_csv(os.path.join(occ_path, CSV_NAME), index=False)

        del all_imgs, all_scores, group_meta
        pbar.update(len(group))

    pbar.close()
    print("Done!")


if __name__ == "__main__":
    main()
