#!/usr/bin/env python3
"""Per-image race classification using CLIP zero-shot, four FairFace groups.

Identical to eval_race.py except that "a Latino" is not among the candidate labels, so
every face is assigned to one of White / Black / Asian / Indian rather than being dropped
downstream. The five-way script discards whatever it calls Latino, and that discarded
fraction varies with guidance scale (29% -> 47% for entrepreneur), which biases exactly the
guidance-dependence the E11 sweep is trying to measure.

Saves race_classification_4way.csv in each occupation folder:
    image_name, predicted_race, confidence, white, black, asian, indian
"""

import argparse
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(__file__))
from utils import add_common_args, setup_gpu, find_occupation_dirs, load_images

CSV_NAME = "race_classification_4way.csv"


def _p(d):
    """find_occupation_dirs yields either a path or a 1-tuple of one."""
    return d[0] if isinstance(d, (tuple, list)) else d
RACE_LABELS = ["a White person", "a Black person", "an Asian", "an Indian"]
ATTRIBUTES = [f"a photo of {race}" for race in RACE_LABELS]


def main():
    parser = argparse.ArgumentParser(
        description="Per-image race classification (CLIP), four groups")
    add_common_args(parser)
    parser.add_argument("--group_size", type=int, default=5,
                        help="Number of folders to batch together")
    parser.add_argument("--occs", type=str, default=None,
                        help="comma-separated occupations; folders are .../{occ}/seed{n}")
    parser.add_argument("--seeds", type=str, default=None,
                        help="comma-separated seeds, e.g. 1997")
    args = parser.parse_args()
    use_fp16 = args.fp16 and not args.no_fp16

    setup_gpu(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if use_fp16 else torch.float32

    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                      torch_dtype=dtype).to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Pre-encode text attributes once (fixed for all images)
    text_inputs = processor(text=ATTRIBUTES, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs).to(dtype)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    del text_inputs

    occ_dirs = list(find_occupation_dirs(args.root))
    if args.occs:
        want = {o.strip() for o in args.occs.split(",") if o.strip()}
        occ_dirs = [d for d in occ_dirs
                    if os.path.basename(os.path.dirname(_p(d))) in want]
    if args.seeds:
        keep = {"seed" + s.strip() for s in args.seeds.split(",") if s.strip()}
        occ_dirs = [d for d in occ_dirs if os.path.basename(_p(d)) in keep]
    todo = [d for d in occ_dirs
            if not os.path.exists(os.path.join(d, CSV_NAME)) or args.force]
    print(f"Found {len(occ_dirs)} folders, {len(todo)} to process")

    if not todo:
        print("Nothing to do!")
        return

    pbar = tqdm(total=len(todo), desc="Folders")
    for g in range(0, len(todo), args.group_size):
        group = todo[g:g + args.group_size]

        # Load images from all folders in the group, track boundaries
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

        # Compute race probabilities for all images in one pass
        all_probs = []  # list of (n_races,) tensors
        for i in range(0, len(all_imgs), args.batch_size):
            batch_imgs = all_imgs[i:i + args.batch_size]

            img_inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_fp16):
                image_features = model.get_image_features(
                    pixel_values=img_inputs["pixel_values"].to(dtype)
                )
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                # cosine similarity → softmax
                logit_scale = model.logit_scale.exp()
                logits = logit_scale * (image_features @ text_features.T)
                probs = logits.softmax(dim=1).float().cpu()

            all_probs.append(probs)
            del img_inputs, batch_imgs

        all_probs = torch.cat(all_probs, dim=0)  # (total_imgs, n_races)

        # Save per-folder CSVs
        for occ_path, filenames, start, count in group_meta:
            if count == 0:
                continue
            folder_probs = all_probs[start:start + count]  # (count, n_races)
            rows = []
            for j, name in enumerate(filenames):
                pred_idx = folder_probs[j].argmax().item()
                rows.append({
                    "image_name": name,
                    "predicted_race": RACE_LABELS[pred_idx],
                    "confidence": round(folder_probs[j, pred_idx].item(), 6),
                    **{race: round(folder_probs[j, k].item(), 6)
                       for k, race in enumerate(RACE_LABELS)},
                })
            pd.DataFrame(rows).to_csv(os.path.join(occ_path, CSV_NAME), index=False)

        del all_imgs, all_probs, group_meta
        pbar.update(len(group))

    pbar.close()
    print("Done!")


if __name__ == "__main__":
    main()
