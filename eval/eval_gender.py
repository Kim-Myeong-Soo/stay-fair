#!/usr/bin/env python3
"""Per-image gender classification using CLIP.

Saves gender_classification.csv in each occupation folder:
    image_name, predicted_gender, confidence, male_probability, female_probability
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

CSV_NAME = "gender_classification.csv"
ATTRIBUTES = ["a photo of a male person", "a photo of a female person"]




def main():
    parser = argparse.ArgumentParser(description="Per-image gender classification (CLIP)")
    add_common_args(parser)
    parser.add_argument("--group_size", type=int, default=5,
                        help="Number of folders to batch together")
    args = parser.parse_args()
    use_fp16 = args.fp16 and not args.no_fp16

    setup_gpu(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if use_fp16 else torch.float32

    print(f"Loading CLIP model... (fp16={use_fp16})")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                      torch_dtype=dtype).to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Pre-encode text attributes once
    text_inputs = processor(text=ATTRIBUTES, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs).to(dtype)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    del text_inputs

    occ_dirs = list(find_occupation_dirs(args.root))
    todo = [d for d in occ_dirs if not os.path.exists(os.path.join(d, CSV_NAME)) or args.force]
    print(f"Found {len(occ_dirs)} folders, {len(todo)} to process")

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

        all_probs = []
        for i in range(0, len(all_imgs), args.batch_size):
            batch_imgs = all_imgs[i:i + args.batch_size]

            img_inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_fp16):
                image_features = model.get_image_features(
                    pixel_values=img_inputs["pixel_values"].to(dtype)
                )
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                logit_scale = model.logit_scale.exp()
                logits = logit_scale * (image_features @ text_features.T)
                probs = logits.softmax(dim=1).float().cpu()

            all_probs.append(probs)
            del img_inputs, batch_imgs

        all_probs = torch.cat(all_probs, dim=0)  # (total_imgs, 2)

        for occ_path, filenames, start, count in group_meta:
            if count == 0:
                continue
            folder_probs = all_probs[start:start + count]
            rows = []
            for j, name in enumerate(filenames):
                male_p = folder_probs[j, 0].item()
                female_p = folder_probs[j, 1].item()
                pred = "male" if male_p > female_p else "female"
                conf = max(male_p, female_p)
                rows.append({
                    "image_name": name,
                    "predicted_gender": pred,
                    "confidence": round(conf, 6),
                    "male_probability": round(male_p, 6),
                    "female_probability": round(female_p, 6),
                })
            pd.DataFrame(rows).to_csv(os.path.join(occ_path, CSV_NAME), index=False)

        del all_imgs, all_probs, group_meta
        pbar.update(len(group))

    pbar.close()
    print("Done!")


if __name__ == "__main__":
    main()
