#!/usr/bin/env python3
"""Per-image person detection using YOLOv11.

Saves person_detection.csv in each occupation folder:
    image_name, has_person, person_count, max_confidence, max_bbox_ratio

max_bbox_ratio: area of largest person bbox / image area (0~1)
"""

import argparse
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(__file__))
from utils import add_common_args, setup_gpu, find_occupation_dirs, load_images

CSV_NAME = "person_detection.csv"
PERSON_CLASS_ID = 0  # COCO class 0 = person




def main():
    parser = argparse.ArgumentParser(description="Per-image person detection (YOLOv11)")
    add_common_args(parser)
    parser.add_argument("--yolo_model", type=str, default="yolo11n.pt",
                        help="YOLO model variant (yolo11n/s/m/l/x.pt)")
    parser.add_argument("--conf_threshold", type=float, default=0.25,
                        help="YOLO confidence threshold")
    parser.add_argument("--group_size", type=int, default=5,
                        help="Number of folders to batch together")
    args = parser.parse_args()
    use_fp16 = args.fp16 and not args.no_fp16

    setup_gpu(args.gpu)

    print(f"Loading YOLO model: {args.yolo_model} (fp16={use_fp16})")
    model = YOLO(args.yolo_model)

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

        # Run YOLO over all images in group
        all_results = []
        for i in range(0, len(all_imgs), args.batch_size):
            batch_imgs = all_imgs[i:i + args.batch_size]
            results = model(batch_imgs, conf=args.conf_threshold,
                            classes=[PERSON_CLASS_ID], verbose=False,
                            half=use_fp16)
            all_results.extend(results)
            del batch_imgs

        # Save per-folder CSVs
        for occ_path, filenames, start, count in group_meta:
            if count == 0:
                continue
            rows = []
            for j, name in enumerate(filenames):
                res = all_results[start + j]
                boxes = res.boxes
                person_mask = boxes.cls == PERSON_CLASS_ID
                person_count = int(person_mask.sum())
                max_conf = float(boxes.conf[person_mask].max()) if person_count > 0 else 0.0

                max_bbox_ratio = 0.0
                if person_count > 0:
                    img_h, img_w = res.orig_shape
                    img_area = img_h * img_w
                    xyxy = boxes.xyxy[person_mask]
                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    max_bbox_ratio = float(areas.max()) / img_area

                rows.append({
                    "image_name": name,
                    "has_person": person_count > 0,
                    "person_count": person_count,
                    "max_confidence": round(max_conf, 6),
                    "max_bbox_ratio": round(max_bbox_ratio, 6),
                })
            pd.DataFrame(rows).to_csv(os.path.join(occ_path, CSV_NAME), index=False)

        del all_imgs, all_results, group_meta
        pbar.update(len(group))

    pbar.close()
    print("Done!")


if __name__ == "__main__":
    main()
