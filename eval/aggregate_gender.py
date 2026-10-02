#!/usr/bin/env python3
"""Aggregate per-image gender/person CSVs into male/female counts per leaf folder.

Reference eval scripts (reference/scripts/eval/) write, per leaf image folder:
  gender_classification.csv: image_name, predicted_gender, confidence, male_probability, female_probability
  person_detection.csv:      image_name, has_person, person_count, max_confidence, max_bbox_ratio

Inclusion filter (standing protocol): an image counts only if
  - YOLO detected a person (has_person == True), AND
  - gender-classifier confidence >= --conf (default 0.7)
Among included images, count predicted_gender male vs female.
Counts reported as #/N (out of included), per user preference.

Usage:
  conda activate eval
  python code/aggregate_gender.py --roots outputs/gd_wide outputs/gd_np outputs/gd_sink \
      --out analysis/gd_gender_counts.csv
"""
import argparse
import os
import re
import sys

import pandas as pd

GENDER_CSV = "gender_classification.csv"
PERSON_CSV = "person_detection.csv"


def find_leaf_folders(root):
    for dirpath, _dirs, files in os.walk(root):
        if GENDER_CSV in files:
            yield dirpath


def parse_gd(path):
    """Extract gd_scale from a path segment like g4_gd-400 -> -400, g4_gd0 -> 0."""
    m = re.search(r"gd(-?\d+)", path)
    return int(m.group(1)) if m else None


def parse_model(path):
    for m in ("flux", "zimage"):
        if f"/{m}/" in path.replace(os.sep, "/"):
            return m
    return "?"


def aggregate_folder(folder, conf):
    g = pd.read_csv(os.path.join(folder, GENDER_CSV))
    p_path = os.path.join(folder, PERSON_CSV)
    if os.path.exists(p_path):
        p = pd.read_csv(p_path)[["image_name", "has_person"]]
        df = g.merge(p, on="image_name", how="left")
        df["has_person"] = df["has_person"].fillna(False).astype(bool)
    else:
        df = g.copy()
        df["has_person"] = True  # no person data -> don't filter on it
    total = len(df)
    incl = df[df["has_person"] & (df["confidence"] >= conf)]
    male = int((incl["predicted_gender"] == "male").sum())
    female = int((incl["predicted_gender"] == "female").sum())
    return dict(total=total, included=len(incl), male=male, female=female)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--out", type=str, default=None, help="optional CSV path")
    args = ap.parse_args()

    rows = []
    for root in args.roots:
        for folder in sorted(find_leaf_folders(root)):
            r = aggregate_folder(folder, args.conf)
            rows.append(dict(
                root=os.path.basename(root.rstrip("/")),
                model=parse_model(folder),
                gd_scale=parse_gd(folder),
                folder=folder,
                **r,
            ))

    if not rows:
        print("No evaluated folders found.", file=sys.stderr)
        return
    out = pd.DataFrame(rows).sort_values(
        ["root", "model", "gd_scale"], kind="stable"
    )

    for (root, model), sub in out.groupby(["root", "model"], sort=False):
        print(f"\n=== {root} / {model}  (conf>={args.conf}, person filter) ===")
        print(f"{'gd_scale':>9} | {'male':>4} {'female':>6} | {'incl':>4}/{'tot':>3} | M/F")
        print("-" * 46)
        for _, r in sub.iterrows():
            print(f"{r['gd_scale']:>9} | {r['male']:>4} {r['female']:>6} | "
                  f"{r['included']:>4}/{r['total']:>3} | {r['male']}/{r['female']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
