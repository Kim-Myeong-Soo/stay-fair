#!/usr/bin/env python3
"""Pick alpha for each occupation from the lookup table, given CFG probe images.

Generate n probe images per occupation with CFG at w=1.5 and w=7.5, evaluate them with
eval/eval_gender.py, then count female images per occupation and look the pair up.

Input CSV columns: occupation, n_female_w1.5, n_female_w7.5   (counts out of n probes)
Output: {occupation: alpha} JSON, usable as --gd_json for generate/vanilla/gen_image_sd3.py

Usage:
  python estimate/apply_lookup.py --probes probes.csv --table data/estimate/sd3_lookup_n5.csv --out sd3_estimate.json
"""
import argparse
import csv
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--table", default="data/estimate/sd3_lookup_n5.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = list(csv.reader(open(args.table)))
    table = [[float(x) for x in r[1:]] for r in rows[1:]]
    n = len(table) - 1

    alpha = {}
    for r in csv.DictReader(open(args.probes)):
        i, j = int(r["n_female_w1.5"]), int(r["n_female_w7.5"])
        assert 0 <= i <= n and 0 <= j <= n, (r, n)
        alpha[r["occupation"]] = table[i][j]
    json.dump(alpha, open(args.out, "w"), indent=2)
    print(f"Wrote {args.out} ({len(alpha)} occupations)")


if __name__ == "__main__":
    main()
