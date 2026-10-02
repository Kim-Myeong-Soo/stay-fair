#!/usr/bin/env python3
"""Build the StayFair (Estimate) lookup table for SD3.

Reference set: 132 prompts (4 templates x 33 occupations).
  data/estimate/sd3_reference_features.csv : CFG female count out of 100 raw images at w=1.5 and w=7.5
  data/estimate/sd3_reference_curves.csv   : female ratio at w in {1.5,3,4.5,6,7.5} for each candidate alpha
                                             (alpha = 0 is CFG)

For each query point (female ratio at w=1.5, female ratio at w=7.5):
  - K=10 nearest reference prompts in that 2-D space, weight 1/(distance + 0.01)
  - for each candidate alpha, weighted mean of the bias-range reduction vs CFG over the neighbours
    where that alpha was measured; eligible if >= 5 measured neighbours and >= 50% of the weight
  - pick the eligible alpha with the largest positive reduction (ties -> smaller |alpha|), else 0

The table is indexed by the number of female images among n probes per scale (default n=5,
so a 6x6 table).

Usage:
  python estimate/build_lookup.py --n 5 --out data/estimate/sd3_lookup_n5.csv
"""
import argparse
import collections
import csv
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHAS = [0, -200, -150, -100, -50, 50]
WS = [1.5, 3.0, 4.5, 6.0, 7.5]
K = 10


def load_references(features_csv, curves_csv):
    curves = {}
    for r in csv.DictReader(open(curves_csv)):
        curves[(int(r["prompt_version"]), r["occupation"], float(r["alpha"]), float(r["guidance_scale"]))] = (
            int(r["n_included"]), int(r["n_female"]))
    feats = collections.defaultdict(dict)
    for r in csv.DictReader(open(features_csv)):
        feats[(int(r["prompt_version"]), r["occupation"])][float(r["guidance_scale"])] = int(r["n_female"]) / int(r["n_total"])

    def bias_range(v, o, a):
        cells = [curves.get((v, o, float(a), w)) for w in WS]
        if any(c is None or c[0] <= 0 for c in cells):
            return None
        ratios = [100 * f / n for n, f in cells]
        return max(ratios) - min(ratios)

    refs = []
    for (v, o), f in sorted(feats.items()):
        refs.append(dict(v=v, occ=o, w15=f[1.5], w75=f[7.5], cfg_range=bias_range(v, o, 0),
                         ranges={a: bias_range(v, o, a) for a in ALPHAS}))
    return refs


def query(refs, q):
    near = sorted([(math.hypot(f["w15"] - q[0], f["w75"] - q[1]), f) for f in refs],
                  key=lambda x: (x[0], x[1]["v"], x[1]["occ"]))[:K]
    weights = [1 / (d + .01) for d, _ in near]
    total = sum(weights)
    cands = []
    for a in ALPHAS:
        obs = [(f, w) for (_, f), w in zip(near, weights) if f["ranges"][a] is not None]
        mass = sum(w for _, w in obs) / total
        gain = sum(w * (f["cfg_range"] - f["ranges"][a]) for f, w in obs) / sum(w for _, w in obs) if obs else None
        cands.append(dict(alpha=a, gain=gain, eligible=a == 0 or (len(obs) >= 5 and mass >= .5)))
    best = max([c for c in cands if c["eligible"]], key=lambda c: (c["gain"], -abs(c["alpha"])))
    return best["alpha"] if best["gain"] > 1e-10 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="probe images per scale")
    ap.add_argument("--features", default=os.path.join(ROOT, "data/estimate/sd3_reference_features.csv"))
    ap.add_argument("--curves", default=os.path.join(ROOT, "data/estimate/sd3_reference_curves.csv"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    refs = load_references(args.features, args.curves)
    n = args.n
    table = [[query(refs, (i / n, j / n)) for j in range(n + 1)] for i in range(n + 1)]
    header = ["w1.5 / w7.5"] + [f"{100 * j / n:g}%" for j in range(n + 1)]
    rows = [[f"{100 * i / n:g}%"] + table[i] for i in range(n + 1)]
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
        print(f"Wrote {args.out}")
    for r in [header] + rows:
        print(",".join(str(x) for x in r))


if __name__ == "__main__":
    main()
