"""Thm (label-forest consistency), assumption (a): do the cells of Cobweb's labels shrink?

An unsupervised tree learns N uniform points in [0,1]^d (d = 1, 2); for 500 test points we
record the label at depth k (root-to-leaf descent by log P(c|x)) and the spread (RMS distance
to the cell mean) of the training points that share it, and the number of occupied cells.
    python experiments/label_cells.py --seed 0
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

from cobweb_forest import Forest, TreeSpec, code_hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = []
    for d in (1, 2):
        for N in (250, 1000, 4000, 16000):
            rng = np.random.default_rng(a.seed)
            X = rng.uniform(0, 1, (N, d)).astype(np.float32)
            F = Forest([TreeSpec("t", raw=list(range(d)), emits=["label"], prior_var=(N ** (-1 / (d + 4))) ** 2,
                                 seed=a.seed)], "t").fit(X, None, protocol="online")
            t = F.trees["t"]
            Xt = rng.uniform(0, 1, (500, d)).astype(np.float32)
            for k in (1, 2, 3, 4, 6, 8, 12, None):
                t.spec.label_depth = k
                lab_tr = [t.label(x, np.zeros(0), [], None) for x in X]
                cells = defaultdict(list)
                for x, l in zip(X, lab_tr):
                    cells[l].append(x)
                spread = {l: float(np.sqrt(np.mean(np.sum((np.array(v) - np.mean(v, 0)) ** 2, 1)))) for l, v in cells.items()}
                lab_te = [t.label(x, np.zeros(0), [], None) for x in Xt]
                rows.append(dict(d=d, N=N, depth=-1 if k is None else k, cells=len(cells),
                                 mean_spread=float(np.mean([spread.get(l, np.nan) for l in lab_te])),
                                 mean_cell_size=float(np.mean([len(cells.get(l, [])) for l in lab_te]))))
                print(rows[-1], flush=True)
    os.makedirs("results/label_cells", exist_ok=True)
    with open(f"results/label_cells/s{a.seed}.json", "w", encoding="utf-8") as fh:
        json.dump(dict(code=code_hashes(), rows=rows), fh, indent=1)


if __name__ == "__main__":
    main()
