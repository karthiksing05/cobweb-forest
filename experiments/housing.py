"""California housing (log median value from 8 features): single tree vs forests vs MLP/KAN.

Forest graphs: "pairs" -- trees over the natural feature pairs (lon,lat), (age,rooms),
(bedrooms,occupancy), (households,income), then pairs of those; "uni" -- univariate trees
on every feature, balanced pairs above them.
    python experiments/housing.py --N 4000 --seed 0
"""
import argparse
import json
import os
import time

import numpy as np

import toys
from cobweb_forest import code_hashes, designs as D

GRAPHS = {"pairs": (([0, 1], [2, 3]), ([4, 5], [6, 7])), "uni": D.balanced(range(8))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="single,act,label,hybrid,mlp,kan")
    a = ap.parse_args()
    d = np.load("data/housing.npz")
    X, y = d["X"], d["y"]
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(X))
    te, va, tr = perm[:3000], perm[3000:4000], perm[4000:4000 + a.N]
    m, s = X[tr].mean(0), X[tr].std(0)
    Z = ((X - m) / s).clip(-4, 4).astype(np.float32)       # KAN grids live on [-1,1]: scale below
    Z = (Z / 4).astype(np.float32)
    data = ((Z[tr], y[tr, None]), (Z[va], y[va, None]), (Z[te], y[te, None]))
    out = dict(N=a.N, seed=a.seed, sd_y=float(y[te].std()), code=code_hashes(), results={})
    os.makedirs("results/housing", exist_ok=True)
    path = f"results/housing/N{a.N}_s{a.seed}.json"
    for meth in a.methods.split(","):
        t0 = time.time()
        if meth == "single":
            out["results"][meth] = toys.run_cobweb(
                lambda kappa: D.single(8, a.N, kappa=kappa, max_nodes=300, leaves_only=True),
                [dict(kappa=k) for k in toys.KAPPAS], data, "online", 300)
        elif meth in ("mlp", "kan"):
            out["results"][meth] = toys.run_torch(meth, 8, [8, 8, 1], data, a.seed)
        else:
            for gname, graph in GRAPHS.items():
                grid = ([dict(kappa=k) for k in toys.KAPPAS] if meth == "act"
                        else [dict(kappa=k, label_depth=dp) for k in toys.KAPPAS for dp in toys.DEPTHS])
                out["results"][f"{meth}:{gname}"] = toys.run_cobweb(
                    lambda graph=graph, **hp: D.ka(graph, meth, a.N, max_nodes=300, leaves_only=True, **hp),
                    grid, data, "online", 300)
        print(a.N, a.seed, meth, "%.0fs" % (time.time() - t0),
              {k: round(v["test"], 4) for k, v in out["results"].items()}, flush=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
