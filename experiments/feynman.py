"""Feynman equations (AI Feynman; Udrescu & Tegmark 2020), the KAN paper's Table 2 setting:
single tree vs Cobweb forests (structure-agnostic balanced graphs over the inputs) vs MLP/KAN.
Inputs are sampled uniformly from the AI-Feynman ranges and mapped to [-1, 1].

    python experiments/feynman.py --eq I.6.20 --N 4000 --seed 0
"""
import argparse
import json
import os
import time

import numpy as np

import toys
from cobweb_forest import code_hashes, designs as D

PI = np.pi
# name: (ranges per variable, formula on the physical variables)
EQS = {
    "I.6.20a": ([(1, 3)], lambda t: np.exp(-t ** 2 / 2) / np.sqrt(2 * PI)),
    "I.6.20": ([(1, 3), (1, 3)], lambda s, t: np.exp(-(t / s) ** 2 / 2) / (np.sqrt(2 * PI) * s)),
    "I.6.20b": ([(1, 3)] * 3, lambda s, t, t1: np.exp(-((t - t1) / s) ** 2 / 2) / (np.sqrt(2 * PI) * s)),
    "I.12.11": ([(1, 5)] * 5, lambda q, ef, b, v, th: q * (ef + b * v * np.sin(th))),
    "I.13.12": ([(1, 5)] * 5, lambda m1, m2, r1, r2, g: g * m1 * m2 * (1 / r2 - 1 / r1)),
    "I.29.16": ([(1, 5)] * 4, lambda x1, x2, t1, t2: np.sqrt(x1 ** 2 + x2 ** 2 - 2 * x1 * x2 * np.cos(t1 - t2))),
    "I.37.4": ([(1, 5)] * 3, lambda i1, i2, dl: i1 + i2 + 2 * np.sqrt(i1 * i2) * np.cos(dl)),
    "I.50.26": ([(1, 3)] * 4, lambda x1, w, t, al: x1 * (np.cos(w * t) + al * np.cos(w * t) ** 2)),
    "II.6.15a": ([(1, 3)] * 6, lambda ef, pd, r, x, y, z: pd / (4 * PI * ef) * 3 * z / r ** 5 * np.sqrt(x ** 2 + y ** 2)),
    "I.9.18": ([(1, 2)] * 2 + [(3, 4), (1, 2), (3, 4), (1, 2), (3, 4), (1, 2), (1, 2)],
               lambda m1, m2, x2, x1, y2, y1, z2, z1, g: g * m1 * m2 / ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", required=True)
    ap.add_argument("--N", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="single,act,label,hybrid,mlp,kan")
    a = ap.parse_args()
    ranges, phys = EQS[a.eq]
    n = len(ranges)

    def f(X):
        args = [lo + (X[:, i] + 1) / 2 * (hi - lo) for i, (lo, hi) in enumerate(ranges)]
        return phys(*args)

    rng = np.random.default_rng(2000 + a.seed)

    def sample(m):
        Xs = rng.uniform(-1, 1, (m, n)).astype(np.float32)
        return Xs, f(Xs)[:, None].astype(np.float32)

    data = (sample(a.N), sample(max(300, a.N // 4)), sample(2000))
    graph = D.balanced(range(n))
    out = dict(eq=a.eq, n=n, N=a.N, seed=a.seed, sd_f=float(data[2][1].std()), code=code_hashes(), results={})
    os.makedirs("results/feynman", exist_ok=True)
    path = f"results/feynman/{a.eq}_N{a.N}_s{a.seed}.json"
    for m in a.methods.split(","):
        t0 = time.time()
        if m == "single":
            out["results"][m] = toys.run_cobweb(
                lambda kappa: D.single(n, a.N, kappa=kappa, max_nodes=300, leaves_only=True),
                [dict(kappa=k) for k in toys.KAPPAS], data, "online", 300)
        elif m in ("mlp", "kan"):
            out["results"][m] = toys.run_torch(m, n, [n, max(2, n), 1], data, a.seed)
        elif n > 1:
            grid = ([dict(kappa=k) for k in toys.KAPPAS] if m == "act"
                    else [dict(kappa=k, label_depth=dp) for k in toys.KAPPAS for dp in toys.DEPTHS])
            out["results"][m] = toys.run_cobweb(
                lambda **hp: D.ka(graph, m, a.N, max_nodes=300, leaves_only=True, **hp), grid, data, "online", 300)
        print(a.eq, a.N, m, "%.0fs" % (time.time() - t0),
              {k: round(v["test"] / out["sd_f"], 4) for k, v in out["results"].items()}, flush=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
