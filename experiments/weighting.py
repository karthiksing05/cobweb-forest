"""Thm 1 / Prop 2 / Thm 2a: error of one learned tree vs N under the four weightings,
all-concept vs leaves-only mixtures, full expansion, and the Lemma-1 certificate.

    python experiments/weighting.py --fn sin --seed 0
Writes results/weighting/<fn>_s<seed>.json.
"""
import argparse
import json
import os

import numpy as np

from cobweb_forest import CobwebHybridTree, code_hashes

FNS = {"sin": lambda x: np.sin(np.pi * x), "j0": None}


def bessel_j0(x):
    t = np.linspace(0, np.pi, 801)
    return np.trapezoid(np.cos(np.outer(x, np.sin(t))), t, axis=1) / np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fn", default="sin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--Ns", default="100,200,400,800,1600,3200,6400")
    a = ap.parse_args()
    f = (lambda x: np.sin(np.pi * x)) if a.fn == "sin" else (lambda x: bessel_j0(20 * x))
    rng = np.random.default_rng(a.seed)
    xt = np.linspace(-1, 1, 401).astype(np.float32)
    interior = np.abs(xt) <= 0.9
    yt = f(xt)
    out = dict(fn=a.fn, seed=a.seed, code=code_hashes(), rows=[])
    for N in map(int, a.Ns.split(",")):
        x = rng.uniform(-1, 1, N).astype(np.float32)
        y = f(x).astype(np.float32)
        sx, sy = x.std(), y.std()
        for sched in ["interpolation", "regression"]:
            h = (2.0 / N) if sched == "interpolation" else N ** (-1 / 5)   # in x-std units
            t = CobwebHybridTree(2, alpha=0.1, prior_var=float(h ** 2), cont_weight=1.0)
            t.set_random_seed(a.seed)
            for xi, yi in zip(x, y):
                t.ifit({}, np.array([xi / sx, yi / sy], dtype=np.float32))
            for weighting in ["path", "prior", "likelihood", "posterior"]:
                for leaves in [False, True]:
                    P = np.array([t.predict_mean({}, np.array([q / sx, np.nan], dtype=np.float32),
                                                 10 ** 7, False, weighting, leaves)[1] * sy for q in xt])
                    err = P - yt
                    out["rows"].append(dict(N=N, sched=sched, h=h, weighting=weighting, leaves_only=leaves,
                                            rmse=float(np.sqrt(np.mean(err ** 2))),
                                            rmse_interior=float(np.sqrt(np.mean(err[interior] ** 2))),
                                            sup=float(np.abs(err).max()),
                                            concepts=t.num_concepts()))
            print(N, sched, [(r["weighting"], r["leaves_only"], round(r["rmse"], 4))
                             for r in out["rows"] if r["N"] == N and r["sched"] == sched], flush=True)
    os.makedirs("results/weighting", exist_ok=True)
    with open(f"results/weighting/{a.fn}_s{a.seed}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
