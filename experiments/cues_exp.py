"""Cue calculus in practice: estimate validities with Cobweb trees, classify cues
(associative / contextual / blocked / irrelevant), build the forest the classification calls
for, verify it at the output (cobweb_forest.cues).

  synth   -- the suite of check_cues with known kinds, over seeds
  housing -- California housing, 8 features, test MSE of the cue forest vs single trees
    python experiments/cues_exp.py --task synth --seed 0
"""
import argparse
import json
import os
import time

import numpy as np

from cobweb_forest import code_hashes, cues

NAMES_H = ["lon", "lat", "age", "rooms", "bedrooms", "pop", "occup", "income"]


def synth(seed, N=3000):
    r = np.random.default_rng(seed)

    def draw(n):
        X = r.uniform(-1, 1, size=(n, 5))
        X[:, 4] = X[:, 0] + 0.01 * r.standard_normal(n)
        y = np.sin(np.pi * X[:, 0]) + np.sign(X[:, 1] * X[:, 2]) + 0.05 * r.standard_normal(n)
        return X.astype(np.float32), y[:, None].astype(np.float32)
    return draw(N), draw(N // 2), draw(N), ["assoc", "ctx1", "ctx2", "noise", "copy"]


def housing(seed, N=4000):
    d = np.load("data/housing.npz")
    X, y = d["X"], d["y"]
    perm = np.random.default_rng(seed).permutation(len(X))
    te, va, tr = perm[:3000], perm[3000:4000], perm[4000:4000 + N]
    m, s = X[tr].mean(0), X[tr].std(0)
    Z = (((X - m) / s).clip(-4, 4) / 4).astype(np.float32)
    f = lambda i: (Z[i], y[i, None].astype(np.float32))
    return f(tr), f(va), f(te), NAMES_H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="synth")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    (X, y), (Xv, yv), (Xt, yt), names = (synth if a.task == "synth" else housing)(a.seed)
    rep = cues.estimate(X, y, Xv, yv, names=names)
    best, F, cols, val = cues.cue_forest(X, y, Xv, yv, rep)
    # test MSE of every candidate, refit on train (cue_forest returns only the chosen one)
    from cobweb_forest import Forest, designs as D
    test = {best: float(np.mean((F.predict(Xt[:, cols])[:, 0] - yt[:, 0]) ** 2))}
    Fa = Forest(D.single(X.shape[1], len(X), kappa=0.25, **cues.KW), "out").fit(X, y, protocol="online")
    test["single-all"] = float(np.mean((Fa.predict(Xt)[:, 0] - yt[:, 0]) ** 2))
    out = dict(task=a.task, seed=a.seed, code=code_hashes(), var_y=float(np.var(yt)),
               kinds={names[i]: k for i, k in rep.kind.items()},
               blocked_by={names[i]: names[j] for i, j in rep.blocked_by.items()},
               validity={"+".join(names[i] for i in k): v for k, v in rep.validity.items()},
               synergy={f"{names[i]}+{names[j]}": rep.synergy(i, j) for (i, j) in rep.funnel_validity},
               groups=[[names[i] for i in g] for g in cues.groups(rep)],
               chosen=best, cols=[names[i] for i in cols], val=val, test=test,
               seconds=time.time() - t0)
    os.makedirs("results/cues", exist_ok=True)
    with open(f"results/cues/{a.task}_s{a.seed}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(rep.table(), "\n", best, out["groups"], test, flush=True)


if __name__ == "__main__":
    main()
