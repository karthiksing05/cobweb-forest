"""Continual learning toy of Liu et al. (2024, Fig. 3.4): a 1-D target with five Gaussian
peaks; the data arrive in five phases, phase k covering only the k-th fifth of the domain.
Lemma (leaf preservation) predicts that a Cobweb tree never loses what earlier phases
taught it.  MLP and KAN are trained phase by phase on the current phase only.

    python experiments/continual.py --seed 0
Writes results/continual/s<seed>.json (predictions after every phase).
"""
import argparse
import json
import os

import numpy as np

from cobweb_forest import Forest, code_hashes, designs as D

CENTRES = np.linspace(-0.8, 0.8, 5)


def f(x):
    return np.sum(np.exp(-((x[:, None] - CENTRES) ** 2) / (2 * 0.06 ** 2)), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per_phase", type=int, default=400)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    edges = np.linspace(-1, 1, 6)
    phases = []
    for k in range(5):
        x = rng.uniform(edges[k], edges[k + 1], a.per_phase).astype(np.float32)
        phases.append((x[:, None], f(x)[:, None].astype(np.float32)))
    xt = np.linspace(-1, 1, 501).astype(np.float32)[:, None]
    yt = f(xt[:, 0])
    out = dict(seed=a.seed, code=code_hashes(), x=xt[:, 0].tolist(), y=yt.tolist(), methods={})

    # Cobweb: one tree, learning online; scales fixed from the domain, not from the data seen.
    N = 5 * a.per_phase
    F = Forest(D.single(1, N, kappa=0.25, max_nodes=300, leaves_only=True), "out")
    t = F.trees["out"]
    t.mu = np.array([0.0, 0.5], dtype=np.float32)
    t.sd = np.array([0.58, 0.4], dtype=np.float32)
    preds = []
    for X, Y in phases:
        for xi, yi in zip(X, Y):
            t.ifit(xi, np.zeros(0, np.float32), [], None, y=yi)
        preds.append(F.predict(xt)[:, 0].tolist())
    out["methods"]["cobweb"] = preds

    import torch
    from cobweb_forest import baselines as B
    torch.set_num_threads(2)
    for kind in ["mlp", "kan"]:
        torch.manual_seed(a.seed)
        model = B.mlp(1, 1, width=64, depth=3) if kind == "mlp" else B.KAN([1, 1], grid=200)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        preds = []
        for X, Y in phases:
            Xt_, Yt_ = torch.tensor(X), torch.tensor(Y)
            for _ in range(2000):
                opt.zero_grad()
                torch.nn.functional.mse_loss(model(Xt_), Yt_).backward()
                opt.step()
            preds.append(B.predict(model, xt)[:, 0].tolist())
        out["methods"][kind] = preds

    # forgetting: error on phases 1..k after phase k
    out["summary"] = {}
    for m, P in out["methods"].items():
        rows = []
        for k in range(5):
            seen = (xt[:, 0] <= edges[k + 1])
            rows.append(float(np.sqrt(np.mean((np.array(P[k])[seen] - yt[seen]) ** 2))))
        out["summary"][m] = dict(rmse_seen_after_phase=rows,
                                 final_rmse=float(np.sqrt(np.mean((np.array(P[-1]) - yt) ** 2))))
    print(json.dumps(out["summary"], indent=1))
    os.makedirs("results/continual", exist_ok=True)
    with open(f"results/continual/s{a.seed}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
