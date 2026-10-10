"""Toy battery mirroring Liu et al. (2024, Sec. 3): single tree vs Cobweb forests vs MLP / KAN.

    python experiments/toys.py --fn expsin --N 4000 --seed 0 [--methods ...] [--noise 0.0]

Each method's hyperparameters are chosen on a fresh validation set (toy data is free);
test RMSE is on 2000 fresh points.  Writes results/toys/<fn>_N<N>_s<seed>[_noise].json.
"""
import argparse
import json
import os
import time

import numpy as np

from cobweb_forest import Forest, code_hashes, designs as D


def bessel_j0(x):
    t = np.linspace(0, np.pi, 801)
    return np.trapezoid(np.cos(np.outer(x, np.sin(t))), t, axis=1) / np.pi


# name: (n, f, KA graphs {label: graph}, KAN widths)
FNS = {
    "sin": (1, lambda X: np.sin(np.pi * X[:, 0]), {}, [1, 1]),
    "j0": (1, lambda X: bessel_j0(20 * X[:, 0]), {}, [1, 1]),
    "expsin": (2, lambda X: np.exp(np.sin(np.pi * X[:, 0]) + X[:, 1] ** 2),
               {"uni": (0, 1)}, [2, 1, 1]),
    "xy": (2, lambda X: X[:, 0] * X[:, 1], {"uni": (0, 1)}, [2, 2, 1]),
    "nested4": (4, lambda X: np.exp((np.sin(np.pi * (X[:, 0] ** 2 + X[:, 1] ** 2)) +
                                     np.sin(np.pi * (X[:, 2] ** 2 + X[:, 3] ** 2))) / 2),
                {"pairs": ([0, 1], [2, 3]), "uni": ((0, 1), (2, 3))}, [4, 4, 2, 1, 1]),
    "exp100": (100, lambda X: np.exp(np.mean(np.sin(np.pi * X / 2) ** 2, 1)),
               {"bal": D.balanced(range(100))}, [100, 1, 1]),
}

KAPPAS = [0.1, 0.25, 0.5, 1.0]
DEPTHS = [2, 3, 4, 6]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a).ravel() - np.asarray(b).ravel()) ** 2)))


def run_cobweb(specs_fn, grid, data, protocol, max_nodes):
    (X, Y), (Xv, Yv), (Xt, Yt) = data
    best = None
    for hp in grid:
        t0 = time.time()
        F = Forest(specs_fn(**hp), "out").fit(X, Y, protocol=protocol)
        v = rmse(F.predict(Xv), Yv)
        rec = dict(hp=hp, val=v, train_s=time.time() - t0, F=F)
        if best is None or v < best["val"]:
            best = rec
    F = best.pop("F")
    t0 = time.time()
    best["test"] = rmse(F.predict(Xt), Yt)
    best["test_s"] = time.time() - t0
    best["concepts"] = int(sum(F.num_concepts().values()))
    best["trees"] = len(F.trees)
    best["protocol"] = protocol
    best["grid"] = grid
    return best


def run_torch(kind, n, widths, data, seed):
    import torch
    from cobweb_forest import baselines as B
    torch.set_num_threads(2)
    (X, Y), (Xv, Yv), (Xt, Yt) = data
    ym, ys = float(Y.mean()), float(Y.std() + 1e-8)
    grid = ([dict(width=w) for w in (32, 64, 128)] if kind == "mlp"
            else [dict(grid=g) for g in (5, 10, 20)])
    best = None
    for hp in grid:
        t0 = time.time()
        model = (B.mlp(n, 1, width=hp["width"], depth=3) if kind == "mlp"
                 else B.KAN(widths, grid=hp["grid"]))
        model = B.train(model, X, (Y - ym) / ys, Xv, (Yv - ym) / ys, seed=seed,
                        epochs=3000 if len(X) <= 4000 else 1000)
        v = rmse(B.predict(model, Xv) * ys + ym, Yv)
        if best is None or v < best["val"]:
            best = dict(hp=hp, val=v, train_s=time.time() - t0, model=model)
    model = best.pop("model")
    best["test"] = rmse(B.predict(model, Xt) * ys + ym, Yt)
    best["params"] = B.n_params(model)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fn", required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0, help="noise sd as a fraction of sd(f)")
    ap.add_argument("--methods", default="single,act,label,hybrid,act-split,mlp,kan")
    ap.add_argument("--out", default="results/toys")
    a = ap.parse_args()

    n, f, graphs, widths = FNS[a.fn]
    rng = np.random.default_rng(1000 + a.seed)
    sd = float(np.std(f(rng.uniform(-1, 1, (20000, n)))))

    def sample(m, noisy):
        Xs = rng.uniform(-1, 1, (m, n)).astype(np.float32)
        Ys = f(Xs)
        if noisy and a.noise > 0:
            Ys = Ys + rng.normal(0, a.noise * sd, m)
        return Xs, Ys[:, None].astype(np.float32)

    data = (sample(a.N, True), sample(max(300, a.N // 4), True), sample(2000, False))
    max_nodes = 100 if n > 10 else 300
    kappas = [0.25, 1.0] if n > 10 else KAPPAS
    depths = [3] if n > 10 else DEPTHS
    out = dict(fn=a.fn, N=a.N, seed=a.seed, noise=a.noise, n=n, sd_f=sd,
               code=code_hashes(), results={})
    path = os.path.join(a.out, f"{a.fn}_N{a.N}_s{a.seed}" + (f"_noise{a.noise}" if a.noise else "") + ".json")
    os.makedirs(a.out, exist_ok=True)

    def save():
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=float)

    for m in a.methods.split(","):
        t0 = time.time()
        if m == "single":
            r = run_cobweb(lambda kappa: D.single(n, a.N, kappa=kappa, max_nodes=max_nodes, leaves_only=True),
                           [dict(kappa=k) for k in kappas], data, "online", max_nodes)
            out["results"][m] = r
        elif m in ("mlp", "kan"):
            out["results"][m] = run_torch(m, n, widths, data, a.seed)
        else:
            mode, proto = (m.split("-") + ["online"])[:2]
            for gname, graph in graphs.items():
                grid = ([dict(kappa=k) for k in kappas] if mode == "act"
                        else [dict(kappa=k, label_depth=d) for k in kappas for d in depths])
                r = run_cobweb(lambda graph=graph, **hp: D.ka(graph, mode, a.N, max_nodes=max_nodes,
                                                               leaves_only=True, **hp),
                               grid, data, proto, max_nodes)
                out["results"][f"{m}:{gname}"] = r
        print(a.fn, a.N, a.seed, m, "%.0fs" % (time.time() - t0),
              {k: round(v["test"], 4) for k, v in out["results"].items()}, flush=True)
        save()


if __name__ == "__main__":
    main()
