"""MNIST with Cobweb forests (Convolutional-Cobweb style) vs one tree vs MLP / KAN.

Designs (each tree learns online with ifit; the class is a discrete attribute):
  single   one tree over the 784 pixels and the class
  label    16 unsupervised trees, one per 7x7 patch position, pass labels (depth k) to an
           output tree over the 16 labels and the class        (Convolutional Cobweb)
  act      16 patch trees that model the class jointly pass their predicted class
           distributions (activations) to an output tree over the 160 values and the class
  hybrid   16 class-modelling patch trees pass labels and activations
  cross    16 patch trees that never see the class predict the centre pixels of their
           neighbouring patches (perceptual prediction) and pass those activations

    python experiments/mnist.py --N 5000 --seed 0 --methods single,label,act,hybrid,mlp,kan
"""
import argparse
import json
import os
import time

import numpy as np

from cobweb_forest import Forest, TreeSpec, code_hashes


def patches(grid=4, side=28):
    k = side // grid
    out = []
    for r in range(grid):
        for c in range(grid):
            out.append([(r * k + i) * side + (c * k + j) for i in range(k) for j in range(k)])
    return out


def design(mode, pv, depth, max_nodes):
    kw = dict(max_nodes=max_nodes, leaves_only=True, alpha=0.01)
    if mode == "single":
        return [TreeSpec("out", raw=list(range(784)), task="class", prior_var=pv, **kw)]
    specs, names = [], []
    for i, idx in enumerate(patches()):
        name = f"p{i}"
        names.append(name)
        if mode == "cross":
            # perceptual prediction: the tree never sees the class; it predicts the centre
            # pixels of the neighbouring patches (a cross-view outcome, Def. cognitive task)
            r, c = divmod(i, 4)
            nb = [(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                  if 0 <= r + dr < 4 and 0 <= c + dc < 4]
            outs = [(r2 * 7 + 3) * 28 + (c2 * 7 + 3) for r2, c2 in nb]
            specs.append(TreeSpec(name, raw=idx, task="raw:" + "+".join(map(str, outs)),
                                  emits=["act"], prior_var=pv, **kw))
            continue
        if mode == "label":
            specs.append(TreeSpec(name, raw=idx, emits=["label"], label_depth=depth, prior_var=pv, **kw))
        else:
            specs.append(TreeSpec(name, raw=idx, task="class", prior_var=pv, label_depth=depth,
                                  emits=["act"] if mode == "act" else ["act", "label"], **kw))
    specs.append(TreeSpec("out", task="class", prior_var=pv,
                          act_from=names if mode in ("act", "hybrid", "cross") else [],
                          label_from=names if mode in ("label", "hybrid") else [], **kw))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="single,label,act,hybrid,mlp,kan")
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=2000)
    a = ap.parse_args()
    d = np.load("data/mnist.npz")
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(d["xtr"]))
    tr, va = perm[:a.N], perm[a.N:a.N + a.n_val]
    te = rng.permutation(len(d["xte"]))[:a.n_test]
    X = d["xtr"][tr].reshape(-1, 784).astype(np.float32) / 255
    C = d["ytr"][tr].astype(int)
    Xv = d["xtr"][va].reshape(-1, 784).astype(np.float32) / 255
    Cv = d["ytr"][va].astype(int)
    Xt = d["xte"][te].reshape(-1, 784).astype(np.float32) / 255
    Ct = d["yte"][te].astype(int)
    out = dict(N=a.N, seed=a.seed, code=code_hashes(), results={})
    os.makedirs("results/mnist", exist_ok=True)
    path = f"results/mnist/N{a.N}_s{a.seed}_{a.methods.replace(',', '+')}.json"
    for m in a.methods.split(","):
        t0 = time.time()
        if m in ("mlp", "kan"):
            import torch
            from cobweb_forest import baselines as B
            torch.set_num_threads(4)
            best = None
            for hp in ([dict(width=w) for w in (128, 256)] if m == "mlp" else [dict(grid=g) for g in (3, 5)]):
                model = B.mlp(784, 10, width=hp["width"], depth=3) if m == "mlp" else B.KAN([784, 32, 10], grid=hp["grid"])
                model = B.train(model, X * 2 - 1, C, Xv * 2 - 1, Cv, task="cls", seed=a.seed, epochs=300, patience=20)
                v = float(np.mean(B.predict(model, Xv * 2 - 1).argmax(1) == Cv))
                if best is None or v > best["val"]:
                    best = dict(hp=hp, val=v, model=model)
            model = best.pop("model")
            best["test"] = float(np.mean(B.predict(model, Xt * 2 - 1).argmax(1) == Ct))
            best["params"] = B.n_params(model)
        else:
            grid = [dict(pv=pv, depth=dp) for pv in (0.1, 0.3, 1.0)
                    for dp in ((3,) if m in ("single", "act", "cross") else (2, 4, 6))]
            best = None
            for hp in grid:
                F = Forest(design(m, hp["pv"], hp["depth"], 200), "out", n_classes=10, raw_scale="global")
                F.fit(X, None, C=C, protocol="online")
                v = float(np.mean(F.predict(Xv) == Cv))
                if best is None or v > best["val"]:
                    best = dict(hp=hp, val=v, F=F)
            F = best.pop("F")
            best["test"] = float(np.mean(F.predict(Xt) == Ct))
            best["concepts"] = int(sum(F.num_concepts().values()))
        best["seconds"] = time.time() - t0
        out["results"][m] = best
        print(a.N, a.seed, m, {k: (round(v["test"], 4), v["hp"]) for k, v in out["results"].items()}, flush=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
