"""Next-word prediction on OpenWebText (2M words, 2000-word vocabulary) from the previous
four words: single tree vs Cobweb forests vs an interpolated n-gram and an MLP LM.

Designs (words are nominal attributes; the next word is the task attribute):
  single        one tree over (w-1, w-2, w-3, w-4, next)
  label-unsup   trees over (w-1, w-2) and (w-3, w-4), unsupervised, pass labels at depth k
                to an output tree over (label_a, label_b, w-1, next)
  label-funnel  the same pair trees also model `next` (the target funnels through them)
  act-funnel    pair trees pass their predicted next-word distribution restricted to the
                top-K words (activations) -- too wide for 2000 classes; omitted
    python experiments/owt.py --N 100000 --seed 0
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np

from cobweb_forest import Forest, TreeSpec, code_hashes

CTX = 4


def windows(ids, start, n):
    X = np.stack([ids[start + CTX - 1 - k: start + CTX - 1 - k + n] for k in range(CTX)], 1)  # w-1..w-4
    y = ids[start + CTX: start + CTX + n]
    return X.astype(np.int64), y.astype(np.int64)


def design(mode, depth, max_nodes, alpha, weighting="prior"):
    kw = dict(max_nodes=max_nodes, alpha=alpha, leaves_only=False, weighting=weighting)
    if mode == "single":
        return [TreeSpec("out", raw_disc=[0, 1, 2, 3], task="class", **kw)]
    task = "class" if mode == "label-funnel" else None
    return [TreeSpec("a", raw_disc=[0, 1], task=task, emits=["label"], label_depth=depth, **kw),
            TreeSpec("b", raw_disc=[2, 3], task=task, emits=["label"], label_depth=depth, **kw),
            TreeSpec("out", label_from=["a", "b"], raw_disc=[0], task="class", **kw)]


def evaluate(probs, y, V):
    acc = float(np.mean([max(p, key=p.get) == t if p else False for p, t in zip(probs, y)]))
    ll = float(np.mean([-np.log(max(p.get(int(t), 0.0), 1e-12)) for p, t in zip(probs, y)]))
    return acc, ll


def ngram(X, y, Xt, yt, V, d=0.75):
    """Interpolated absolute-discounting n-gram (orders 1..5)."""
    counts = [defaultdict(Counter) for _ in range(CTX + 1)]
    for row, t in zip(X, y):
        for o in range(CTX + 1):
            counts[o][tuple(row[:o])][int(t)] += 1
    uni = counts[0][()]
    tot = sum(uni.values())

    def prob(row, t):
        p = (uni[t] + 1) / (tot + V)
        for o in range(1, CTX + 1):
            c = counts[o].get(tuple(row[:o]))
            if not c:
                break
            n = sum(c.values())
            p = max(c[t] - d, 0) / n + d * len(c) / n * p
        return p
    acc_hits, ll = 0, 0.0
    for row, t in zip(Xt, yt):
        ll -= np.log(prob(row, int(t)))
    # accuracy: argmax over the highest-order context's candidates (cheap approximation)
    for row, t in zip(Xt, yt):
        best = None
        for o in range(CTX, -1, -1):
            c = counts[o].get(tuple(row[:o]))
            if c:
                best = c.most_common(1)[0][0]
                break
        acc_hits += int(best == t)
    return acc_hits / len(yt), ll / len(yt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="ngram,single,label-unsup,label-funnel,mlp")
    ap.add_argument("--n_test", type=int, default=10000)
    a = ap.parse_args()
    d = np.load("data/owt.npz")
    ids, V = d["ids"], len(d["vocab"])
    # train on the start of the stream, validate/test on its end (disjoint documents)
    X, y = windows(ids, 0, a.N)
    Xv, yv = windows(ids, len(ids) - 2 * a.n_test - 2 * CTX, 3000)
    Xt, yt = windows(ids, len(ids) - a.n_test - CTX - 1, a.n_test)
    out = dict(N=a.N, seed=a.seed, V=V, code=code_hashes(), results={})
    os.makedirs("results/owt", exist_ok=True)
    path = f"results/owt/N{a.N}_s{a.seed}_{a.methods.replace(',', '+')}.json"
    for m in a.methods.split(","):
        t0 = time.time()
        if m == "ngram":
            acc, ll = ngram(X, y, Xt, yt, V)
            res = dict(test_acc=acc, test_nll=ll)
        elif m == "mlp":
            import torch
            import torch.nn as nn
            from cobweb_forest import baselines as B
            torch.set_num_threads(4)

            class LM(nn.Module):
                def __init__(self, dim=64, hid=256):
                    super().__init__()
                    self.emb = nn.Embedding(V, dim)
                    self.net = nn.Sequential(nn.Linear(CTX * dim, hid), nn.SiLU(), nn.Linear(hid, V))

                def forward(self, x):
                    return self.net(self.emb(x.long()).flatten(1))
            model = B.train(LM(), X, y, Xv, yv, task="cls", seed=a.seed, epochs=100, patience=3, batch=512)
            logits = torch.as_tensor(B.predict(model, Xt))
            res = dict(test_acc=float((logits.argmax(1).numpy() == yt).mean()),
                       test_nll=float(torch.nn.functional.cross_entropy(logits, torch.as_tensor(yt)).item()),
                       params=B.n_params(model))
        else:
            best = None
            depths = (None,) if m == "single" else (2, 3, 4)
            for depth in depths:
                for alpha in (0.01, 0.1):
                    for weighting in ("prior", "path"):
                        F = Forest(design(m, depth, 300, alpha, weighting), "out", n_classes=V)
                        F.fit(None, None, C=y, X_disc=X, protocol="online")
                        acc, ll = evaluate(F.predict_proba(None, Xv), yv, V)
                        if best is None or ll < best["val_nll"]:
                            best = dict(hp=dict(depth=depth, alpha=alpha, weighting=weighting),
                                        val_nll=ll, val_acc=acc, F=F)
            F = best.pop("F")
            acc, ll = evaluate(F.predict_proba(None, Xt), yt, V)
            res = dict(best, test_acc=acc, test_nll=ll, concepts=int(sum(F.num_concepts().values())))
        res["seconds"] = time.time() - t0
        out["results"][m] = res
        print(a.N, m, {k: (round(v["test_acc"], 4), round(v["test_nll"], 3)) for k, v in out["results"].items()}, flush=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
