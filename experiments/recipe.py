"""The recipe for building a forest (proc:recipe), run under the theorems' protocol on targets
whose right structure the theory determines exactly.

Target: Y = sum_p c_p x_{a_p} x_{b_p} + sum_u h_u(x_u) + 0.1 xi, inputs independent uniform on
[-1, 1], plus irrelevant inputs.  For this additive target:
  * a product pair (a_p, b_p) has synergy deficit c_p^2 Var(x_a x_b) > 0: each one-input activation
    is the constant 0 (prop:tasks(b)), so the pair must be merged;
  * every other pair has deficit 0 (the target is additive and each univariate activation is
    injective), so it must not be merged;
  * an irrelevant input adds nothing alone or in any pair, so it must be pruned.
The recipe sees only data.  Steps:
  1. relevance: input i is kept if it lowers the held-out error alone or in some pair by more
     than tau (pruning, step 6 of proc:recipe);
  2. synergy: merge kept pairs whose deficit (activations of two one-input trees vs one tree over
     the pair) exceeds tau (union-find);
  3. compose the groups with two-input trees passing activations; choose among this forest, one
     tree over the kept inputs and one tree over all inputs by validation error.
All trees: count-weighted leaf mixture, full expansion; forests: split protocol.

    python experiments/recipe.py --n 8 --N 4000 --seed 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cobweb_forest import Forest, code_hashes, designs as D  # noqa: E402

BIG = 10 ** 7
KW = dict(max_nodes=BIG, leaves_only=True, weighting="prior")
TAU = 0.02
KAPPA = 0.7


def mse(F, X, y):
    return float(np.mean((F.predict(X)[:, 0] - y[:, 0]) ** 2))


def instance(n, seed):
    rng = np.random.default_rng(5000 + seed)
    P, U = {8: (2, 2), 12: (3, 3)}[n]
    perm = rng.permutation(n)
    pairs = [tuple(sorted(perm[2 * p:2 * p + 2].tolist())) for p in range(P)]
    unis = perm[2 * P:2 * P + U].tolist()
    irr = sorted(perm[2 * P + U:].tolist())
    c = rng.uniform(1.0, 2.0, P)
    kinds = rng.integers(0, 2, U)

    def f(X):
        out = sum(c[p] * X[:, a] * X[:, b] for p, (a, b) in enumerate(pairs))
        for k, u in zip(kinds, unis):
            out = out + (X[:, u] if k == 0 else np.sin(np.pi * X[:, u] / 2))
        return out

    def draw(m):
        X = rng.uniform(-1, 1, (m, n))
        return X.astype(np.float32), (f(X) + 0.1 * rng.normal(size=m))[:, None].astype(np.float32)
    truth = dict(pairs=pairs, univariate=sorted(unis), irrelevant=irr)
    return truth, draw, f


def union_groups(n, keep, merges):
    parent = {i: i for i in keep}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j in merges:
        parent[find(i)] = find(j)
    out = {}
    for i in keep:
        out.setdefault(find(i), []).append(i)
    return sorted(sorted(g) for g in out.values())


def balanced(groups):
    nodes = [g if len(g) > 1 else g[0] for g in groups]
    if len(nodes) == 1:
        return nodes[0]
    while len(nodes) > 1:
        nodes = [tuple(nodes[i:i + 2]) if i + 1 < len(nodes) else nodes[i] for i in range(0, len(nodes), 2)]
    return nodes[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    n, N = a.n, a.N
    truth, draw, f = instance(n, a.seed)
    tr, va = draw(N), draw(max(1000, N // 2))
    Xt = np.random.default_rng(9).uniform(-1, 1, (2000, n)).astype(np.float32)
    ft = f(Xt)
    var = float(np.var(va[1]))
    t0 = time.time()
    base = float(np.mean((va[1][:, 0] - tr[1].mean()) ** 2))
    single = {}
    for i in range(n):
        F = Forest(D.single(1, N, kappa=KAPPA, **KW), "out").fit(tr[0][:, [i]], tr[1], protocol="online")
        single[i] = mse(F, va[0][:, [i]], va[1])
    pair, act = {}, {}
    for i in range(n):
        for j in range(i + 1, n):
            cols = [i, j]
            F = Forest(D.single(2, N, kappa=KAPPA, **KW), "out").fit(tr[0][:, cols], tr[1], protocol="online")
            pair[(i, j)] = mse(F, va[0][:, cols], va[1])
            A = Forest(D.ka((0, 1), "act", N, kappa=KAPPA, **KW), "out").fit(tr[0][:, cols], tr[1], protocol="split")
            act[(i, j)] = mse(A, va[0][:, cols], va[1])
    P = lambda i, j: pair[(min(i, j), max(i, j))]
    relevance = {i: max([(base - single[i]) / var] + [(single[j] - P(i, j)) / var for j in range(n) if j != i])
                 for i in range(n)}
    keep = [i for i in range(n) if relevance[i] > TAU]
    deficit = {k: (act[k] - pair[k]) / var for k in pair}
    merges = [k for k in pair if k[0] in keep and k[1] in keep and deficit[k] > TAU]
    groups = union_groups(n, keep, merges)
    true_groups = sorted([list(p) for p in truth["pairs"]] + [[u] for u in truth["univariate"]])
    # step 3: candidates, verified at the output
    cands = {"single-all": (D.single(n, N, kappa=KAPPA, **KW), list(range(n)))}
    if keep:
        cands["single-kept"] = (D.single(len(keep), N, kappa=KAPPA, **KW), keep)
        if len(groups) > 1:
            loc = [[keep.index(i) for i in g] for g in groups]
            cands["recipe-forest"] = (D.ka(balanced(loc), "act", N, kappa=KAPPA, **KW), keep)
    res = {}
    for name, (specs, cols) in cands.items():
        F = Forest(specs, "out").fit(tr[0][:, cols], tr[1], protocol="split")
        res[name] = dict(val=mse(F, va[0][:, cols], va[1]),
                         test_rmse=float(np.sqrt(np.mean((F.predict(Xt[:, cols])[:, 0] - ft) ** 2))) / float(np.std(ft)))
    pick = min(res, key=lambda k: res[k]["val"])
    out = dict(n=n, N=N, seed=a.seed, truth=truth, keep=keep, groups=groups,
               pruned_correct=sorted(set(range(n)) - set(keep)) == truth["irrelevant"],
               groups_correct=groups == true_groups,
               deficits={f"{i},{j}": d for (i, j), d in deficit.items()},
               relevance=relevance, candidates=res, pick=pick, seconds=time.time() - t0, code=code_hashes())
    os.makedirs("results/recipe", exist_ok=True)
    with open(f"results/recipe/n{n}_N{N}_s{a.seed}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(n, N, a.seed, "pruned ok", out["pruned_correct"], "groups ok", out["groups_correct"], groups, "truth", true_groups,
          "pick", pick, {k: round(v["test_rmse"], 3) for k, v in res.items()}, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
