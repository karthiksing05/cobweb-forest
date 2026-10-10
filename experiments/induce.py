"""Task induction by verification (theory Procedure "task induction"), end to end.

  1. one tree per raw coordinate;
  2. estimate the synergy deficit of every pair (validation MSE of the funnel pair forest minus
     that of one tree over the pair, / Var Y) and merge the pairs above a threshold tau
     (union-find);
  3. compose the groups bottom-up with two-input trees (a balanced graph over the groups) and
     choose the operator (act / hybrid) -- and tau -- by the validation error at the output.
The candidate set also contains the single tree over all inputs and the generic balanced
forest over the coordinates, so the selection can always fall back on them.

    python experiments/induce.py --target feynman:I.12.11 --N 4000 --seed 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cobweb_forest import Forest, code_hashes, designs as D  # noqa: E402

KW = dict(max_nodes=300, leaves_only=True)
TAUS = (0.02, 0.05, 0.1, 0.2)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a).ravel() - np.asarray(b).ravel()) ** 2)))


def make_target(name, seed):
    rng = np.random.default_rng(3000 + seed)
    if name.startswith("feynman:"):
        import feynman as FY
        ranges, phys = FY.EQS[name.split(":", 1)[1]]
        n = len(ranges)

        def f(X):
            return phys(*[lo + (X[:, i] + 1) / 2 * (hi - lo) for i, (lo, hi) in enumerate(ranges)])

        def draw(m):
            X = rng.uniform(-1, 1, (m, n)).astype(np.float32)
            return X, f(X)[:, None].astype(np.float32)
        return n, draw
    if name == "nested4":
        def draw(m):
            X = rng.uniform(-1, 1, (m, 4)).astype(np.float32)
            y = np.exp((np.sin(np.pi * (X[:, 0] ** 2 + X[:, 1] ** 2)) + np.sin(np.pi * (X[:, 2] ** 2 + X[:, 3] ** 2))) / 2)
            return X, y[:, None].astype(np.float32)
        return 4, draw
    if name == "housing":
        d = np.load("data/housing.npz")
        Z = ((d["X"] - d["X"].mean(0)) / d["X"].std(0)).astype(np.float32)
        perm = rng.permutation(len(Z))
        state = {"i": 0}

        def draw(m):
            idx = perm[state["i"]:state["i"] + m]
            state["i"] += m
            return Z[idx], d["y"][idx, None].astype(np.float32)
        return 8, draw
    raise ValueError(name)


def groups_from(Dm, tau):
    n = len(Dm)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            if Dm[i, j] > tau:
                parent[find(i)] = find(j)
    out = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return sorted(out.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--N", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    n, draw = make_target(a.target, a.seed)
    tr, va, te = draw(a.N), draw(max(1000, a.N // 4)), draw(2000)
    var = float(np.var(va[1]))
    t0 = time.time()
    # step 2: synergy deficits
    Dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            Xtr, Xva = tr[0][:, [i, j]], va[0][:, [i, j]]
            act = Forest(D.ka((0, 1), "act", a.N, kappa=0.25, **KW), "out").fit(Xtr, tr[1], protocol="online")
            mrg = Forest(D.single(2, a.N, kappa=0.25, **KW), "out").fit(Xtr, tr[1], protocol="online")
            Dm[i, j] = Dm[j, i] = (rmse(act.predict(Xva), va[1]) ** 2 - rmse(mrg.predict(Xva), va[1]) ** 2) / var
    # step 3: candidates
    cands = {"single": D.single(n, a.N, kappa=0.25, **KW)}
    if n > 1:
        cands["balanced-act"] = D.ka(D.balanced(range(n)), "act", a.N, kappa=0.25, **KW)
    induced = {}
    for tau in TAUS:
        G = groups_from(Dm, tau)
        induced[tau] = G
        if len(G) == 1:
            continue
        graph = D.balanced([list(g) for g in G])
        for mode in ("act", "hybrid"):
            key = f"induced-{mode}-tau{tau}"
            cands[key] = D.ka(graph, mode, a.N, kappa=0.25, label_depth=3, **KW)
    res = {}
    for name, specs in cands.items():
        F = Forest(specs, "out").fit(tr[0], tr[1], protocol="online")
        res[name] = dict(val=rmse(F.predict(va[0]), va[1]), test=rmse(F.predict(te[0]), te[1]))
    pick = min(res, key=lambda k: res[k]["val"])
    sd = float(np.std(te[1]))
    print(a.target, a.N, a.seed, "groups", {t: g for t, g in induced.items()}, "pick", pick,
          {k: round(v["test"] / sd, 3) for k, v in res.items()}, flush=True)
    os.makedirs("results/induce", exist_ok=True)
    with open(f"results/induce/{a.target.replace(':', '_')}_N{a.N}_s{a.seed}.json", "w", encoding="utf-8") as fh:
        json.dump(dict(args=vars(a), code=code_hashes(), seconds=time.time() - t0, deficits=Dm.tolist(),
                       groups={str(t): g for t, g in induced.items()}, candidates=res, pick=pick, sd_y=sd),
                  fh, indent=1)


if __name__ == "__main__":
    main()
