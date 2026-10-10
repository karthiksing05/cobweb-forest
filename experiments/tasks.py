"""Cognitive-task experiments (theory Secs. 5-6).

  locality   Thm locality: in the sign worlds P_a (Y = sign(x1 x2)) and P_b (Y = sign(x1 x3),
             x3 unobserved) a tree that perceives x2 sees the same data; only the composing
             tree's verification separates the worlds.
  select     Thm selection: candidates are whole task assignments (funnel / label at several
             depths / hybrid / one merged tree); the validation error of Y at the output picks
             one.  Records the pick, and every candidate's validation and test error.
  synergy    Def. synergy deficit: for every pair of coordinates, validation MSE of the funnel
             pair forest minus that of one tree over both coordinates, divided by Var(Y).
  multiview  Prop multiview: Gaussian two-view latent model; cross-view trees never see Y.

    python experiments/tasks.py <which> --N 2000 --seed 0 [--target xy]
"""
import argparse
import json
import os
import time

import numpy as np

from cobweb_forest import Forest, TreeSpec, code_hashes, designs as D

KW = dict(max_nodes=300, leaves_only=True)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a).ravel() - np.asarray(b).ravel()) ** 2)))


# ------------------------------------------------------------------------------- targets
def target(name, rng, m, noisy=True):
    """Returns X (m, n) and Y (m, 1); the test sets are drawn with noisy=False."""
    if name in ("xy", "sign", "expsin"):
        X = rng.uniform(-1, 1, (m, 2)).astype(np.float32)
        y = {"xy": X[:, 0] * X[:, 1], "sign": np.sign(X[:, 0] * X[:, 1]),
             "expsin": np.exp(np.sin(np.pi * X[:, 0]) + X[:, 1] ** 2)}[name]
    elif name == "nested4":
        X = rng.uniform(-1, 1, (m, 4)).astype(np.float32)
        y = np.exp((np.sin(np.pi * (X[:, 0] ** 2 + X[:, 1] ** 2)) + np.sin(np.pi * (X[:, 2] ** 2 + X[:, 3] ** 2))) / 2)
    elif name == "housing":
        d = np.load("data/housing.npz")
        idx = rng.choice(len(d["X"]), m, replace=False)
        Z = (d["X"] - d["X"].mean(0)) / d["X"].std(0)
        return Z[idx].astype(np.float32), d["y"][idx, None].astype(np.float32)
    else:
        raise ValueError(name)
    return X, y[:, None].astype(np.float32)


# ------------------------------------------------------------------------------- candidates
def candidates(n, N, groups=None):
    """Whole task assignments over coordinates 0..n-1 (two-level graphs)."""
    leaves = tuple(range(n)) if groups is None else tuple(groups)
    c = {"funnel": D.ka(leaves, "act", N, kappa=0.25, **KW)}
    for depth in (2, 3, 4):
        c[f"label-d{depth}"] = D.ka(leaves, "label", N, kappa=0.25, label_depth=depth, **KW)
    c["hybrid-d3"] = D.ka(leaves, "hybrid", N, kappa=0.25, label_depth=3, **KW)
    c["merge"] = D.single(n, N, kappa=0.25, **KW)
    return c


def fit_eval(specs, train, val, test):
    F = Forest(specs, "out").fit(train[0], train[1], protocol="online")
    return rmse(F.predict(val[0]), val[1]), rmse(F.predict(test[0]), test[1]), F


# ------------------------------------------------------------------------------- experiments
def run_locality(a):
    rng = np.random.default_rng(a.seed)
    out = {}
    for world in ("a", "b"):
        def draw(m):
            X3 = rng.uniform(-1, 1, (m, 3)).astype(np.float32)
            other = X3[:, 1] if world == "a" else X3[:, 2]
            return X3[:, :2].copy(), np.sign(X3[:, 0] * other)[:, None].astype(np.float32)
        tr, va, te = draw(a.N), draw(max(500, a.N // 4)), draw(2000)
        res = {}
        # local: a univariate funnel tree on x2 alone -- its own validation score and messages
        loc = Forest([TreeSpec("out", raw=[1], task="y", prior_var=D.bandwidth(a.N, 1, 0.25), **KW)], "out")
        loc.fit(tr[0], tr[1], protocol="online")
        P = loc.predict(va[0])[:, 0]
        res["local_x2_val_rmse"] = rmse(P, va[1])
        res["local_x2_const_rmse"] = rmse(np.full(len(va[1]), tr[1].mean()), va[1])
        res["local_x2_act_sd"] = float(np.std(P))
        res["local_x2_concepts"] = loc.trees["out"].tree.num_concepts()
        # composing tree: whole assignments
        for name, specs in candidates(2, a.N).items():
            v, t, _ = fit_eval(specs, tr, va, te)
            res[f"compose:{name}"] = dict(val=v, test=t)
        out[world] = res
        print(world, {k: (v if not isinstance(v, dict) else round(v["test"], 3)) for k, v in res.items()}, flush=True)
    return out


def run_select(a):
    rng = np.random.default_rng(a.seed)
    n = {"xy": 2, "sign": 2, "expsin": 2, "nested4": 4}[a.target]
    tr = target(a.target, rng, a.N)
    va = target(a.target, rng, max(500, a.N // 4))
    te = target(a.target, rng, 2000, noisy=False)
    cands = candidates(n, a.N)
    if a.target == "nested4":
        cands["funnel-pairs"] = D.ka(([0, 1], [2, 3]), "act", a.N, kappa=0.25, **KW)
        cands["hybrid-pairs"] = D.ka(([0, 1], [2, 3]), "hybrid", a.N, kappa=0.25, label_depth=3, **KW)
    res = {}
    for name, specs in cands.items():
        t0 = time.time()
        v, t, _ = fit_eval(specs, tr, va, te)
        res[name] = dict(val=v, test=t, seconds=time.time() - t0)
    pick = min(res, key=lambda k: res[k]["val"])
    best_test = min(res, key=lambda k: res[k]["test"])
    print(a.target, a.N, a.seed, "pick", pick, "best-test", best_test,
          {k: round(v["test"], 3) for k, v in res.items()}, flush=True)
    return dict(target=a.target, candidates=res, pick=pick, best_test=best_test,
                sd_y=float(te[1].std()))


def run_synergy(a):
    rng = np.random.default_rng(a.seed)
    n = {"nested4": 4, "housing": 8, "expsin": 2, "xy": 2}[a.target]
    tr = target(a.target, rng, a.N)
    va = target(a.target, rng, max(1000, a.N // 4))
    var = float(np.var(va[1]))
    M = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            Xtr, Xva = tr[0][:, [i, j]], va[0][:, [i, j]]
            act = Forest(D.ka((0, 1), "act", a.N, kappa=0.25, **KW), "out").fit(Xtr, tr[1], protocol="online")
            mrg = Forest(D.single(2, a.N, kappa=0.25, **KW), "out").fit(Xtr, tr[1], protocol="online")
            d = (rmse(act.predict(Xva), va[1]) ** 2 - rmse(mrg.predict(Xva), va[1]) ** 2) / var
            M[i, j] = M[j, i] = d
            print(a.target, i, j, round(d, 4), flush=True)
    return dict(target=a.target, deficit_over_var=M.tolist())


def run_multiview(a):
    rng = np.random.default_rng(a.seed)
    lam1, lam2 = np.array([1.0, 0.8]), np.array([0.9, -0.7])
    s1, s2 = np.array([0.5, 0.6]), np.array([0.6, 0.5])

    def draw(m, kind):
        z = rng.normal(size=m)
        E1, E2 = rng.normal(size=(m, 2)) * s1, rng.normal(size=(m, 2)) * s2
        X = np.concatenate([np.outer(z, lam1) + E1, np.outer(z, lam2) + E2], 1).astype(np.float32)
        y = np.sin(2 * z) if kind == "shared" else E1[:, 0]
        return X, (y + 0.05 * rng.normal(size=m))[:, None].astype(np.float32), z

    def oracle(X, kind):
        # posterior of z given all four coordinates (prior N(0,1))
        L = np.concatenate([lam1, lam2]); S = np.concatenate([s1, s2]) ** 2
        P = 1 + np.sum(L * L / S)
        mu = (X @ (L / S)) / P
        if kind == "shared":
            return np.sin(2 * mu) * np.exp(-2 / P)
        return X[:, 0] - lam1[0] * mu          # E[E1_0 | X]

    out = {}
    for kind in ("shared", "view-specific"):
        tr, va = draw(a.N, kind), draw(max(500, a.N // 4), kind)
        te = draw(2000, kind)
        f_te = oracle(te[0], kind)
        designs = {
            "cross-view": [TreeSpec("t1", raw=[0, 1], task="raw:2", emits=["act"], prior_var=D.bandwidth(a.N, 2, 0.5), **KW),
                           TreeSpec("t2", raw=[2, 3], task="raw:0", emits=["act"], prior_var=D.bandwidth(a.N, 2, 0.5), **KW),
                           TreeSpec("out", act_from=["t1", "t2"], task="y", prior_var=D.bandwidth(a.N, 2, 0.5), **KW)],
            "funnel": D.ka(([0, 1], [2, 3]), "act", a.N, kappa=0.5, **KW),
            "label": D.ka(([0, 1], [2, 3]), "label", a.N, kappa=0.5, label_depth=3, **KW),
            "single": D.single(4, a.N, kappa=0.5, **KW),
        }
        res = {"oracle_vs_y": rmse(f_te, te[1])}
        for name, specs in designs.items():
            F = Forest(specs, "out").fit(tr[0], tr[1], protocol="online")
            P = F.predict(te[0])[:, 0]
            res[name] = dict(rmse_vs_f=rmse(P, f_te), rmse_vs_y=rmse(P, te[1]), val=rmse(F.predict(va[0]), va[1]))
        res["sd_f"] = float(np.std(f_te))
        out[kind] = res
        print(kind, a.N, {k: (round(v["rmse_vs_f"], 3) if isinstance(v, dict) else round(v, 3)) for k, v in res.items()}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["locality", "select", "synergy", "multiview"])
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="xy")
    a = ap.parse_args()
    t0 = time.time()
    res = {"locality": run_locality, "select": run_select, "synergy": run_synergy, "multiview": run_multiview}[a.which](a)
    os.makedirs("results/tasks", exist_ok=True)
    tag = f"{a.which}_{a.target}_N{a.N}_s{a.seed}" if a.which in ("select", "synergy") else f"{a.which}_N{a.N}_s{a.seed}"
    with open(f"results/tasks/{tag}.json", "w", encoding="utf-8") as fh:
        json.dump(dict(args=vars(a), code=code_hashes(), seconds=time.time() - t0, result=res), fh, indent=1, default=float)


if __name__ == "__main__":
    main()
