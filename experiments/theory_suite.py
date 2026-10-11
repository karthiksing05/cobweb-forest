"""Theory <-> measurement suite: every experiment measures a quantity a theorem predicts.

All trees are real hybrid Cobweb trees.  Constructed trees (cobweb_forest.construct) realise the
existence proofs exactly; learned trees use only Cobweb's ifit.  Unless stated otherwise every tree
predicts with the count-weighted leaves-only mixture under full expansion (the setting of the
consistency theorems), and forests are trained with the split protocol.

  r1  one constructed tree, grid of G leaves, d = 1, 2, 3       predicted: sup error ~ G^(-1/d)
  r2  compositional target of two-input maps, n = 2, 4, 8:
        constructed forest of 2-input trees, G leaves each        predicted: sup error ~ G^(-1/2), any n
        constructed single tree, G leaves                          predicted: sup error ~ G^(-1/n)
  l1  one learned tree, noisy target, d = 1, 2, 3               predicted: interior RMSE ~ N^(-2/(d+4))
  l2  learned funnel forest vs one learned tree on Phi(sum of 2-D blocks), n = 2, 4, 8
                                                                 predicted: N^(-1/3) vs N^(-2/(n+4))
  l3  task deficits: funnel forest on x1*x2 (predicted floor 1), label forest and one tree (-> 0);
      cross-view forest on a view-specific Gaussian target (predicted floor: closed form)

    python experiments/theory_suite.py r1
    python experiments/theory_suite.py l1 --d 2 --N 4000 --seed 0
"""
import argparse
import json
import math
import os
import time

import numpy as np

from cobweb_forest import Forest, TreeSpec, code_hashes
from cobweb_forest import designs as D
from cobweb_forest.construct import build

BIG = 10 ** 7
KW = dict(max_nodes=BIG, leaves_only=True, weighting="prior")   # count-weighted leaves, full expansion
KAPPAS = (0.35, 0.7, 1.4)
OUT = "results/theory"


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    obj["code"] = code_hashes()
    with open(f"{OUT}/{name}.json", "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=float)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float).ravel() - np.asarray(b, float).ravel()) ** 2)))


def slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


# ------------------------------------------------------------------------- constructed trees
def grid_tree(fn, lo, hi, side, beta=0.5):
    """Depth-one tree with one leaf per cell centre of a side^d grid on the box [lo, hi]
    (per-coordinate arrays), holding (centre, fn(centre)); variance floor (beta * spacing)^2."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    d = len(lo)
    axes = [lo[k] + (np.arange(side) + 0.5) * (hi[k] - lo[k]) / side for k in range(d)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, d)
    leaves = [({}, np.append(p, fn(p))) for p in pts]
    h = beta * float(np.max((hi - lo) / side))
    return build(leaves, d + 1, prior_var=h * h), len(pts)


def tree_predict(t, x):
    q = np.append(np.asarray(x, np.float32), np.float32(np.nan)).astype(np.float32)
    return float(t.predict_mean({}, q, BIG, False, "prior", True)[len(x)])


def run_r1(a):
    rng = np.random.default_rng(a.seed)
    targets = {1: lambda p: math.sin(2 * math.pi * p[0]),
               2: lambda p: math.sin(2 * math.pi * p[0]) * math.cos(math.pi * p[1]),
               3: lambda p: math.sin(2 * math.pi * p[0]) * math.cos(math.pi * p[1]) + p[2] ** 2}
    sides = {1: (8, 16, 32, 64, 128, 256, 512), 2: (4, 8, 16, 32, 64), 3: (3, 5, 8, 12, 16)}
    out = {}
    for d, g in targets.items():
        Q = rng.uniform(0, 1, (400, d))
        rows = []
        for s in sides[d]:
            t, G = grid_tree(g, np.zeros(d), np.ones(d), s)
            err = max(abs(tree_predict(t, x) - g(x)) for x in Q)
            rows.append(dict(G=G, sup_err=err))
            print("r1", d, G, round(err, 5), flush=True)
        out[d] = dict(rows=rows, predicted_slope=-1.0 / d,
                      measured_slope=slope([r["G"] for r in rows], [r["sup_err"] for r in rows]))
    save("r1", out)


# compositional target: a balanced binary tree of the two-input map gnode.  "smooth" uses a C^inf
# map; "lip" puts kinks inside every node's input box, where the Lipschitz rate is tight.
KIND = {"v": "smooth"}


def gnode(a, b):
    if KIND["v"] == "lip":
        return 0.5 * (a + b) + 0.25 * abs(math.sin(2 * math.pi * a * b))
    return 0.5 * (a + b) + 0.25 * math.sin(2 * math.pi * a * b)


def comp_levels(x):
    """Values at every level of the binary composition (level 0 = x)."""
    levels = [list(x)]
    while len(levels[-1]) > 1:
        v = levels[-1]
        levels.append([gnode(v[2 * i], v[2 * i + 1]) for i in range(len(v) // 2)])
    return levels


def run_r2(a):
    KIND["v"] = a.kind
    rng = np.random.default_rng(a.seed)
    out = {}
    for n in (2, 4, 8):
        Q = rng.uniform(0, 1, (300, n))
        f = lambda x: comp_levels(x)[-1][0]
        # input box of every node, from the exact composition, with a margin for the
        # deviation of approximate children
        S = rng.uniform(0, 1, (20000, n))
        L = [comp_levels(x) for x in S]
        depth = len(L[0]) - 1
        boxes = {}
        for lev in range(depth):
            for i in range(len(L[0][lev]) // 2):
                va = np.array([l[lev][2 * i] for l in L]); vb = np.array([l[lev][2 * i + 1] for l in L])
                lo = np.array([va.min(), vb.min()]); hi = np.array([va.max(), vb.max()])
                m = 0.1 * (hi - lo) + 1e-3
                boxes[(lev, i)] = (lo - m, hi + m)
        forest_rows = []
        for s in (4, 8, 16, 32, 64):
            trees = {k: grid_tree(lambda p: gnode(p[0], p[1]), lo, hi, s)[0] for k, (lo, hi) in boxes.items()}
            err = 0.0
            lev_err = [0.0] * depth                      # e_l: max error of the values at level l+1
            for x in Q:
                v = list(x)
                ex = comp_levels(x)
                for lev in range(depth):
                    v = [tree_predict(trees[(lev, i)], [v[2 * i], v[2 * i + 1]]) for i in range(len(v) // 2)]
                    lev_err[lev] = max(lev_err[lev], max(abs(a - b) for a, b in zip(v, ex[lev + 1])))
                err = max(err, abs(v[0] - f(x)))
            # a-posteriori hypothesis of thm:compose: every input stays in its max-norm margin
            margins = [min(float(np.min(boxes[(lev, i)][1] - boxes[(lev, i)][0]) * 0.1 / 1.2)
                           for i in range(len(L[0][lev]) // 2)) for lev in range(depth)]
            inside = all(lev_err[l - 1] <= margins[l] for l in range(1, depth))
            forest_rows.append(dict(G_per_tree=s * s, trees=len(trees), G_total=s * s * len(trees), sup_err=err,
                                    level_err=lev_err, margins=margins, inputs_inside_margin=bool(inside)))
            print("r2 forest", n, s * s, round(err, 5), flush=True)
        single_sides = {2: (4, 8, 16, 32, 64), 4: (3, 4, 6, 8, 10, 12), 8: (2, 3, 4, 5)}[n]
        single_rows = []
        for s in single_sides:
            t, G = grid_tree(f, np.zeros(n), np.ones(n), s)
            err = max(abs(tree_predict(t, x) - f(x)) for x in Q)
            single_rows.append(dict(G_total=G, sup_err=err))
            print("r2 single", n, G, round(err, 5), flush=True)
        out[n] = dict(forest=forest_rows, single=single_rows,
                      predicted_slope_forest=-0.5, predicted_slope_single=-1.0 / n,
                      measured_slope_forest=slope([r["G_per_tree"] for r in forest_rows], [r["sup_err"] for r in forest_rows]),
                      measured_slope_single=slope([r["G_total"] for r in single_rows], [r["sup_err"] for r in single_rows]))
    save("r2" if a.kind == "smooth" else f"r2_{a.kind}", out)


# ------------------------------------------------------------------------- learned trees
def fit_select(make, train, val, test_X, protocol="split"):
    """Fit one design per kappa, keep the one with the lowest validation RMSE."""
    best = None
    for k in KAPPAS:
        F = Forest(make(k), "out").fit(train[0], train[1], protocol=protocol)
        v = rmse(F.predict(val[0]), val[1])
        if best is None or v < best[0]:
            best = (v, k, F)
    v, k, F = best
    return dict(kappa=k, val=v), F.predict(test_X)[:, 0]


def run_l1(a):
    rng = np.random.default_rng(a.seed)
    d, N, sig = a.d, a.N, 0.3
    g = lambda X: np.sin(2 * np.pi * X[:, 0]) * (np.cos(np.pi * X[:, 1]) if X.shape[1] > 1 else 1) \
        + (X[:, 2] ** 2 if X.shape[1] > 2 else 0)
    draw = lambda m: (lambda X: (X.astype(np.float32), (g(X) + sig * rng.normal(size=m))[:, None].astype(np.float32)))(rng.uniform(0, 1, (m, d)))
    tr, va = draw(N), draw(1000)
    Xt = rng.uniform(0.25, 0.75, (2000, d)).astype(np.float32)          # interior queries
    info, P = fit_select(lambda k: D.single(d, N, kappa=k, **KW), tr, va, Xt, protocol="online")
    out = dict(d=d, N=N, seed=a.seed, sigma=sig, rmse_interior=rmse(P, g(Xt)), predicted_rate=-2 / (d + 4), **info)
    print("l1", out["d"], N, round(out["rmse_interior"], 4), info, flush=True)
    save(f"l1_d{d}_N{N}_s{a.seed}", out)


def blocks_target(X):
    n = X.shape[1]
    s = np.mean([np.sin(np.pi * (X[:, 2 * j] ** 2 + X[:, 2 * j + 1] ** 2)) for j in range(n // 2)], 0)
    return np.exp(s)


def balanced_pairs(n):
    nodes = [[2 * j, 2 * j + 1] for j in range(n // 2)]
    while len(nodes) > 1:
        nodes = [tuple(nodes[i:i + 2]) for i in range(0, len(nodes), 2)]
    return nodes[0]


def run_l2(a):
    rng = np.random.default_rng(a.seed)
    n, N, sig = a.n, a.N, 0.1
    draw = lambda m: (lambda X: (X.astype(np.float32), (blocks_target(X) + sig * rng.normal(size=m))[:, None].astype(np.float32)))(rng.uniform(-1, 1, (m, n)))
    tr, va = draw(N), draw(1000)
    Xt = rng.uniform(-1, 1, (2000, n)).astype(np.float32)
    Xi = rng.uniform(-0.75, 0.75, (2000, n)).astype(np.float32)
    ft, fi = blocks_target(Xt), blocks_target(Xi)
    out = dict(n=n, N=N, seed=a.seed, sd_f=float(np.std(ft)), results={})
    designs = {"single": lambda k: D.single(n, N, kappa=k, **KW)}
    if n > 2:
        designs["forest"] = lambda k: D.ka(balanced_pairs(n), "act", N, kappa=k, **KW)
    for name, make in designs.items():
        t0 = time.time()
        info, P = fit_select(make, tr, va, np.concatenate([Xt, Xi]))
        out["results"][name] = dict(rmse=rmse(P[:2000], ft), rmse_interior=rmse(P[2000:], fi),
                                    seconds=time.time() - t0, **info)
        print("l2", n, N, name, {k: round(v, 4) for k, v in out["results"][name].items()}, flush=True)
    save(f"l2_n{n}_N{N}_s{a.seed}", out)


# Gaussian two-view model (as in experiments/tasks.py): exact deficit of the cross-view forest
LAM1, LAM2 = np.array([1.0, 0.8]), np.array([0.9, -0.7])
S1, S2 = np.array([0.5, 0.6]), np.array([0.6, 0.5])


def multiview_deficit():
    """Every variable is linear in w = (z, e11, e12, e21, e22) ~ N(0, I).  f = E[E1_0 | X] and the
    cross-view ideal output F = E[E1_0 | mu_1, mu_2] are linear; return Delta / Var f exactly."""
    A = np.zeros((4, 5))                                    # X = A w
    A[0:2, 0] = LAM1; A[2:4, 0] = LAM2
    A[0, 1], A[1, 2], A[2, 3], A[3, 4] = S1[0], S1[1], S2[0], S2[1]
    y = np.zeros(5); y[1] = S1[0]                           # Y = E1_0 (noise-free part)

    def proj(B, v):
        """coefficients (in w) of E[v.w | B w]"""
        C = B @ B.T
        return (B.T @ np.linalg.solve(C, B @ v))
    f = proj(A, y)
    mu1 = proj(A[0:2], np.eye(5)[0]); mu2 = proj(A[2:4], np.eye(5)[0])   # E[z | view v], as rows in w
    M = np.stack([mu1, mu2])
    F = proj(M, y)
    return float(np.sum((f - F) ** 2) / np.sum(f ** 2))


def run_l3(a):
    rng = np.random.default_rng(a.seed)
    N = a.N
    out = dict(N=N, seed=a.seed)
    # (a) x1 * x2
    draw = lambda m: (lambda X: (X.astype(np.float32), (X[:, 0] * X[:, 1] + 0.05 * rng.normal(size=m))[:, None].astype(np.float32)))(rng.uniform(-1, 1, (m, 2)))
    tr, va = draw(N), draw(1000)
    Xt = rng.uniform(-1, 1, (2000, 2)).astype(np.float32); ft = Xt[:, 0] * Xt[:, 1]
    res = {}
    for name, make in {"funnel": lambda k: D.ka((0, 1), "act", N, kappa=k, **KW),
                       "single": lambda k: D.single(2, N, kappa=k, **KW)}.items():
        info, P = fit_select(make, tr, va, Xt)
        res[name] = dict(ratio=rmse(P, ft) / float(np.std(ft)), **info)
    best = None
    # Thm labelcons needs alpha_N * M_N -> 0: label smoothing alpha = 1e-4 (the default 0.1 violates it)
    for depth in (1, 2, 3, 4):
        info, P = fit_select(lambda k: D.ka((0, 1), "label", N, kappa=k, label_depth=depth, alpha=1e-4, **KW), tr, va, Xt)
        if best is None or info["val"] < best[0]["val"]:
            best = (dict(depth=depth, **info), P)
    res["label"] = dict(ratio=rmse(best[1], ft) / float(np.std(ft)), **best[0])
    out["xy"] = dict(predicted={"funnel": 1.0, "label": 0.0, "single": 0.0}, measured=res)
    # (b) cross-view forest on a view-specific target
    def mv(m):
        z = rng.normal(size=m)
        E1, E2 = rng.normal(size=(m, 2)) * S1, rng.normal(size=(m, 2)) * S2
        X = np.concatenate([np.outer(z, LAM1) + E1, np.outer(z, LAM2) + E2], 1).astype(np.float32)
        return X, (E1[:, 0] + 0.05 * rng.normal(size=m))[:, None].astype(np.float32)
    tr, va, te = mv(N), mv(1000), mv(2000)
    L = np.concatenate([LAM1, LAM2]); Sv = np.concatenate([S1, S2]) ** 2
    P_ = 1 + np.sum(L * L / Sv); mu = (te[0] @ (L / Sv)) / P_
    f = te[0][:, 0] - LAM1[0] * mu
    make = lambda k: [TreeSpec("t1", raw=[0, 1], task="raw:2", emits=["act"], prior_var=D.bandwidth(N, 2, k), **KW),
                      TreeSpec("t2", raw=[2, 3], task="raw:0", emits=["act"], prior_var=D.bandwidth(N, 2, k), **KW),
                      TreeSpec("out", act_from=["t1", "t2"], task="y", prior_var=D.bandwidth(N, 2, k), **KW)]
    info, P = fit_select(make, tr, va, te[0])
    out["multiview"] = dict(predicted_ratio=math.sqrt(multiview_deficit()),
                            measured_ratio=rmse(P, f) / float(np.std(f)), **info)
    print("l3", N, json.dumps({k: v for k, v in out.items() if k in ("xy", "multiview")}, default=float)[:400], flush=True)
    save(f"l3_N{N}_s{a.seed}", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["r1", "r2", "l1", "l2", "l3", "mvcheck"])
    ap.add_argument("--d", type=int, default=1)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", default="smooth", choices=["smooth", "lip"])
    a = ap.parse_args()
    if a.which == "mvcheck":
        print("predicted cross-view floor (RMSE/sd f):", math.sqrt(multiview_deficit()))
        return
    {"r1": run_r1, "r2": run_r2, "l1": run_l1, "l2": run_l2, "l3": run_l3}[a.which](a)


if __name__ == "__main__":
    main()
