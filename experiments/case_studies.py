"""Case studies: the constructive proofs of the paper, executed with real Cobweb trees.

  bernstein  The universality proof (thm:univ-forest), with Bernstein polynomials as the
             approximating algebra: one-input trees compute the basis b_{m,i}(x), b_{m,j}(y);
             two-input trees compute c_ij * a * b and the sums.  Predicted: the forest converges
             to B_m f as the trees refine (lem:stable), and B_m f to f at rate 1/m.
  feynman    Composition beats dimension (thm:compose) on physics formulas: every operation of
             the formula's expression graph becomes one Cobweb tree (fan-in <= 2); compared with
             one tree over all n inputs at equal total leaves.

    python experiments/case_studies.py bernstein --m 6
    python experiments/case_studies.py feynman --eq I.9.18
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(__file__))
from theory_suite import grid_tree, tree_predict, save as _save  # noqa: E402

OUT = "results/cases"


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    from cobweb_forest import code_hashes
    obj["code"] = code_hashes()
    with open(f"{OUT}/{name}.json", "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=float)


# --------------------------------------------------------------------------------- forests
class Node:
    """One vertex of a forest of constructed trees: fn maps its input values to its output."""

    def __init__(self, fn, kids):
        self.fn, self.kids, self.tree = fn, kids, None


def exact(nodes, order, x):
    v = {}
    for k in order:
        nd = nodes[k]
        v[k] = nd.fn(*[x[c[1]] if c[0] == "x" else v[c[1]] for c in nd.kids])
    return v


def build_forest(nodes, order, sample, side, margin=0.1, clamp=False):
    """Construct every vertex's tree on the box of its exact inputs (over `sample`) enlarged
    by `margin` of its width on each side; G = side^2 leaves per tree (side^2 points on a line
    for one-input trees)."""
    V = [exact(nodes, order, x) for x in sample]
    leaves = 0
    for k in order:
        nd = nodes[k]
        cols = [np.array([x[c[1]] for x in sample]) if c[0] == "x" else np.array([v[c[1]] for v in V])
                for c in nd.kids]
        lo = np.array([c.min() for c in cols]); hi = np.array([c.max() for c in cols])
        w = np.maximum(hi - lo, 1e-6)
        elo, ehi = lo, hi
        lo, hi = lo - margin * w, hi + margin * w
        s = side if len(cols) == 2 else side * side
        if clamp:   # Lipschitz extension of g_v to the enlarged box: g_v(clip(u)) (thm:compose)
            fn_ext = lambda p, fn=nd.fn, a=elo, b=ehi: fn(*np.clip(p, a, b))
        else:
            fn_ext = lambda p, fn=nd.fn: fn(*p)
        nd.tree, G = grid_tree(fn_ext, lo, hi, s)
        leaves += G
    return leaves


def run_forest(nodes, order, x):
    v = {}
    for k in order:
        nd = nodes[k]
        v[k] = tree_predict(nd.tree, [x[c[1]] if c[0] == "x" else v[c[1]] for c in nd.kids])
    return v[order[-1]]


# ------------------------------------------------------------------------------ Bernstein
def f_target(x, y):
    return math.exp(math.sin(math.pi * x) + y * y)


def bern(m, i, t):
    return math.comb(m, i) * t ** i * (1 - t) ** (m - i)


def bernstein_nodes(m):
    nodes, order = {}, []

    def add(key, fn, kids):
        nodes[key] = Node(fn, kids); order.append(key)
        return ("n", key)
    bx = [add(f"bx{i}", lambda t, i=i: bern(m, i, t), [("x", 0)]) for i in range(m + 1)]
    by = [add(f"by{j}", lambda t, j=j: bern(m, j, t), [("x", 1)]) for j in range(m + 1)]
    terms = []
    for i in range(m + 1):
        for j in range(m + 1):
            c = f_target(i / m, j / m)
            terms.append(add(f"p{i}_{j}", lambda a, b, c=c: c * a * b, [bx[i], by[j]]))
    k = 0
    while len(terms) > 1:                               # balanced binary sum
        nxt = []
        for a in range(0, len(terms) - 1, 2):
            nxt.append(add(f"s{k}", lambda u, v: u + v, [terms[a], terms[a + 1]])); k += 1
        if len(terms) % 2:
            nxt.append(terms[-1])
        terms = nxt
    if order[-1] != terms[0][1]:                        # make the root last
        order.remove(terms[0][1]); order.append(terms[0][1])
    return nodes, order


def run_bernstein(a):
    rng = np.random.default_rng(a.seed)
    Q = rng.uniform(0, 1, (300, 2))
    S = rng.uniform(0, 1, (4000, 2))
    S = np.concatenate([S, [[0, 0], [0, 1], [1, 0], [1, 1]]])
    m = a.m
    nodes, order = bernstein_nodes(m)
    Bm = lambda x: exact(nodes, order, x)[order[-1]]
    e_bern = max(abs(Bm(x) - f_target(*x)) for x in Q)
    rows = []
    for side in (8, 16, 32, 64):
        t0 = time.time()
        L = build_forest(nodes, order, S, side)
        P = [run_forest(nodes, order, x) for x in Q]
        rows.append(dict(side=side, trees=len(order), leaves=L,
                         err_vs_f=max(abs(p - f_target(*x)) for p, x in zip(P, Q)),
                         err_vs_Bm=max(abs(p - Bm(x)) for p, x in zip(P, Q)), seconds=time.time() - t0))
        print("bernstein", m, side, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    save(f"bernstein_m{m}", dict(m=m, trees=len(order), bernstein_err=e_bern, rows=rows))


# ------------------------------------------------------------------------------- Feynman
FEYN = {   # name: (variables, ranges, sympy expression string)
    "I.6.20": ("s t", [(1, 3), (1, 3)], "exp(-(t/s)**2/2)/(sqrt(2*pi)*s)"),
    "I.6.20b": ("s t t1", [(1, 3)] * 3, "exp(-((t-t1)/s)**2/2)/(sqrt(2*pi)*s)"),
    "I.12.11": ("q ef b v th", [(1, 5)] * 5, "q*(ef + b*v*sin(th))"),
    "I.13.12": ("m1 m2 r1 r2 g", [(1, 5)] * 5, "g*m1*m2*(1/r2 - 1/r1)"),
    "I.29.16": ("x1 x2 t1 t2", [(1, 5)] * 4, "sqrt(x1**2 + x2**2 - 2*x1*x2*cos(t1 - t2))"),
    "I.37.4": ("i1 i2 dl", [(1, 5)] * 3, "i1 + i2 + 2*sqrt(i1*i2)*cos(dl)"),
    "I.50.26": ("x1 w t al", [(1, 3)] * 4, "x1*(cos(w*t) + al*cos(w*t)**2)"),
    "II.6.15a": ("ef pd r x y z", [(1, 3)] * 6, "pd/(4*pi*ef)*3*z/r**5*sqrt(x**2 + y**2)"),
    "I.9.18": ("m1 m2 x2 x1 y2 y1 z2 z1 g",
               [(1, 2), (1, 2), (3, 4), (1, 2), (3, 4), (1, 2), (3, 4), (1, 2), (1, 2)],
               "g*m1*m2/((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)"),
}


def expr_forest(names, expr):
    """Fan-in <= 2 DAG of the expression: n-ary Add/Mul become chains of binary nodes,
    constants are folded into unary nodes, identical subexpressions share one node."""
    syms = sp.symbols(names)
    idx = {s: i for i, s in enumerate(syms)}
    nodes, order, memo = {}, [], {}

    def new(fn, kids, key):
        if key in memo:
            return memo[key]
        k = f"n{len(order)}"
        nodes[k] = Node(fn, kids); order.append(k)
        memo[key] = ("n", k)
        return memo[key]

    def lam(e):
        return sp.lambdify(sp.Symbol("u"), e, "math")

    def walk(e):
        if e in idx:
            return ("x", idx[e])
        free = e.free_symbols
        args = list(e.args)
        if isinstance(e, (sp.Add, sp.Mul)):
            const = [a for a in args if not a.free_symbols]
            var = [a for a in args if a.free_symbols]
            c = e.func(*const) if const else None
            ch = walk(var[0])
            acc_expr = var[0]
            for v in var[1:]:
                ch2 = walk(v)
                acc_expr = e.func(acc_expr, v)
                op = (lambda p, q: p + q) if isinstance(e, sp.Add) else (lambda p, q: p * q)
                ch = new(op, [ch, ch2], acc_expr)
            if c is not None:
                cv = float(c)
                op = (lambda p, cv=cv: p + cv) if isinstance(e, sp.Add) else (lambda p, cv=cv: p * cv)
                ch = new(op, [ch], e)
            return ch
        # unary functions and powers of one variable argument
        var = [a for a in args if a.free_symbols]
        if len(var) == 1:
            u = sp.Symbol("u")
            g = lam(e.xreplace({var[0]: u}))
            return new(lambda p, g=g: float(g(p)), [walk(var[0])], e)
        if len(var) == 2:                               # e.g. Pow(a, b) with both variable
            u, w = sp.symbols("u w")
            g = sp.lambdify((u, w), e.func(u, w), "math")
            return new(lambda p, q, g=g: float(g(p, q)), [walk(var[0]), walk(var[1])], e)
        raise ValueError(e)
    root = walk(sp.sympify(expr, locals={s.name: s for s in syms}))
    if root[0] == "x":
        raise ValueError("trivial")
    if order[-1] != root[1]:
        order.remove(root[1]); order.append(root[1])
    return nodes, order


def run_feynman(a):
    names, ranges, expr = FEYN[a.eq]
    n = len(ranges)
    rng = np.random.default_rng(a.seed)
    lo = np.array([r[0] for r in ranges], float); hi = np.array([r[1] for r in ranges], float)
    draw = lambda m: lo + rng.uniform(0, 1, (m, n)) * (hi - lo)
    Q, S = draw(500), np.concatenate([draw(5000), lo[None], hi[None]])
    nodes, order = expr_forest(names, expr)
    f = lambda x: exact(nodes, order, x)[order[-1]]
    fq = np.array([f(x) for x in Q])
    sd = float(np.std(fq))
    out = dict(eq=a.eq, n=n, trees=len(order), fanin=max(len(nd.kids) for nd in nodes.values()),
               sd_f=sd, forest=[], single=[])
    for side in (4, 8, 16, 32, 64, 128):
        L = build_forest(nodes, order, S, side, clamp=True)
        P = np.array([run_forest(nodes, order, x) for x in Q])
        out["forest"].append(dict(side=side, leaves=L, rmse=float(np.sqrt(np.mean((P - fq) ** 2))) / sd,
                                  sup=float(np.max(np.abs(P - fq))) / sd))
        print("feynman forest", a.eq, L, round(out["forest"][-1]["rmse"], 4), flush=True)
    budget = 300000
    sides = [s for s in range(2, 200) if s ** n <= budget]
    sides = sorted(set([sides[0], sides[len(sides) // 3], sides[2 * len(sides) // 3], sides[-1]]))
    for s in sides:
        t, G = grid_tree(lambda p: f(p), lo, hi, s)
        P = np.array([tree_predict(t, x) for x in Q])
        out["single"].append(dict(side=s, leaves=G, rmse=float(np.sqrt(np.mean((P - fq) ** 2))) / sd,
                                  sup=float(np.max(np.abs(P - fq))) / sd))
        print("feynman single", a.eq, G, round(out["single"][-1]["rmse"], 4), flush=True)
    save(f"feynman_{a.eq}", out)


def run_constants(a):
    """The constant of thm:compose for each formula's forest: every tree errs by eta_v ~ L_v w_v / m
    (prop:rate), and an error at v is amplified by at most A_v, the largest product of the
    Lipschitz constants along a path from v's consumers to the output.  kappa = sum_v L_v w_v A_v,
    divided by sd(f), predicts how hard each formula is for a forest at fixed m."""
    out = {}
    rng = np.random.default_rng(0)
    for eq, (names, ranges, expr) in FEYN.items():
        nodes, order = expr_forest(names, expr)
        n = len(ranges)
        lo = np.array([r[0] for r in ranges], float); hi = np.array([r[1] for r in ranges], float)
        S = lo + rng.uniform(0, 1, (3000, n)) * (hi - lo)
        V = [exact(nodes, order, x) for x in S]
        Lv, wv = {}, {}
        for k in order:
            nd = nodes[k]
            cols = np.array([[x[c[1]] if c[0] == "x" else v[c[1]] for c in nd.kids] for x, v in zip(S, V)])
            wv[k] = float(np.max(cols.max(0) - cols.min(0)))
            eps = 1e-5 * np.maximum(cols.max(0) - cols.min(0), 1e-6)
            g = []
            for u in cols[:1000]:
                grad = 0.0
                for j in range(len(u)):
                    d = np.zeros(len(u)); d[j] = eps[j]
                    try:
                        grad += abs(nd.fn(*(u + d)) - nd.fn(*(u - d))) / (2 * eps[j])
                    except ValueError:                 # at the edge of the domain: one-sided
                        grad += abs(nd.fn(*(u + d)) - nd.fn(*u)) / eps[j]
                g.append(grad)                       # l1 norm of the gradient = max-norm Lipschitz
            Lv[k] = float(np.max(g))
        consumers = {k: [j for j in order if ("n", k) in nodes[j].kids] for k in order}
        A = {}
        for k in reversed(order):
            A[k] = 1.0 if not consumers[k] else max(Lv[j] * A[j] for j in consumers[k])
        fvals = np.array([v[order[-1]] for v in V])
        kappa = sum(Lv[k] * wv[k] * A[k] for k in order) / float(np.std(fvals))
        out[eq] = dict(kappa=kappa, n=n, trees=len(order), L=Lv, w=wv, A=A)
        print(eq, "kappa/sd(f) =", round(kappa, 1), flush=True)
    save("constants", out)


def bump_nodes(n, c, rho):
    """f(x) = (rho - max_i |x_i - c_i|)_+ as a fan-in-2 composition of 1-Lipschitz maps:
    one-input trees |x_i - c_i|, a balanced binary tree of max, and the ramp (rho - t)_+."""
    nodes, order = {}, []

    def add(key, fn, kids):
        nodes[key] = Node(fn, kids); order.append(key)
        return ("n", key)
    level = [add(f"a{i}", lambda t, ci=float(c[i]): abs(t - ci), [("x", i)]) for i in range(n)]
    k = 0
    while len(level) > 1:
        nxt = [add(f"m{k + j}", lambda u, v: max(u, v), [level[2 * j], level[2 * j + 1]]) for j in range(len(level) // 2)]
        k += len(level) // 2
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    add("out", lambda t, r=float(rho): max(r - t, 0.0), [level[0]])
    return nodes, order


def run_separation(a):
    """thm:separation: for every single tree with G leaves there is a fan-in-2, 1-Lipschitz
    composition (a bump in an empty cell) on which its error is at least ~G^(-1/n)/4; a forest
    of two-input trees approximates the same bump at rate G^(-1/2) per tree."""
    rng = np.random.default_rng(a.seed)
    out = {}
    for n in (2, 4, 8):
        rows = []
        sides = {2: (8, 16, 32, 64), 4: (4, 6, 8, 10), 8: (2, 3, 4, 5)}[n]
        for s in sides:
            c = np.full(n, (s // 2) / s)                  # a cell corner: every leaf (cell centre) is >= 0.5/s away
            rho = 0.5 / s
            nodes, order = bump_nodes(n, c, rho)
            f = lambda x: exact(nodes, order, x)[order[-1]]
            Q = np.concatenate([c[None], c + rng.uniform(-rho, rho, (200, n)), rng.uniform(0, 1, (100, n))])
            t, G = grid_tree(lambda p: f(p), np.zeros(n), np.ones(n), s)
            e_single = max(abs(tree_predict(t, x) - f(x)) for x in Q)
            # boxes must cover each vertex's input set over all of [0,1]^n (thm:compose): include the
            # centre (where every |x_i - c_i| and every max is 0) and corners (where they are largest)
            corners = rng.integers(0, 2, (200, n)).astype(float)
            S = np.concatenate([rng.uniform(0, 1, (3000, n)), c + rng.uniform(-rho, rho, (1000, n)),
                                c[None], corners])
            frs = []
            for fs in (8, 16, 32, 64):
                L = build_forest(nodes, order, S, fs, clamp=True)
                e = max(abs(run_forest(nodes, order, x) - f(x)) for x in Q)
                frs.append(dict(side=fs, leaves=L, sup_err=e))
            rows.append(dict(side=s, G_single=G, rho=rho, single_err=e_single, predicted_single_err=rho, forest=frs))
            print("separation", n, G, "single", round(e_single, 4), "rho", round(rho, 4),
                  "forest", [(r["leaves"], round(r["sup_err"], 4)) for r in frs], flush=True)
        out[n] = rows
    save("separation", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["bernstein", "feynman", "graph", "constants", "separation"])
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--eq", default="I.12.11")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.which == "graph":
        for eq, (names, _, expr) in FEYN.items():
            nodes, order = expr_forest(names, expr)
            print(eq, len(names.split()), "inputs,", len(order), "trees, fan-in",
                  max(len(nd.kids) for nd in nodes.values()))
        return
    {"bernstein": run_bernstein, "feynman": run_feynman, "constants": run_constants, "separation": run_separation}[a.which](a)


if __name__ == "__main__":
    main()
