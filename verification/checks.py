"""Executable verifier for theory/main.tex.

Each check instantiates the objects a statement talks about -- with the real C++ tree, built
either by Cobweb's ``ifit`` or, for the constructions in the proofs, exactly
(``cobweb_forest.construct``) -- and tests the statement's conclusion numerically.  A check
returns a dict with the label of the statement it verifies, ``passed`` and the measured
quantities.  ``verification/run.py`` runs them all; ``tests/test_theory.py`` asserts them.
"""
from __future__ import annotations

import math

import numpy as np

from cobweb_forest import CobwebHybridTree, Forest, TreeSpec
from cobweb_forest import designs as D
from cobweb_forest.construct import build, leaves_of, nodes_of

BIG = 10 ** 7
F32 = np.float32


def _q(*v):
    return np.array(v, dtype=F32)


def _rng(seed):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------------------
# helpers: the formal model of Def. "concept" / "multi-node prediction", written from the
# definitions (not from the code), so that C0 can compare the two.
# ---------------------------------------------------------------------------------------
def ell(node, tree, disc, q):
    """log-likelihood l_c(q) of Def. concept: Dirichlet scoring with (K_A + 1) alpha, Gaussian
    with variance s^2 + h^2 on the observed (non-NaN) coordinates."""
    out = 0.0
    a = tree.alpha
    for attr, vals in disc.items():
        K = len(tree.attr_vals.get(attr, ()))
        if K == 0:
            continue
        n_a = node.a_count.get(attr, 0.0)
        for v, w in vals.items():
            n_v = node.av_count.get(attr, {}).get(v, 0.0) if v in tree.attr_vals[attr] else 0.0
            out += w * (math.log(n_v + a) - math.log(n_a + (K + 1) * a))
    if q.size:
        var = np.asarray(node.variance(), dtype=np.float64)
        mu = np.asarray(node.mean, dtype=np.float64)
        for j in range(q.size):
            if not np.isnan(q[j]):
                out += -0.5 * (math.log(2 * math.pi * var[j]) + (q[j] - mu[j]) ** 2 / var[j])
    return out


def formal_scores(tree, disc, q, weighting):
    """Scores of every concept under Def. multi-node prediction (full expansion)."""
    out = {}

    def rec(node, path_ll):
        l = ell(node, tree, disc, q)
        s = {"likelihood": l, "prior": l + math.log(node.count / tree.root.count),
             "path": path_ll + l, "posterior": path_ll}[weighting]
        if node is tree.root or node.uid == tree.root.uid:
            s = l            # the search is seeded with the root's likelihood under every weighting
        out[node.uid] = (s, node)
        if node.children:
            u = [math.log(c.count) + ell(c, tree, disc, q) for c in node.children]
            m = max(u)
            lse = m + math.log(sum(math.exp(x - m) for x in u))
            for c, uc in zip(node.children, u):
                rec(c, path_ll + uc - lse)
    rec(tree.root, 0.0)
    return out


def mixture(scored, dim, leaves_only=False):
    items = [(s, n) for s, n in scored.values() if not (leaves_only and n.children)]
    m = max(s for s, _ in items)
    w = np.array([math.exp(s - m) for s, _ in items])
    w /= w.sum()
    return float(sum(wi * n.mean[dim] for wi, (_, n) in zip(w, items))), w, [n for _, n in items]


def random_tree(seed, n=300, size=2, disc=True, add_only=False, dup_frac=0.0):
    rng = _rng(seed)
    t = CobwebHybridTree(size, alpha=0.5, prior_var=0.05)
    t.set_random_seed(seed)
    t.add_only = add_only
    data = []
    for i in range(n):
        if data and rng.random() < dup_frac:
            d, c = data[rng.integers(len(data))]
        else:
            c = rng.normal(size=size).astype(F32)
            d = {0: {int(rng.integers(4)): 1.0}} if disc else {}
        data.append((d, c))
        t.ifit(d, c)
    return t, data


# ---------------------------------------------------------------------------------------
# C0  Def. multi-node prediction / Def. concept  ==  the implementation
# ---------------------------------------------------------------------------------------
def check_model_matches_code(seed=0):
    t, data = random_tree(seed, n=150)
    rng = _rng(seed + 1)
    worst_s, worst_m = 0.0, 0.0
    for _ in range(10):
        q = _q(rng.normal(), np.nan)
        disc = {0: {int(rng.integers(4)): 1.0}}
        for w in ("likelihood", "prior", "path", "posterior"):
            formal = formal_scores(t, disc, q, w)
            code = {n.uid: s for s, _, n in t.expanded(disc, q, BIG, False, w)}
            assert set(code) == set(formal)
            worst_s = max(worst_s, max(abs(code[u] - formal[u][0]) / (1 + abs(formal[u][0])) for u in code))
            for lo in (False, True):
                m_formal, _, _ = mixture(formal, 1, lo)
                m_code = float(t.predict_mean(disc, q, BIG, False, w, lo)[1])
                worst_m = max(worst_m, abs(m_formal - m_code))
    # prediction of a nominal value uses K_A alpha in the denominator (no extra slot)
    leaf = leaves_of(t)[0]
    K = len(t.attr_vals[0])
    n_a = leaf.a_count.get(0, 0.0)
    v = next(iter(t.attr_vals[0]))
    p_code = leaf.predict_probs()[0][v]
    p_formal = (leaf.av_count.get(0, {}).get(v, 0.0) + t.alpha) / (n_a + K * t.alpha)
    ok = worst_s < 1e-4 and worst_m < 1e-4 and abs(p_code - p_formal) < 1e-9
    return dict(label="def:predict", passed=bool(ok), max_rel_score_diff=worst_s,
                max_mixture_diff=worst_m, predict_probs_uses_K_alpha=abs(p_code - p_formal) < 1e-9)


# ---------------------------------------------------------------------------------------
# C1  Lemma leaf preservation (with the corrected (iii))
# ---------------------------------------------------------------------------------------
def check_leaf_preservation(seed=0):
    results = {}
    for mode in ("full", "add_only", "add_only+new"):
        rng = _rng(seed)
        t = CobwebHybridTree(2, alpha=0.5, prior_var=0.05)
        t.set_random_seed(seed)
        t.add_only = mode != "full"
        t.allow_new = mode == "add_only+new"
        seen_leaves, data, ok_monotone = set(), [], True
        for i in range(400):
            if data and rng.random() < 0.15:
                d, c = data[rng.integers(len(data))]          # exact duplicate
            else:
                c = rng.normal(size=2).astype(F32)
                d = {0: {int(rng.integers(3)): 1.0}}
            data.append((d, c))
            t.ifit(d, c)
            now = {n.uid for n in leaves_of(t)}
            ok_monotone &= seen_leaves <= now
            seen_leaves = now
        ok_add = True
        for n in nodes_of(t):
            if n.children:
                cnt = sum(c.count for c in n.children)
                mean = sum(c.count * np.asarray(c.mean) for c in n.children) / cnt
                ok_add &= abs(cnt - n.count) < 1e-6 and np.allclose(mean, n.mean, atol=1e-4)
        leaves = leaves_of(t)
        identical = all(np.all(np.asarray(l.sum_sq) < 1e-5) for l in leaves)     # one instance value per leaf
        distinct = len({(tuple(sorted((a, tuple(sorted(v.items()))) for a, v in d.items())),
                         tuple(np.round(c, 5))) for d, c in data})
        results[mode] = dict(no_leaf_deleted=bool(ok_monotone), additive=bool(ok_add),
                             leaves_hold_identical_instances=bool(identical),
                             n_leaves=len(leaves), n_distinct=distinct, n_instances=len(data),
                             leaves_ge_distinct=len(leaves) >= distinct)
    ok = all(r["no_leaf_deleted"] and r["additive"] and r["leaves_hold_identical_instances"]
             and r["leaves_ge_distinct"] for r in results.values())
    return dict(label="lem:leaves", passed=bool(ok), **results)


# ---------------------------------------------------------------------------------------
# C2  Lemma concept bias / certificate
# ---------------------------------------------------------------------------------------
def check_bias_certificate(seed=0):
    rng = _rng(seed)
    g = lambda x: math.sin(3 * x[0]) / 3 + math.cos(2 * x[1]) / 2      # Lipschitz, L = sqrt(2)
    L = math.sqrt(2)
    t = CobwebHybridTree(3, alpha=0.5, prior_var=0.01)
    t.set_random_seed(seed)
    for _ in range(300):
        x = rng.uniform(-1, 1, 2)
        t.ifit({}, _q(x[0], x[1], g(x)))
    worst = -1e9
    for _ in range(40):
        x = rng.uniform(-1, 1, 2)
        q = _q(x[0], x[1], np.nan)
        for w in ("likelihood", "prior", "path", "posterior"):
            for M in (20, BIG):
                exp_ = t.expanded({}, q, M, False, w)
                s = np.array([e[0] for e in exp_])
                wt = np.exp(s - s.max()); wt /= wt.sum()
                yhat = sum(wi * e[2].mean[2] for wi, e in zip(wt, exp_))
                bound = L * sum(wi * (np.linalg.norm(x - np.asarray(e[2].mean[:2])) +
                                      math.sqrt(float(np.sum(np.asarray(e[2].sum_sq[:2]) / e[2].count))))
                                for wi, e in zip(wt, exp_))
                worst = max(worst, abs(yhat - g(x)) - bound)
    return dict(label="lem:bias", passed=bool(worst <= 1e-5), max_violation=worst)


# ---------------------------------------------------------------------------------------
# C3/C4  Thm single (depth-one grid tree, rate) and Prop prior (root weight stays > 0)
# ---------------------------------------------------------------------------------------
def _grid_tree(G_side, d, g, beta=1.0):
    delta = 1.0 / G_side
    h = delta / beta
    axes = [np.arange(G_side) * delta + delta / 2] * d
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, d)
    spec = [({}, np.concatenate([p, [g(p)]])) for p in pts]
    return build(spec, d + 1, prior_var=h * h), h


def check_single_tree(seed=0):
    rng = _rng(seed)
    out = {}
    for d, g, Ls, sides in ((1, lambda p: math.sin(2 * math.pi * p[0]), 2 * math.pi, (8, 16, 32, 64, 128)),
                            (2, lambda p: math.sin(2 * math.pi * p[0]) * math.cos(math.pi * p[1]),
                             math.hypot(2 * math.pi, math.pi), (6, 12, 24))):
        X = rng.uniform(0, 1, (150, d))
        rows = []
        for G_side in sides:
            t, h = _grid_tree(G_side, d, g)
            errs = {"lik": 0.0, "path": 0.0, "leaves": 0.0, "prior": 0.0}
            root_w = {"lik": 0.0, "path": 0.0, "prior": 1.0}
            for x in X:
                q = _q(*x, np.nan)
                for key, w, lo in (("lik", "likelihood", False), ("path", "path", False),
                                   ("leaves", "path", True), ("prior", "prior", False)):
                    errs[key] = max(errs[key], abs(float(t.predict_mean({}, q, BIG, False, w, lo)[d]) - g(x)))
                for key, w in (("lik", "likelihood"), ("path", "path"), ("prior", "prior")):
                    e = t.expanded({}, q, BIG, False, w)
                    s = np.array([z[0] for z in e]); wt = np.exp(s - s.max()); wt /= wt.sum()
                    rw = float(wt[[z[2].uid for z in e].index(t.root.uid)])
                    root_w[key] = max(root_w[key], rw) if key != "prior" else min(root_w[key], rw)
            G = G_side ** d
            rows.append(dict(G=G, h=h, sup_err=errs, max_root_w=root_w,
                             bound_shape=Ls * G ** (-1 / d)))
        out[f"d{d}"] = rows
    ok = True
    for rows in out.values():
        for a, b in zip(rows, rows[1:]):
            for key in ("lik", "path", "leaves"):
                ok &= b["sup_err"][key] < a["sup_err"][key]               # error decreases with G
            ok &= b["max_root_w"]["lik"] < a["max_root_w"]["lik"]       # root weight vanishes
        for r in rows:
            for key in ("lik", "path", "leaves"):
                ok &= r["sup_err"][key] <= 3 * r["bound_shape"]          # within the rate's shape
    ok_prior = all(r["max_root_w"]["prior"] > 0.05 for rows in out.values() for r in rows) and \
        all(rows[-1]["sup_err"]["prior"] > 0.1 for rows in out.values())
    return dict(label="thm:single+prop:prior", passed=bool(ok and ok_prior), single_ok=bool(ok),
                prior_inconsistent=bool(ok_prior), rows=out)


# ---------------------------------------------------------------------------------------
# C5  Thm label forests (construction)
# ---------------------------------------------------------------------------------------
def check_label_construction(seed=0):
    rng = _rng(seed)
    f = lambda x: x[0] * x[1] + math.sin(math.pi * x[0])
    Lf, fmax = math.hypot(1 + math.pi, 1), 2.0
    out = []
    for K, alpha in ((8, 1e-3), (16, 1e-4), (32, 1e-5)):
        delta = 1.0 / K
        grid = np.arange(K) * delta + delta / 2
        uni = build([({}, np.array([g])) for g in grid], 1, prior_var=(delta / 2) ** 2)
        leaf_uid = {l.uid: int(round((l.mean[0] - delta / 2) / delta)) for l in leaves_of(uni)}
        spec = [({0: {i: 1.0}, 1: {j: 1.0}}, np.array([f((grid[i], grid[j]))])) for i in range(K) for j in range(K)]
        out_tree = build(spec, 1, alpha=alpha, prior_var=1e-4)
        worst, worst_bound, ok_nn = 0.0, 0.0, True
        for x in rng.uniform(0, 1, (200, 2)):
            labs = []
            for p in range(2):
                lab = leaf_uid[uni.get_leaf({}, _q(x[p])).uid]
                ok_nn &= lab == int(np.argmin(np.abs(grid - x[p])))
                labs.append(lab)
            yhat = float(out_tree.predict_mean({0: {labs[0]: 1.0}, 1: {labs[1]: 1.0}}, _q(np.nan), BIG, False,
                                               "likelihood", True)[0])
            bound = Lf * math.sqrt(2) * delta / 2 + 2 * fmax * ((1 + K * alpha) ** 2 - 1)
            worst = max(worst, abs(yhat - f(x)) - bound)
            worst_bound = max(worst_bound, bound)
        out.append(dict(K=K, alpha=alpha, max_violation=worst, bound=worst_bound, labels_are_nearest=bool(ok_nn)))
    ok = all(r["max_violation"] <= 1e-6 and r["labels_are_nearest"] for r in out)
    return dict(label="thm:label", passed=bool(ok), rows=out)


# ---------------------------------------------------------------------------------------
# C6  Prop interweaving (exact mixture identity)
# ---------------------------------------------------------------------------------------
def _dump(tree):
    import json, os, tempfile
    fd, p = tempfile.mkstemp(suffix=".json"); os.close(fd)
    tree.dump_json(p)
    doc = json.load(open(p, encoding="utf-8")); os.remove(p)
    return doc


def check_interweave(seed=0):
    import json, os, tempfile
    rng = _rng(seed)
    trees = []
    for j in range(2):
        t = CobwebHybridTree(2, alpha=0.5, prior_var=0.05)
        t.set_random_seed(seed + j)
        for _ in range(80):
            t.ifit({}, (rng.normal(size=2) + 3 * j).astype(F32))
        trees.append(t)
    docs = [_dump(t) for t in trees]
    nodes, roots = [], []
    for doc in docs:
        off = len(nodes) + 1
        for nd in doc["nodes"]:
            nd = dict(nd)
            nd["id"] = nd["id"] + off
            nd["parent_id"] = 0 if nd["parent_id"] is None else nd["parent_id"] + off
            if nd["parent_id"] == 0:
                roots.append(nd)
            nodes.append(nd)
    n = sum(r["count"] for r in roots)
    mean = sum(r["count"] * np.array(r["mean"]) for r in roots) / n
    ss = sum(np.array(r["sum_sq"]) + r["count"] * (np.array(r["mean"]) - mean) ** 2 for r in roots)
    root = {"id": 0, "parent_id": None, "concept_id": "R", "count": n, "a_count": {}, "sum_n_logn": {},
            "av_count": {}, "mean": mean.tolist(), "sum_sq": ss.tolist()}
    doc = dict(docs[0]); doc["nodes"] = [root] + nodes
    fd, p = tempfile.mkstemp(suffix=".json"); os.close(fd)
    json.dump(doc, open(p, "w", encoding="utf-8"))
    T = CobwebHybridTree(2, alpha=0.5, prior_var=0.05); T.load_json(p); os.remove(p)
    worst = 0.0
    for _ in range(30):
        q = _q(rng.normal() * 2 + 1.5, np.nan)
        for w in ("likelihood", "path"):
            yT = float(T.predict_mean({}, q, BIG, False, w, False)[1])
            sub = [float(t.predict_mean({}, q, BIG, False, w, False)[1]) for t in trees]
            Z = [sum(math.exp(e[0]) for e in t.expanded({}, q, BIG, False, w)) for t in trees]
            lr = math.exp(ell(T.root, T, {}, q))
            if w == "path":
                u = [math.log(t.root.count) + ell(t.root, t, {}, q) for t in trees]
                m = max(u); post = [math.exp(x - m) for x in u]; s = sum(post); post = [x / s for x in post]
                lam = [pj * zj for pj, zj in zip(post, Z)]
            else:
                lam = Z
            tot = lr + sum(lam)
            pred = (lr * T.root.mean[1] + sum(l * y for l, y in zip(lam, sub))) / tot
            worst = max(worst, abs(pred - yT))
    return dict(label="prop:interweave", passed=bool(worst < 1e-3), max_abs_diff=worst)


# ---------------------------------------------------------------------------------------
# C7  Lemma DAG / Thm CKA mechanism: composing constructed trees approximates a KA-form f
# ---------------------------------------------------------------------------------------
def check_composition(seed=0):
    rng = _rng(seed)
    phi1, phi2, Phi = (lambda x: math.sin(math.pi * x)), (lambda x: x * x), (lambda u: math.exp(u / 2))
    f = lambda x: Phi(phi1(x[0]) + phi2(x[1]))
    rows = []
    for G in (8, 16, 32, 64):
        def uni(fn, lo, hi):
            pts = np.linspace(lo, hi, G)
            d = (hi - lo) / (G - 1)
            return build([({}, np.array([p, fn(p)])) for p in pts], 2, prior_var=d * d)
        T1, T2, TP = uni(phi1, -0.1, 1.1), uni(phi2, -0.1, 1.1), uni(Phi, -0.2, 2.3)
        a = np.linspace(-0.2, 1.2, G); b = np.linspace(-0.2, 1.2, G)
        da = (a[1] - a[0])
        TA = build([({}, np.array([u, v, u + v])) for u in a for v in b], 3, prior_var=da * da)
        err = 0.0
        for x in rng.uniform(0, 1, (150, 2)):
            m1 = float(T1.predict_mean({}, _q(x[0], np.nan), BIG, False, "likelihood", True)[1])
            m2 = float(T2.predict_mean({}, _q(x[1], np.nan), BIG, False, "likelihood", True)[1])
            s = float(TA.predict_mean({}, _q(m1, m2, np.nan), BIG, False, "likelihood", True)[2])
            y = float(TP.predict_mean({}, _q(s, np.nan), BIG, False, "likelihood", True)[1])
            err = max(err, abs(y - f(x)))
        rows.append(dict(G=G, sup_err=err))
    ok = all(b["sup_err"] < a["sup_err"] for a, b in zip(rows, rows[1:])) and rows[-1]["sup_err"] < 0.05
    return dict(label="lem:dag+thm:cka", passed=bool(ok), rows=rows)


# ---------------------------------------------------------------------------------------
# C8  Thm consistency: a learned tree's leaves-only LIK mixture IS Nadaraya-Watson
# ---------------------------------------------------------------------------------------
def check_tree_is_nw(seed=0):
    rng = _rng(seed)
    h = 0.15
    t = CobwebHybridTree(3, alpha=0.5, prior_var=h * h)
    t.set_random_seed(seed)
    X = rng.uniform(-1, 1, (400, 2)); Y = np.sin(3 * X[:, 0]) * X[:, 1] + 0.1 * rng.normal(size=400)
    for x, y in zip(X, Y):
        t.ifit({}, _q(x[0], x[1], y))
    worst_lik, worst_path = 0.0, 0.0
    for x in rng.uniform(-1, 1, (40, 2)):
        k = np.exp(-np.sum((X - x) ** 2, 1) / (2 * h * h))
        nw = float(np.sum(k * Y) / np.sum(k))
        lik = float(t.predict_mean({}, _q(x[0], x[1], np.nan), BIG, False, "likelihood", True)[2])
        path = float(t.predict_mean({}, _q(x[0], x[1], np.nan), BIG, False, "path", True)[2])
        worst_lik = max(worst_lik, abs(lik - nw)); worst_path = max(worst_path, abs(path - nw))
    return dict(label="thm:consistency", passed=bool(worst_lik < 1e-3), max_abs_diff_lik_vs_NW=worst_lik,
                max_abs_diff_path_vs_NW=worst_path,
                note="path leaves-only is not Nadaraya-Watson in a learned tree (structure-dependent)")


# ---------------------------------------------------------------------------------------
# C9  Prop activations can fail (xy) and labels repair it
# ---------------------------------------------------------------------------------------
def _xy_data(rng, N):
    X = rng.uniform(-1, 1, (N, 2)).astype(F32)
    return X, (X[:, 0] * X[:, 1])[:, None].astype(F32)


def check_xy(seed=0):
    """Prop xy: the funnel activations vanish and the funnel forest learns nothing.  Labels:
    at every label depth the error falls with N, and the best depth reaches well below the
    funnel forest (Thm labelcons: the depth must grow with N so that cells stay o(N))."""
    rng = _rng(seed)
    Xt, Yt = _xy_data(rng, 400)
    res = {}

    def run(mode, N, depth):
        X, Y = _xy_data(rng, N)
        F = Forest(D.ka((0, 1), mode, N, kappa=0.25, label_depth=depth, max_nodes=300, leaves_only=True), "out")
        F.fit(X, Y, protocol="online")
        return F, float(np.sqrt(np.mean((F.predict(Xt)[:, 0] - Yt[:, 0]) ** 2)) / Yt.std())

    F, res["funnel_N1000"] = run("act", 1000, 3)
    acts = np.array([[F.messages(x)[n]["act"][0] for n in ("t1", "t2")] for x in Xt[:200]])
    res["act_sd_over_sd_y"] = float(acts.std() / Yt.std())
    for depth in (2, 4):
        for N in (1000, 4000):
            res[f"label_d{depth}_N{N}"] = run("label", N, depth)[1]
    ok = (res["funnel_N1000"] > 0.9 and res["act_sd_over_sd_y"] < 0.2
          and all(res[f"label_d{d}_N4000"] < res[f"label_d{d}_N1000"] for d in (2, 4))
          and min(res["label_d2_N4000"], res["label_d4_N4000"]) < res["funnel_N1000"] - 0.3)
    return dict(label="prop:xy", passed=bool(ok), **res)


# ---------------------------------------------------------------------------------------
# C10 Thm locality obstruction (sign construction): local laws identical, composition differs
# ---------------------------------------------------------------------------------------
def _ks(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    d = np.max(np.abs(np.searchsorted(a, grid, "right") / len(a) - np.searchsorted(b, grid, "right") / len(b)))
    n = len(a) * len(b) / (len(a) + len(b))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 100))
    return d, max(0.0, min(1.0, p))


def sign_data(rng, N, world):
    X = rng.uniform(-1, 1, (N, 3)).astype(F32)
    other = X[:, 1] if world == "a" else X[:, 2]
    return X[:, :2].copy(), np.sign(X[:, 0] * other)[:, None].astype(F32)


def check_locality(seed=0):
    rng = _rng(seed)
    N = 3000
    Da, Db = sign_data(rng, N, "a"), sign_data(rng, N, "b")
    # (i) the univariate tree's training stream (X_2, Y) has the same law in both worlds
    pvals = []
    for yv in (-1, 1):
        pvals.append(_ks(Da[0][Da[1][:, 0] == yv, 1], Db[0][Db[1][:, 0] == yv, 1])[1])
    p_pos = (np.mean(Da[1] > 0), np.mean(Db[1] > 0))
    # (ii) composing tree: labels of both coordinates, outcome Y
    res = {}
    for world, (X, Y) in (("a", Da), ("b", Db)):
        Xt, Yt = sign_data(rng, 600, world)
        for mode in ("act", "label"):
            F = Forest(D.ka((0, 1), mode, N, kappa=0.25, label_depth=3, max_nodes=300, leaves_only=True), "out")
            F.fit(X, Y, protocol="online")
            res[f"{world}:{mode}"] = float(np.sqrt(np.mean((F.predict(Xt)[:, 0] - Yt[:, 0]) ** 2)))
    ok = min(pvals) > 1e-3 and abs(p_pos[0] - p_pos[1]) < 0.05 and \
        res["a:label"] < 0.6 and res["a:act"] > 0.9 and res["b:label"] > 0.9
    return dict(label="thm:locality", passed=bool(ok), ks_pvalues_x2_given_y=pvals,
                p_y_pos=[float(p) for p in p_pos], rmse=res)


# ---------------------------------------------------------------------------------------
# C11 Prop multi-view: the posterior given all views depends on the views only through the
#     per-view posterior means (hence through the cross-view activations)  [exact algebra]
# ---------------------------------------------------------------------------------------
def check_multiview(seed=0):
    rng = _rng(seed)
    r, dims = 2, (3, 4, 2)
    Lam = [rng.normal(size=(d, r)) for d in dims]
    Psi = [np.diag(rng.uniform(0.2, 1.0, d)) for d in dims]
    P0 = np.eye(r)
    worst = 0.0
    for _ in range(20):
        z = rng.normal(size=r)
        xs = [L @ z + rng.multivariate_normal(np.zeros(d), S) for L, S, d in zip(Lam, Psi, dims)]
        # joint posterior from all views
        Pj = P0 + sum(L.T @ np.linalg.inv(S) @ L for L, S in zip(Lam, Psi))
        mj = np.linalg.solve(Pj, sum(L.T @ np.linalg.inv(S) @ x for L, S, x in zip(Lam, Psi, xs)))
        # per-view posteriors combined: P = sum P_v - (m-1) P0,  P mu = sum P_v mu_v
        Pv = [P0 + L.T @ np.linalg.inv(S) @ L for L, S in zip(Lam, Psi)]
        mv = [np.linalg.solve(P, L.T @ np.linalg.inv(S) @ x) for P, L, S, x in zip(Pv, Lam, Psi, xs)]
        Pc = sum(Pv) - (len(dims) - 1) * P0
        mc = np.linalg.solve(Pc, sum(P @ m for P, m in zip(Pv, mv)))
        worst = max(worst, float(np.max(np.abs(Pc - Pj))), float(np.max(np.abs(mc - mj))))
    return dict(label="prop:multiview", passed=bool(worst < 1e-8), max_abs_diff=worst)


# ---------------------------------------------------------------------------------------
# C12 Remark: branch-and-bound early stop is unsound for Gaussian blocks
# ---------------------------------------------------------------------------------------
def check_branch_and_bound(seed=0):
    found = None
    for s in range(seed, seed + 30):
        t, _ = random_tree(s, n=200, size=2, disc=False)
        t2 = CobwebHybridTree(2, alpha=0.5, prior_var=1e-4)      # small variance: densities > 1
        t2.set_random_seed(s)
        rng = _rng(s)
        for _ in range(200):
            t2.ifit({}, (rng.normal(size=2) * 0.05).astype(F32))
        for q in rng.normal(size=(20, 2)) * 0.05:
            q = q.astype(F32)
            got = t2.categorize({}, q, True, False, BIG, "path")
            scores = {n.uid: s_ for s_, _, n in t2.expanded({}, q, BIG, False, "path")}
            if scores[got.uid] < max(scores.values()) - 1e-6:
                found = dict(seed=s, returned=scores[got.uid], best=max(scores.values()))
                break
        if found:
            break
    # nominal-only counterexample (found by the round-2 reviewer): root -> {curr (1 x a=0),
    # q -> {d (9 x a=0), e (76 x a=1)}}, query a=0.  `curr` is popped first and the stop fires.
    E0 = np.zeros(0, F32)
    tn = build([({0: {0: 1.0}}, E0, 1), [({0: {0: 1.0}}, E0, 9), ({0: {1: 1.0}}, E0, 76)]], 0, alpha=0.01)
    got = tn.categorize({0: {0: 1.0}}, E0, True, False, BIG, "path")
    scores = {n.uid: s_ for s_, _, n in tn.expanded({0: {0: 1.0}}, E0, BIG, False, "path")}
    nominal = dict(returned=scores[got.uid], best=max(scores.values()),
                   returned_is_root=got.uid == tn.root.uid)
    ok_nominal = scores[got.uid] < max(scores.values()) - 1e-6
    return dict(label="rem:structure (branch-and-bound)", passed=bool(found is not None and ok_nominal),
                counterexample_continuous=found, counterexample_nominal=nominal)


ALL = [check_model_matches_code, check_leaf_preservation, check_bias_certificate, check_single_tree,
       check_label_construction, check_interweave, check_composition, check_tree_is_nw, check_xy,
       check_locality, check_multiview, check_branch_and_bound]


# ---------------------------------------------------------------------------------------
# C13 Prop sufficient class: H_S(s) = E[Phi(s + Z)] is strictly increasing (Monte Carlo),
#     and a funnel forest on a monotone-additive target is close to f
# ---------------------------------------------------------------------------------------
def check_suffclass(seed=0):
    rng = _rng(seed)
    # exp100-like with 6 coordinates: Phi(s) = exp(s/6), phi(x) = sin^2(pi x / 2); S = first 2 coords
    Zrest = np.sum(np.sin(np.pi * rng.uniform(-1, 1, (200000, 4)) / 2) ** 2, 1)
    s_grid = np.linspace(0, 2, 41)
    H = np.array([np.mean(np.exp((s + Zrest) / 6)) for s in s_grid])
    mono = bool(np.all(np.diff(H) > 0))
    # funnel forest (online) on exp(sin(pi x1) + x2^2): error relative to sd(f) is small and falls with N
    f = lambda X: np.exp(np.sin(np.pi * X[:, 0]) + X[:, 1] ** 2)
    errs = {}
    Xt = rng.uniform(-1, 1, (400, 2)).astype(F32)
    for N in (1000, 8000):
        X = rng.uniform(-1, 1, (N, 2)).astype(F32)
        F = Forest(D.ka((0, 1), "act", N, kappa=0.25, max_nodes=300, leaves_only=True), "out")
        F.fit(X, f(X)[:, None].astype(F32), protocol="online")
        errs[N] = float(np.sqrt(np.mean((F.predict(Xt)[:, 0] - f(Xt)) ** 2)) / f(Xt).std())
    ok = mono and errs[8000] < errs[1000] and errs[8000] < 0.25
    return dict(label="prop:suffclass", passed=bool(ok), H_strictly_increasing=mono, rmse_over_sd=errs)


# ---------------------------------------------------------------------------------------
# C14 Cor parity: with Y = prod sign(X_j), j = 1..3, every pair of coordinates is independent
#     of Y; a label forest over the three coordinates learns Y, a funnel forest does not
# ---------------------------------------------------------------------------------------
def check_parity(seed=0):
    rng = _rng(seed)
    N = 6000
    X = rng.uniform(-1, 1, (N, 3)).astype(F32)
    Y = np.prod(np.sign(X), 1)
    # (X1, X2) | Y = +1 vs Y = -1: compare the law of x1*x2 and of x1, x2 by KS
    pv = []
    for stat in (X[:, 0] * X[:, 1], X[:, 0], X[:, 1], X[:, 0] + X[:, 1]):
        pv.append(_ks(stat[Y > 0], stat[Y < 0])[1])
    Xt = rng.uniform(-1, 1, (600, 3)).astype(F32); Yt = np.prod(np.sign(Xt), 1)
    res = {}
    for mode, depth in (("act", 3), ("label", 2)):
        F = Forest(D.ka((0, 1, 2), mode, N, kappa=0.25, label_depth=depth, max_nodes=300, leaves_only=True), "out")
        F.fit(X, Y[:, None].astype(F32), protocol="online")
        res[mode] = float(np.sqrt(np.mean((F.predict(Xt)[:, 0] - Yt) ** 2)))
    ok = min(pv) > 1e-3 and res["act"] > 0.9 and res["label"] < res["act"] - 0.3
    return dict(label="cor:parity", passed=bool(ok), ks_pvalues_pair_vs_y=pv, rmse=res)


# ---------------------------------------------------------------------------------------
# C15 Thm ideal forest (split protocol): sup error to F_A = f falls with N on a sufficient task
# ---------------------------------------------------------------------------------------
def check_ideal(seed=0):
    rng = _rng(seed)
    f = lambda X: np.exp(np.sin(np.pi * X[:, 0]) + X[:, 1] ** 2)
    g = np.linspace(-0.9, 0.9, 25)
    Xg = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2).astype(F32)
    errs = {}
    for N in (1500, 6000, 24000):
        X = rng.uniform(-1, 1, (N, 2)).astype(F32)
        F = Forest(D.ka((0, 1), "act", N, kappa=0.5, max_nodes=2000, leaves_only=True, weighting="likelihood"), "out")
        F.fit(X, f(X)[:, None].astype(F32), protocol="split")
        errs[N] = float(np.max(np.abs(F.predict(Xg)[:, 0] - f(Xg))))
    Ns = sorted(errs)
    ok = all(errs[b] < errs[a] for a, b in zip(Ns, Ns[1:]))
    return dict(label="thm:ideal", passed=bool(ok), sup_err=errs)


# ---------------------------------------------------------------------------------------
# C16 Thm selection: on xy the validation error at the output never picks the funnel forest
# ---------------------------------------------------------------------------------------
def check_selection(seed=0):
    picks = []
    for s in range(seed, seed + 3):
        rng = _rng(100 + s)
        tr, va = _xy_data(rng, 2000), _xy_data(rng, 500)
        cands = {"funnel": D.ka((0, 1), "act", 2000, kappa=0.25, max_nodes=300, leaves_only=True),
                 "label-d2": D.ka((0, 1), "label", 2000, kappa=0.25, label_depth=2, max_nodes=300, leaves_only=True),
                 "merge": D.single(2, 2000, kappa=0.25, max_nodes=300, leaves_only=True)}
        val = {}
        for name, specs in cands.items():
            F = Forest(specs, "out").fit(tr[0], tr[1], protocol="online")
            val[name] = float(np.sqrt(np.mean((F.predict(va[0])[:, 0] - va[1][:, 0]) ** 2)))
        picks.append(min(val, key=val.get))
    return dict(label="thm:selection", passed=all(p != "funnel" for p in picks), picks=picks)


ALL += [check_suffclass, check_parity, check_ideal, check_selection]


# ---------------------------------------------------------------------------------------
# C17 Provenance: every result the paper reports was produced by a build without the
#     leaves-only budget bug, or by code paths that bug cannot reach.
# ---------------------------------------------------------------------------------------
FIXED_BUILD = "296dd96809ff"          # leaf-counted budget (commit d78ffdd)
UNAFFECTED = {                        # experiment -> why the pre-fix build is equivalent
    "weighting": "full expansion (budget 1e7), never budget-limited",
    "label_cells": "labels only (root-to-leaf descent), no mixture",
    "owt": "all-concept mixtures (leaves_only=False); the fix changed only leaves-only budgets",
}


def check_provenance(root="results"):
    import glob, json, os
    bad, counts = [], {}
    for p in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        exp = os.path.relpath(p, root).split(os.sep)[0]
        h = d.get("code", {}).get("cobweb_forest/_core.cpython-312-x86_64-linux-gnu.so")
        methods = set(d.get("results", {}).keys())
        cobweb_free = methods and methods <= {"mlp", "kan", "ngram"}
        ok = h == FIXED_BUILD or exp in UNAFFECTED or cobweb_free
        counts[exp] = counts.get(exp, 0) + 1
        if not ok:
            bad.append((p, h))
    return dict(label="app:verify (provenance)", passed=not bad, files=counts, stale=bad[:20],
                unaffected_rules=UNAFFECTED)


ALL += [check_provenance]


# ---------------------------------------------------------------------------------------
# C18 Count-weighted leaves (leaves-only PRIOR) = Nadaraya-Watson with multiplicities,
#     also when instances repeat (atoms), where the count-blind LIK mixture fails.
#     (Found by the calibrated reviewer: Thm labelcons fails for targets with atoms.)
# ---------------------------------------------------------------------------------------
def check_count_weighting(seed=0):
    rng = _rng(seed)
    # (a) a learned tree on data with exact repeats: leaves-only PRIOR = NW with multiplicities
    h = 0.2
    t = CobwebHybridTree(2, alpha=0.5, prior_var=h * h)
    t.set_random_seed(seed)
    base = rng.uniform(-1, 1, 60)
    X = rng.choice(base, 600).astype(F32)                     # each x repeats ~10 times
    Y = (rng.random(600) < 0.5 + 0.4 * np.sin(2 * X)).astype(F32)   # Bernoulli: atoms in y
    for x, y in zip(X, Y):
        t.ifit({}, _q(x, y))
    worst_prior, worst_lik = 0.0, 0.0
    for x in np.linspace(-0.9, 0.9, 25):
        k = np.exp(-(X - x) ** 2 / (2 * h * h))
        nw = float(np.sum(k * Y) / np.sum(k))
        pr = float(t.predict_mean({}, _q(x, np.nan), BIG, False, "prior", True)[1])
        lk = float(t.predict_mean({}, _q(x, np.nan), BIG, False, "likelihood", True)[1])
        worst_prior, worst_lik = max(worst_prior, abs(pr - nw)), max(worst_lik, abs(lk - nw))
    # (b) the label-forest construction with a Bernoulli target: per cell, PRIOR leaves give
    #     the partitioning estimate, LIK leaves give the average of the distinct y values
    K = 5
    cells = rng.integers(0, K, 1500)
    p = 0.2 + 0.6 * (cells + 0.5) / K
    Yc = (rng.random(1500) < p).astype(F32)
    out = CobwebHybridTree(1, alpha=1e-4, prior_var=1e-3)
    out.set_random_seed(seed)
    for c, y in zip(cells, Yc):
        out.ifit({0: {int(c): 1.0}}, _q(y))
    gap_prior, gap_lik = 0.0, 0.0
    for c in range(K):
        part = float(Yc[cells == c].mean())
        pr = float(out.predict_mean({0: {c: 1.0}}, _q(np.nan), BIG, False, "prior", True)[0])
        lk = float(out.predict_mean({0: {c: 1.0}}, _q(np.nan), BIG, False, "likelihood", True)[0])
        gap_prior, gap_lik = max(gap_prior, abs(pr - part)), max(gap_lik, abs(lk - part))
    ok = worst_prior < 1e-3 and gap_prior < 0.01 and worst_lik > 0.02 and gap_lik > 0.05
    return dict(label="thm:consistency+thm:labelcons (counts)", passed=bool(ok),
                prior_vs_nw=worst_prior, lik_vs_nw=worst_lik,
                prior_vs_partitioning=gap_prior, lik_vs_partitioning=gap_lik)


ALL += [check_count_weighting]


# ---------------------------------------------------------------------------------------
# C19 Weightings on a hierarchy (found by the round-3 reviewer): on a balanced binary tree over
#     a grid, the PRIOR all-concept mixture converges (slowly, each level carrying O(1) mass),
#     unlike on the depth-one tree of Prop prior; the POSTERIOR mixture keeps a floor even
#     restricted to leaves.
# ---------------------------------------------------------------------------------------
def _binary(points):
    if len(points) == 1:
        return points[0]
    m = len(points) // 2
    return [_binary(points[:m]), _binary(points[m:])]


def check_hierarchy_weightings(seed=0):
    g = lambda x: math.sin(math.pi * (2 * x - 1))
    xs = np.linspace(0.3, 0.7, 41)                      # interior queries
    rows = []
    for D in (5, 7, 9, 11):
        G = 2 ** D
        grid = (np.arange(G) + 0.5) / G
        t = build(_binary([({}, np.array([x, g(x)])) for x in grid]), 2, prior_var=(1.0 / G) ** 2)
        err = {}
        for key, w, lo in (("prior_all", "prior", False), ("posterior_leaves", "posterior", True),
                           ("lik_leaves", "likelihood", True)):
            e = [float(t.predict_mean({}, _q(x, np.nan), BIG, False, w, lo)[1]) - g(x) for x in xs]
            err[key] = float(np.sqrt(np.mean(np.square(e))))
        rows.append(dict(D=D, G=G, **err))
    prior = [r["prior_all"] for r in rows]
    post = [r["posterior_leaves"] for r in rows]
    ok = (all(b < a for a, b in zip(prior, prior[1:]))           # prior all-concept keeps falling
          and post[-1] > 0.5 * post[0] and post[-1] > 0.01       # posterior leaves-only: a floor
          and rows[-1]["lik_leaves"] < 0.01)                      # control: lik leaves-only converges
    return dict(label="prop:prior (hierarchy)", passed=bool(ok), rows=rows)


ALL += [check_hierarchy_weightings]


# ---------------------------------------------------------------------------------------
# Cue calculus: the Lean proofs (lean/) and their empirical counterpart (cobweb_forest.cues)
# ---------------------------------------------------------------------------------------
LEAN_THEOREMS = [
    "validity_nonneg", "condExp_const_of_validity_zero", "condExp_blocked",
    "conditional_validity_blocked", "Chain.fanin_one", "ka_tree_count", "sign_worlds_local_law",
    "local_rule_same", "sign_worlds_differ", "parity_fibre_balanced", "merged_prediction",
    "merged_weights",
]


def check_lean(root="lean"):
    """`lake build` succeeds, every listed theorem exists, none depends on `sorryAx`, and no
    source file contains `sorry`/`admit`/`axiom`."""
    import glob
    import os
    import re
    import subprocess
    env = dict(os.environ)
    elan = env.get("ELAN_HOME", "/coc/flash10/ksingara3/.elan")
    env.update(ELAN_HOME=elan, PATH=os.path.join(elan, "bin") + ":" + env.get("PATH", ""))
    b = subprocess.run(["lake", "build"], cwd=root, env=env, capture_output=True, text=True)
    audit = "import CobwebForest\n" + "".join(f"#print axioms CobwebForest.{t}\n" for t in LEAN_THEOREMS)
    a = subprocess.run(["lake", "env", "lean", "--stdin"], cwd=root, env=env, input=audit,
                       capture_output=True, text=True)
    deps = dict(re.findall(r"'CobwebForest\.(\S+)' depends on axioms: \[(.*?)\]", a.stdout))
    src = "\n".join(open(f, encoding="utf-8").read() for f in glob.glob(f"{root}/CobwebForest/*.lean"))
    banned = re.findall(r"\b(sorry|admit|axiom)\b", re.sub(r"/-.*?-/|--[^\n]*", "", src, flags=re.S))
    ok = (b.returncode == 0 and a.returncode == 0 and set(deps) == set(LEAN_THEOREMS)
          and not any("sorryAx" in v for v in deps.values()) and not banned)
    return dict(label="lean", passed=bool(ok), theorems=len(deps), axioms=deps, banned=banned,
                build_tail=b.stdout[-300:] + b.stderr[-300:], audit_err=a.stderr[-300:])


def check_cues(seed=0, N=3000):
    """Synthetic cue suite with known kinds: x0 associative, (x1, x2) a purely contextual pair
    (Y depends on sign(x1) sign(x2)), x3 irrelevant, x4 a noisy copy of x0 (blocked by it).
    The estimator classifies every cue; the contextual cue's own activation is near-constant
    (empirical side of `condExp_const_of_validity_zero`); the selected forest reads only kept
    cues and beats the single tree over all cues."""
    from cobweb_forest import cues
    r = _rng(seed)

    def draw(n):
        X = r.uniform(-1, 1, size=(n, 5))
        X[:, 4] = X[:, 0] + 0.01 * r.standard_normal(n)
        y = np.sin(np.pi * X[:, 0]) + np.sign(X[:, 1] * X[:, 2]) + 0.05 * r.standard_normal(n)
        return X.astype(F32), y[:, None].astype(F32)
    X, y = draw(N)
    Xv, yv = draw(N // 2)
    rep = cues.estimate(X, y, Xv, yv, names=["assoc", "ctx1", "ctx2", "noise", "copy"])
    want = {0: {"associative", "blocked"}, 1: {"contextual"}, 2: {"contextual"},
            3: {"irrelevant"}, 4: {"blocked", "associative"}}
    kinds_ok = all(rep.kind[i] in w for i, w in want.items()) and \
        ("blocked" in (rep.kind[0], rep.kind[4]))                   # exactly one copy is blocked
    best, F, cols, scores = cues.cue_forest(X, y, Xv, yv, rep)
    ok = (kinds_ok and abs(rep.v(1)) < 0.05 and abs(rep.v(2)) < 0.05 and rep.v(1, 2) > 0.3
          and 3 not in cols and scores[best] < scores["single-all"])
    return dict(label="sec:cues", passed=bool(ok), kinds={rep.names[i]: k for i, k in rep.kind.items()},
                v={",".join(map(str, k)): round(v, 3) for k, v in rep.validity.items()},
                chosen=best, cols=cols, scores=scores)


ALL += [check_lean, check_cues]
