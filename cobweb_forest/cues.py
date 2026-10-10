"""A cue calculus for Cobweb forests: estimate what each cue contributes, classify it, and
build the forest the classification calls for.

Cognitive science distinguishes kinds of cue -- associative, contextual, retrieval,
self-generated, attentional -- and studies cue validity (Brunswik) and cue competition
(blocking, Kamin; Rescorla-Wagner).  For an outcome V and cue sets A, C define

    validity             v(A)     = Var(V) - E Var(V | A)          (variance V owes to A)
    conditional validity v(A | C) = E Var(V | C) - E Var(V | C, A)  >= 0

and classify a cue A (theory/main.tex, Sec. "Cues"):

    associative   v(A) > 0                      -> its funnel activation carries it
    contextual    v(A) = 0 but v(A | C) > 0     -> invisible alone; its activation is constant,
                                                   so it must be merged with C or passed as a label
    blocked       v(A) > 0 but v(A | C) = 0     -> redundant given a kept, more valid cue C; drop it
    irrelevant    v(A | C) = 0 for every C tried

Every expectation is estimated the Cobweb way: a tree over (cues, outcome) learned with
``ifit`` and scored on held-out data, so E Var(V | A) is the held-out MSE of the tree with cue
set A (an upper estimate, tight when the tree is consistent).  ``cue_forest`` turns the
classification into a forest and verifies it, together with the single tree, at the output
(theory: selection by verification).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import designs as D
from .forest import Forest

KW = dict(max_nodes=300, leaves_only=True)


def _mse(F, X, y):
    return float(np.mean((F.predict(X)[:, 0] - y[:, 0]) ** 2))


@dataclass
class CueReport:
    """Estimated validities (as fractions of Var V) and the resulting classification."""
    names: List[str]
    var: float
    validity: Dict[Tuple[int, ...], float] = field(default_factory=dict)       # v(S) for |S| <= 2
    funnel_validity: Dict[Tuple[int, int], float] = field(default_factory=dict)  # pair via activations
    kind: Dict[int, str] = field(default_factory=dict)
    blocked_by: Dict[int, int] = field(default_factory=dict)
    contextual_pairs: List[Tuple[int, int]] = field(default_factory=list)

    def v(self, *S):
        return self.validity.get(tuple(sorted(S)), 0.0)

    def conditional(self, a, c):
        """v(a | c) = v({a, c}) - v({c})."""
        return self.v(a, c) - self.v(c)

    def synergy(self, i, j):
        """Validity of the pair lost by passing activations (the synergy deficit)."""
        return self.v(i, j) - self.funnel_validity.get(tuple(sorted((i, j))), 0.0)

    def table(self):
        rows = []
        for i, nm in enumerate(self.names):
            rows.append(f"{nm:>10s}  v={self.v(i):+.3f}  kind={self.kind.get(i, '?')}"
                        + (f"  blocked by {self.names[self.blocked_by[i]]}" if i in self.blocked_by else ""))
        return "\n".join(rows)


def estimate(X, y, Xv, yv, names: Sequence[str] = None, kappa: float = 0.25,
             tau: float = 0.02) -> CueReport:
    """Estimate single and pairwise validities with Cobweb trees and classify every cue.

    tau is the threshold, as a fraction of Var(V), below which a validity counts as zero."""
    n = X.shape[1]
    N = len(X)
    names = list(names) if names is not None else [f"x{i}" for i in range(n)]
    var = float(np.var(yv))
    rep = CueReport(names=names, var=var)
    base = float(np.mean((yv[:, 0] - y.mean()) ** 2))
    for i in range(n):
        F = Forest(D.single(1, N, kappa=kappa, **KW), "out").fit(X[:, [i]], y, protocol="online")
        rep.validity[(i,)] = (base - _mse(F, Xv[:, [i]], yv)) / var
    for i, j in combinations(range(n), 2):
        Xi, Xvi = X[:, [i, j]], Xv[:, [i, j]]
        F = Forest(D.single(2, N, kappa=kappa, **KW), "out").fit(Xi, y, protocol="online")
        rep.validity[(i, j)] = (base - _mse(F, Xvi, yv)) / var
        A = Forest(D.ka((0, 1), "act", N, kappa=kappa, **KW), "out").fit(Xi, y, protocol="online")
        rep.funnel_validity[(i, j)] = (base - _mse(A, Xvi, yv)) / var
    # classification.  Blocking is competitive (Kamin): visit cues from the most valid down;
    # a cue is blocked only by a cue already kept, so of two redundant copies one survives.
    kept = []
    for i in sorted(range(n), key=rep.v, reverse=True):
        cond = {j: rep.conditional(i, j) for j in range(n) if j != i}
        if rep.v(i) > tau:
            blockers = [j for j in kept if cond[j] <= tau]
            if blockers:
                rep.kind[i] = "blocked"
                rep.blocked_by[i] = max(blockers, key=rep.v)
            else:
                rep.kind[i] = "associative"
                kept.append(i)
        elif any(c > tau for c in cond.values()):
            rep.kind[i] = "contextual"
        else:
            rep.kind[i] = "irrelevant"
    for i, j in combinations(range(n), 2):
        if rep.synergy(i, j) > tau and "blocked" not in (rep.kind[i], rep.kind[j]) \
                and "irrelevant" not in (rep.kind[i], rep.kind[j]):
            rep.contextual_pairs.append((i, j))
    return rep


def groups(rep: CueReport) -> List[List[int]]:
    """Kept cues grouped so that contextual pairs share a tree (union-find)."""
    keep = [i for i, k in rep.kind.items() if k in ("associative", "contextual")]
    parent = {i: i for i in keep}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j in rep.contextual_pairs:
        if i in parent and j in parent:
            parent[find(i)] = find(j)
    out = {}
    for i in keep:
        out.setdefault(find(i), []).append(i)
    return sorted(out.values())


def cue_forest(X, y, Xv, yv, rep: CueReport, kappa: float = 0.25):
    """Build the forest the classification calls for, and verify it at the output against the
    single tree over the kept cues and over all cues.  Returns (chosen name, forest, cue
    indices it reads, validation MSEs of every candidate)."""
    N = len(X)
    G = groups(rep)
    keep = sorted(i for g in G for i in g)
    cands = {}
    if keep:
        Xk, Xvk = X[:, keep], Xv[:, keep]
        local = [[keep.index(i) for i in g] for g in G]
        if len(local) > 1:
            graph = D.balanced([g for g in local])
            for mode in ("act", "hybrid"):
                cands[f"cue-forest-{mode}"] = (D.ka(graph, mode, N, kappa=kappa, label_depth=3, **KW), keep)
        cands["single-kept"] = (D.single(len(keep), N, kappa=kappa, **KW), keep)
    cands["single-all"] = (D.single(X.shape[1], N, kappa=kappa, **KW), list(range(X.shape[1])))
    scores, fitted = {}, {}
    for name, (specs, cols) in cands.items():
        F = Forest(specs, "out").fit(X[:, cols], y, protocol="online")
        scores[name] = _mse(F, Xv[:, cols], yv)
        fitted[name] = (F, cols)
    best = min(scores, key=scores.get)
    return best, fitted[best][0], fitted[best][1], scores
