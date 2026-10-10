import sys; sys.path.insert(0, '/tmp/referee_k7')
import numpy as np
from collections import Counter
from cw import Tree
rng = np.random.default_rng(0)
pts = rng.uniform(0, 10, size=(8, 1))
T = Tree(1, alpha=1.0, prior_var=0.01, seed=0)
for i in range(60): T.ifit({}, pts[rng.integers(8)])
c = Counter(round(float(l.mean[0]), 6) for l in T.leaves())
print("instance held by >1 leaf:", {k: v for k, v in c.items() if v > 1})
print("leaf counts for it:", [l.count for l in T.leaves() if round(float(l.mean[0]),6) in [k for k,v in c.items() if v>1]])
for dim, npts, ns, pv in [(1, 20, 300, 0.01), (2, 20, 300, 0.01), (2, 50, 500, 0.1), (3, 30, 400, 1.0)]:
    bad = 0; extra = 0
    for seed in range(20):
        r = np.random.default_rng(seed); P = r.uniform(0, 10, size=(npts, dim))
        T = Tree(dim, alpha=1.0, prior_var=pv, seed=seed)
        for i in range(ns): T.ifit({}, P[r.integers(npts)])
        L = T.leaves(); d = len(set(tuple(np.round(l.mean, 6)) for l in L))
        bad += len(L) > d; extra += len(L) - d
    print(f"dim={dim} npts={npts} stream={ns} prior_var={pv}: streams violating bijection {bad}/20, extra leaves total {extra}")
