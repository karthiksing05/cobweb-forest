import sys; sys.path.insert(0, '/tmp/referee_k7')
import numpy as np
from cw import Tree
def run(seed, npts, nstream, dim, pv):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 10, size=(npts, dim))
    T = Tree(dim, alpha=1.0, prior_var=pv, seed=seed)
    for i in range(nstream):
        T.ifit({}, pts[rng.integers(npts)])
    L = T.leaves()
    # map each leaf to the distinct instance it holds (leaf mean, zero variance expected)
    keys = [tuple(np.round(l.mean, 6)) for l in L]
    bad_var = sum(np.max(l.ss / l.count) > 1e-9 for l in L)
    # additivity check: internal node count/mean equal pooled children
    add_err = 0.0
    for n in T.nodes():
        if n.children:
            c = sum(ch.count for ch in n.children)
            m = sum(ch.count * ch.mean for ch in n.children) / c
            add_err = max(add_err, abs(c - n.count), float(np.max(np.abs(m - n.mean))))
    return len(L), len(set(keys)), bad_var, add_err
tot = 0
for seed in range(40):
    nl, nd, bv, ae = run(seed, npts=8, nstream=60, dim=1, pv=0.01)
    if nl != nd: tot += 1
    if seed < 8 or nl != nd: print(f"seed {seed}: leaves={nl} distinct-instances-stored={nd} leaves_with_var={bv} additivity_err={ae:.2e}")
print("streams where #leaves > #distinct instances:", tot, "/ 40")
