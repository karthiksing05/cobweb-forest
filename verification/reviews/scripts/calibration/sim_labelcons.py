import sys; sys.path.insert(0, '/tmp/referee_k7')
import numpy as np, math
from cw import Tree
def run(N, K, alpha, binary, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, N); lab = np.minimum((X * K).astype(int), K - 1)
    m = 0.2 + 0.6 * X
    Y = (rng.uniform(size=N) < m).astype(float) if binary else m + rng.uniform(-0.2, 0.2, N)
    T = Tree(1, alpha=alpha, prior_var=0.01, seed=seed)
    for i in range(N): T.ifit({0: {int(lab[i]): 1.0}}, [Y[i]])
    L = T.leaves(); out = []
    for t in range(K):
        s = np.array([l.loglik({0: {t: 1.0}}, np.array([np.nan])) for l in L]); w = np.exp(s - s.max())
        pred = float(np.sum(w * np.array([l.mean[0] for l in L])) / w.sum())
        part = Y[lab == t].mean(); truth = 0.2 + 0.6 * (t + 0.5) / K
        nleaf = sum(1 for l in L if 0 in l.av and t in l.av[0])
        out.append((t, round(pred, 3), round(part, 3), round(truth, 3), int((lab == t).sum()), nleaf))
    return out
for binary in [True, False]:
    print("binary Y" if binary else "continuous Y", "(N=1500, K=5, alpha=1e-4): (cell, tree prediction, partitioning estimate, truth, #points, #leaves)")
    for r in run(1500, 5, 1e-4, binary): print("  ", r)
