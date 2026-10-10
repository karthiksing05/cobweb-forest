import sys; sys.path.insert(0, '/tmp/referee_k7')
import numpy as np, cw
from cw import Tree
centres = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
def run(N, tol, h, seed=0):
    rng = np.random.default_rng(seed)
    X = centres[rng.integers(5, size=N)] + 1e-6 * rng.uniform(size=N)    # X has a density (uniform on 5 intervals of width 1e-6)
    Y = (rng.uniform(size=N) < 0.9).astype(float)                         # m(x) = 0.9 everywhere
    T = Tree(2, alpha=1.0, prior_var=h*h, seed=seed)
    orig = cw.Node.exact
    cw.Node.exact = lambda self, d, c, tol=tol: orig(self, d, c, tol)
    for i in range(N): T.ifit({}, np.array([X[i], Y[i]]))
    cw.Node.exact = orig
    L = T.leaves(); preds = []
    for c in centres:
        s = np.array([l.loglik({}, np.array([c + 5e-7, np.nan])) for l in L]); w = np.exp(s - s.max())
        preds.append(float(np.sum(w * np.array([l.mean[1] for l in L])) / w.sum()))
    return len(L), np.round(preds, 3)
for N in [200, 800]:
    for tol, name in [(1e-5, "implementation tol=1e-5"), (0.0, "exact matching")]:
        nl, p = run(N, tol, h=0.05)
        print(f"N={N} {name:24s}: leaves={nl:4d}  predictions at the 5 intervals {p}  (target 0.9)")
