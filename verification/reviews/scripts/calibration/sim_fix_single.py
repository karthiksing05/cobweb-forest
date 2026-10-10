import sys; sys.path.insert(0, '/tmp/referee_k7')
import numpy as np
from cw import Tree
# (1) PRIOR leaves-only (score = loglik + log n_c) on the binary label example
rng = np.random.default_rng(0); N, K = 1500, 5
X = rng.uniform(0, 1, N); lab = np.minimum((X*K).astype(int), K-1); Y = (rng.uniform(size=N) < 0.2 + 0.6*X).astype(float)
T = Tree(1, alpha=1e-4, prior_var=0.01, seed=0)
for i in range(N): T.ifit({0: {int(lab[i]): 1.0}}, [Y[i]])
L = T.leaves()
for t in range(K):
    s = np.array([l.loglik({0: {t: 1.0}}, np.array([np.nan])) + np.log(l.count) for l in L]); w = np.exp(s - s.max())
    print(f"cell {t}: PRIOR-leaves prediction {np.sum(w*np.array([l.mean[0] for l in L]))/w.sum():.3f}  partitioning {Y[lab==t].mean():.3f}")
# (2) thm:single grid tree, d=1, beta=1: sup error for LIK / PATH (full expansion, root by likelihood)
g = lambda x: np.abs(x - 0.5) + 0.3*np.sin(7*x)
for G in [20, 40, 80, 160, 320]:
    h = 1.0/G; gam = (np.arange(G)+0.5)/G; xs = np.linspace(0, 1, 2001)
    k = np.exp(-(xs[:, None]-gam[None])**2/(2*h*h))/np.sqrt(2*np.pi*h*h)
    vr = gam.var() + h*h; Nr = np.exp(-(xs-gam.mean())**2/(2*vr))/np.sqrt(2*np.pi*vr); muY = g(gam).mean()
    for name, mass in [("LIK", k), ("PATH", k**2/k.sum(1, keepdims=True))]:
        yh = (Nr*muY + mass @ g(gam)) / (Nr + mass.sum(1))
        e = np.max(np.abs(yh - g(xs)))
        print(f"G={G:4d} {name:4s}: sup error {e:.4f}  error*G {e*G:.3f}  root weight max {np.max(Nr/(Nr+mass.sum(1))):.2e}")
