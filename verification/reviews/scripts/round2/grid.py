# Depth-one grid tree of thm:single / prop:prior, d = 1, scores exactly as in CobwebHybridTree::expand.
import numpy as np
g = lambda x: np.sin(3 * np.pi * x)
beta = 1.0
xs = np.linspace(0.003, 0.997, 997)
def npdf(x, m, v): return np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)
for h in [0.02, 0.01, 0.005, 0.0025, 0.00125]:
    delta = beta * h; G = int(round(1 / delta)); gam = (np.arange(G) + 0.5) / G
    yg = g(gam); vroot = gam.var() + h * h; mroot = gam.mean(); yroot = yg.mean()
    res = {}
    for x in [xs]:
        K = npdf(x[:, None], gam[None, :], h * h)            # leaf likelihoods k_gamma(x)
        Nr = npdf(x, mroot, vroot)                           # root likelihood (x coordinate only)
        post = K / K.sum(1, keepdims=True)                   # sibling-normalised posteriors (equal counts)
        W = {'lik': (Nr, K), 'path': (Nr, post * K), 'prior': (Nr, K / G), 'posterior': (Nr, post)}
        for name, (wr, wl) in W.items():
            tot = wr + wl.sum(1); lam = wr / tot
            yhat = (wr * yroot + (wl * yg).sum(1)) / tot
            yleaf = (wl * yg).sum(1) / wl.sum(1)
            res[name] = (np.abs(yhat - g(x)).max(), lam.max(), lam.min(), np.abs(yleaf - g(x)).max())
    print(f"h={h:<8} G={G:<4} " + "  ".join(f"{k}: err={v[0]:.4f} lam[{v[2]:.3f},{v[1]:.3f}] leaves-only err={v[3]:.4f}" for k, v in res.items()))
