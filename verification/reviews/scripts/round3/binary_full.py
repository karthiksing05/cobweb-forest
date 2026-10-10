"""Full (non-windowed) level-vectorised computation on the balanced binary hierarchy,
for PRIOR and POSTERIOR all-concept and POSTERIOR leaves-only.  Queries in chunks."""
import numpy as np

def logN(x, mu, var):
    return -0.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)

def run(g, D, xq):
    G = 2 ** D; delta = 1.0 / G; h2 = delta ** 2
    pts = (np.arange(G) + 0.5) * delta
    cs = np.concatenate([[0.0], np.cumsum(g(pts))])
    X = xq[:, None]
    gx = g(xq)
    # running log-sum-exp accumulators for numerator/denominator per weighting
    acc = {w: [np.full(len(xq), -np.inf), np.zeros(len(xq))] for w in ['prior', 'posterior']}
    def add(w, s, y):   # s: (Q, n) scores, y: (n,) values
        M, num = acc[w]
        m = np.maximum(M, s.max(1))
        num = num * np.exp(M - m) + (np.exp(s - m[:, None]) * y[None, :]).sum(1)
        den_key = w + '_den'
        acc[w] = [m, num]
        acc.setdefault(den_key, [np.full(len(xq), -np.inf), np.zeros(len(xq))])
        Md, den = acc[den_key]
        md = np.maximum(Md, s.max(1))
        den = den * np.exp(Md - md) + np.exp(s - md[:, None]).sum(1)
        acc[den_key] = [md, den]
    pi_prev = None
    for k in range(D + 1):
        m_ = 2 ** (D - k)
        idx = np.arange(2 ** k)
        mu = (idx + 0.5) * m_ * delta
        s2 = delta ** 2 * (m_ * m_ - 1) / 12.0
        ybar = (cs[(idx + 1) * m_] - cs[idx * m_]) / m_
        ll = logN(X, mu[None, :], s2 + h2)
        if k == 0:
            pi = np.zeros((len(xq), 1))
            add('prior', ll, ybar); add('posterior', ll, ybar)
        else:
            sib = idx ^ 1
            lp = ll - np.logaddexp(ll, ll[:, sib])
            pi = pi_prev[:, idx >> 1] + lp
            add('prior', ll + np.log(m_ / G), ybar)
            add('posterior', pi, ybar)
        pi_prev = pi
        if k == D:
            w = np.exp(pi - pi.max(1, keepdims=True))
            leaves = (w * ybar[None, :]).sum(1) / w.sum(1)
    out = {}
    for w in ['prior', 'posterior']:
        M, num = acc[w]; Md, den = acc[w + '_den']
        pred = num / den * np.exp(M - Md)
        out[w] = np.sqrt(np.mean((pred - gx) ** 2))
    out['post_leaves'] = np.sqrt(np.mean((leaves - gx) ** 2))
    return out

g = lambda x: np.sin(np.pi * (2 * x - 1))
rng = np.random.default_rng(1)
xq_all = rng.uniform(0.1, 0.9, 200)
print(' D       G   PRIOR-all  (xD)   POSTERIOR-all (xD)   POSTERIOR leaves-only')
for D in [4, 6, 8, 10, 12, 14, 16, 18]:
    chunks = np.array_split(xq_all, max(1, 2 ** D // 4096))
    parts = [run(g, D, c) for c in chunks]
    # combine RMSE over chunks
    out = {k: np.sqrt(np.mean([p[k] ** 2 for p in parts])) for k in parts[0]}
    print(f'{D:2d} {2**D:8d}   {out["prior"]:.4f} ({out["prior"]*D:.3f})   {out["posterior"]:.4f} ({out["posterior"]*D:.3f})      {out["post_leaves"]:.4f}')
