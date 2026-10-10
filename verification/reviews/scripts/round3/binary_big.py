"""Balanced binary hierarchy over G=2^D cell-centre grid points on [0,1], h = spacing.
Level k (depth from root) has 2^k nodes of m=2^(D-k) consecutive points.
All-concept PRIOR and POSTERIOR full expansion (root scored by its likelihood), windowed."""
import numpy as np

def logN(x, mu, var):
    return -0.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)

def run(g, D, xq, W=80):
    G = 2 ** D; delta = 1.0 / G; h2 = delta ** 2
    pts = (np.arange(G) + 0.5) * delta
    cs = np.concatenate([[0.0], np.cumsum(g(pts))])
    res = {}
    out = {'prior': [], 'posterior': [], 'post_leaves': [], 'lik': []}
    for x in xq:
        S = {'prior': [], 'posterior': [], 'lik': []}; V = []
        pi_prev = None; idx_prev = None
        leaves_pi = None; leaves_y = None
        for k in range(D + 1):
            m = 2 ** (D - k)
            c = int(np.floor(x * 2 ** k))
            idx = np.arange(max(0, c - W), min(2 ** k, c + W + 1))
            mu = (idx + 0.5) * m * delta
            s2 = delta ** 2 * (m * m - 1) / 12.0
            ybar = (cs[(idx + 1) * m] - cs[idx * m]) / m
            ll = logN(x, mu, s2 + h2)
            if k == 0:
                pi = np.zeros(1); score_prior = ll; score_post = ll; score_lik = ll
            else:
                # sibling-normalised posterior within each parent (equal counts)
                sib = idx ^ 1
                ll_sib = logN(x, (sib + 0.5) * m * delta, s2 + h2)
                lp = ll - np.logaddexp(ll, ll_sib)
                par = idx >> 1
                pos = np.searchsorted(idx_prev, par)
                ok = (pos < len(idx_prev)) & (idx_prev[np.minimum(pos, len(idx_prev) - 1)] == par)
                pi_par = np.where(ok, pi_prev[np.minimum(pos, len(idx_prev) - 1)], -np.inf)
                pi = pi_par + lp
                score_prior = ll + np.log(m / G)
                score_post = pi
                score_lik = ll
            S['prior'].append(score_prior); S['posterior'].append(score_post); S['lik'].append(score_lik)
            V.append(ybar)
            pi_prev, idx_prev = pi, idx
            if k == D:
                leaves_pi, leaves_y = pi, ybar
        Vall = np.concatenate(V)
        for wt in ['prior', 'posterior', 'lik']:
            s = np.concatenate(S[wt]); w = np.exp(s - s.max())
            out[wt].append(w @ Vall / w.sum() - g(np.array([x]))[0])
        w = np.exp(leaves_pi - leaves_pi.max())
        out['post_leaves'].append(w @ leaves_y / w.sum() - g(np.array([x]))[0])
    return {k: np.sqrt(np.mean(np.array(v) ** 2)) for k, v in out.items()}, \
           {k: np.max(np.abs(np.array(v))) for k, v in out.items()}

g = lambda x: np.sin(np.pi * (2 * x - 1))
rng = np.random.default_rng(1)
xq = rng.uniform(0.1, 0.9, 300)
print(' D        G   PRIOR-all rmse  (x D)   sup   | POSTERIOR-all rmse (x D) | POST leaves-only | LIK-all')
for D in [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]:
    r, s = run(g, D, xq)
    print(f'{D:2d} {2**D:9d}   {r["prior"]:.4f}  ({r["prior"]*D:.3f})  {s["prior"]:.4f} |   {r["posterior"]:.4f} ({r["posterior"]*D:.3f})    |   {r["post_leaves"]:.4f}        | {r["lik"]:.5f}')
