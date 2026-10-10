import numpy as np
rng = np.random.default_rng(0)

# ---------- 1. tau-drift of a leaf (is_exact_match: var<=tau and |x-mean|<=tau) ----------
tau = 1e-5
for n in [10, 100, 1000, 10000]:
    # adversarial: every new instance sits at current mean + 0.999 tau
    mean, ss, xs = 0.0, 0.0, [0.0]
    cnt = 1
    for k in range(n - 1):
        x = mean + 0.999 * tau
        var = ss / cnt
        assert var <= tau and abs(x - mean) <= tau
        cnt += 1
        d = x - mean; mean += d / cnt; ss += d * (x - mean)
        xs.append(x)
    xs = np.array(xs)
    H = np.sum(1.0 / np.arange(1, n + 1))
    print(f'n={n:6d}: max|x_i - final mean|/tau = {np.max(np.abs(xs - mean))/tau:.3f}, '
          f'final mean drift/tau = {mean/tau:.3f}, H_n = {H:.3f}, var={ss/cnt:.2e}')

# ---------- 2. Thm label: LIK leaves-only output tree over label tuples ----------
def label_forest_error(f, n, K, alpha, nq=4000):
    centres = (np.arange(K) + 0.5) / K
    # all K^n tuples
    grids = np.stack(np.meshgrid(*[np.arange(K)] * n, indexing='ij'), -1).reshape(-1, n)
    leaf_y = f(centres[grids])
    X = rng.uniform(0, 1, size=(nq, n))
    t = np.minimum((X * K).astype(int), K - 1)       # nearest centre = cell index
    # likelihood: matching attr (1+a)/(1+(K+1)a), mismatching a/(1+(K+1)a)
    lm, lx = np.log((1 + alpha) / (1 + (K + 1) * alpha)), np.log(alpha / (1 + (K + 1) * alpha))
    err = []
    for q in range(nq):
        m = (grids == t[q]).sum(1)
        s = m * lm + (n - m) * lx
        w = np.exp(s - s.max()); w /= w.sum()
        err.append(abs(w @ leaf_y - f(X[q][None])[0]))
    err = np.array(err)
    # bound: omega_f(sqrt(n) delta/2) + 2||f|| ((1+K a)^n - 1); estimate omega numerically by Lipschitz
    return err.max()

f = lambda Z: np.sin(3 * Z[:, 0]) * np.cos(2 * Z[:, 1])
for K, alpha in [(5, 0.01), (10, 0.001), (10, 0.05), (20, 1e-4)]:
    e = label_forest_error(f, 2, K, alpha, nq=1500)
    L = np.sqrt(9 + 4)    # Lipschitz bound of f
    bound = L * np.sqrt(2) * (1 / K) / 2 + 2 * 1.0 * ((1 + K * alpha) ** 2 - 1)
    print(f'Thm label: n=2 K={K:3d} alpha={alpha:g}: max err={e:.4f}  bound={bound:.4f}  ok={e <= bound}')

# ---------- 3. Thm labelcons per-cell inequality ----------
viol = 0; worst = 0
for trial in range(3000):
    n = rng.integers(1, 4); Ka = rng.integers(2, 8, size=n); Np = rng.integers(5, 200)
    alpha = 10 ** rng.uniform(-4, 0)
    B = 1.0
    T = np.stack([rng.integers(0, Ka[a], size=Np) for a in range(n)], 1)
    Y = rng.uniform(-B, B, size=Np)
    # repeats: duplicate some instances exactly so some leaves have counts > 1
    dup = rng.random(Np) < 0.3
    Y[dup] = np.round(Y[dup], 1); T[dup] = T[dup]
    # leaves = distinct (t, y)
    keys = {}
    for i in range(Np):
        k = (tuple(T[i]), Y[i]); keys[k] = keys.get(k, 0) + 1
    leaves = list(keys.items())
    # K_a as seen
    Kseen = [len(set(T[:, a])) for a in range(n)]
    tq = T[rng.integers(0, Np)]
    nC = sum(c for (t, y), c in leaves if t == tuple(tq))
    ybarC = sum(c * y for (t, y), c in leaves if t == tuple(tq)) / nC
    s, v = [], []
    for (t, y), c in leaves:
        lp = 0.0
        for a in range(n):
            if t[a] == tq[a]:
                lp += np.log((c + alpha) / (c + (Kseen[a] + 1) * alpha))
            else:
                lp += np.log(alpha / (c + (Kseen[a] + 1) * alpha))
        s.append(np.log(c) + lp); v.append(y)
    s = np.array(s); w = np.exp(s - s.max()); w /= w.sum()
    yhat = w @ np.array(v)
    kappa = np.prod([1 / (1 + (Kseen[a] + 1) * alpha) for a in range(n)])
    bound = 2 * B * (min(1, alpha * Np / (kappa * nC)) + (1 - kappa) / kappa)
    gap = abs(yhat - ybarC)
    if gap > bound + 1e-12: viol += 1
    worst = max(worst, gap / bound if bound > 0 else 0)
print(f'Thm labelcons per-cell bound: violations={viol} of 3000, max gap/bound={worst:.3f}')
