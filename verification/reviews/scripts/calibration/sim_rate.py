import numpy as np
rng = np.random.default_rng(0)
def nw_err(N, h, n, reps=20):
    e = []
    for r in range(reps):
        X = rng.uniform(0, 1, (N, n)); Y = np.sin(2*np.pi*X).sum(1)          # noise-free C^infinity target
        Q = rng.uniform(0.3, 0.7, (300, n)); f = np.sin(2*np.pi*Q).sum(1)    # interior queries
        D2 = ((Q[:, None, :] - X[None, :, :])**2).sum(-1)
        W = np.exp(-(D2 - D2.min(1, keepdims=True)) / (2*h*h))
        e.append(np.sqrt(np.mean(((W @ Y) / W.sum(1) - f)**2)))
    return np.mean(e)
for n in [1, 2]:
    Ns = [250, 500, 1000, 2000, 4000]; best = []
    for N in Ns:
        hs = np.geomspace(0.1 * N**(-1/n), 0.3, 25)
        errs = [nw_err(N, h, n, reps=8) for h in hs]
        best.append(min(errs))
    slope = np.polyfit(np.log(Ns), np.log(best), 1)[0]
    print(f"n={n}: best RMSE {['%.2e'%b for b in best]}  fitted exponent {slope:.3f}  (paper N^-{2/n:.2f}, NW random-design theory N^-{2/(n+2):.2f})")
