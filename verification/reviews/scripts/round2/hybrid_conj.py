# Idealised hybrid forest: two univariate trees on X1, X2 ~ U[-1,1] pass (label, funnel activation) to an
# output tree that models (label1, label2, a1, a2, Y) with count-weighted leaves-only mixture, full expansion.
# Y = f(X) = X1^2 + X1*X2.  Ideal funnel activations: a1 = E[Y|X1] = X1^2, a2 = E[Y|X2] = 1/3 (constant).
# Label cells: K equal intervals per coordinate, K^2 = o(N), alpha*M_N -> 0  (conditions (a)-(c) of thm:labelcons).
# Output-tree weight of stored point i for query x: alpha^{#label disagreements} * exp(-|a(x)-a_i|^2 / 2h^2).
import numpy as np
rng = np.random.default_rng(0)
f = lambda x1, x2: x1**2 + x1*x2
def run(N, h, ntest=2000):
    K = int(N ** 0.3); alpha = 1.0 / (K * K * np.log(N))
    X1, X2 = rng.uniform(-1, 1, N), rng.uniform(-1, 1, N); Y = f(X1, X2)
    L1 = np.minimum((X1 + 1) / 2 * K, K - 1).astype(int); L2 = np.minimum((X2 + 1) / 2 * K, K - 1).astype(int)
    A1 = X1 ** 2
    t1, t2 = rng.uniform(-1, 1, ntest), rng.uniform(-1, 1, ntest)
    err = []
    for x1, x2 in zip(t1, t2):
        l1 = min(int((x1 + 1) / 2 * K), K - 1); l2 = min(int((x2 + 1) / 2 * K), K - 1)
        logw = (np.log(alpha) * ((L1 != l1).astype(float) + (L2 != l2).astype(float))
                - (A1 - x1 ** 2) ** 2 / (2 * h * h))           # a2 is constant: its kernel factor is 1
        w = np.exp(logw - logw.max())
        err.append((w @ Y / w.sum() - f(x1, x2)) ** 2)
    return np.sqrt(np.mean(err)) / np.std(f(t1, t2))
for N in [2000, 8000, 32000]:
    print(f"N={N:6d}  RMSE/sd with h=N^-1/4: {run(N, N ** -0.25):.3f}    with h=N^-3: {run(N, float(N) ** -3):.3f}")
