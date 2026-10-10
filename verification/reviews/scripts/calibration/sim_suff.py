import numpy as np
rng = np.random.default_rng(1); N = 2_000_000
X1, X2 = rng.uniform(-1, 1, N), rng.uniform(-1, 1, N)
Y = (X1 + X2) ** 2                      # Phi(t)=t^2 continuous, g_j = identity, independent blocks, xi = 0
m1 = X1**2 + 1/3; m2 = X2**2 + 1/3      # E[Y|X1], E[Y|X2] (exact)
# F_A = E[Y | m1, m2] = E[Y | |X1|, |X2|]: average over the 4 sign patterns
FA = X1**2 + X2**2 + 2*np.abs(X1)*np.abs(X2)*np.mean([1*1, 1*-1, -1*1, -1*-1])
print("Delta = E[(f - F_A)^2] =", np.mean((Y - FA)**2), " (exact 4/9 =", 4/9, ")  Var(f) =", Y.var())
# empirical check of E[Y|m1,m2] by binning |X1|,|X2|
b1 = np.minimum((np.abs(X1)*20).astype(int), 19); b2 = np.minimum((np.abs(X2)*20).astype(int), 19)
idx = b1*20 + b2; s = np.bincount(idx, Y, 400) / np.bincount(idx, None, 400)
print("binned E[Y|m1,m2] residual MSE:", np.mean((Y - s[idx])**2))
