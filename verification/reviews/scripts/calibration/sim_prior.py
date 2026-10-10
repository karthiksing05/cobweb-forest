import numpy as np
x = 0.3; d = 1
for beta in [1.0, 2.0, 3.0]:
    rows = []
    for K in [50, 51, 100, 101, 200, 201, 400, 401]:
        delta = 1.0 / K; h = delta / beta
        g = (np.arange(K) + 0.5) * delta
        k = np.exp(-(x - g)**2 / (2*h*h)) / np.sqrt(2*np.pi*h*h)
        leaves = k.sum() / K                                   # prior 1/G times likelihood
        vroot = g.var() + h*h
        Nroot = np.exp(-(x - g.mean())**2 / (2*vroot)) / np.sqrt(2*np.pi*vroot)
        lam = Nroot / (Nroot + leaves)
        rows.append(f"h={h:.5f}: leafmass={leaves:.4f} lambda={lam:.4f}")
    print(f"beta={beta}; claimed limit lambda={Nroot/(1+Nroot):.4f}")
    print("   " + "\n   ".join(rows))
