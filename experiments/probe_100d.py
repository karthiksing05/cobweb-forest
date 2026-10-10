"""Probe: the 100-D KAN function exp(mean_i sin^2(pi x_i / 2)), single tree vs activation forest."""
import sys, time
import numpy as np
from cobweb_forest import Forest, designs as D

rng = np.random.default_rng(0)
n = 100
f = lambda X: np.exp(np.mean(np.sin(np.pi * X / 2) ** 2, 1))
N = int(sys.argv[1])
X = rng.uniform(-1, 1, (N, n)).astype(np.float32); Y = f(X)[:, None].astype(np.float32)
Xt = rng.uniform(-1, 1, (300, n)).astype(np.float32); Yt = f(Xt)
print("std", Yt.std(), flush=True)
for kappa in [0.3, 1.0]:
    for name, sp in [("single", D.single(n, N, kappa=kappa, max_nodes=300, leaves_only=True)),
                     ("act-balanced", D.ka(D.balanced(range(n)), "act", N, kappa=kappa, max_nodes=100, leaves_only=True))]:
        for proto in (["online", "split"] if name != "single" else ["online"]):
            t = time.time(); F = Forest(sp, "out").fit(X, Y, protocol=proto); P = F.predict(Xt)[:, 0]
            print(N, kappa, name, proto, "rmse %.4f" % np.sqrt(np.mean((P - Yt) ** 2)), "%.0fs" % (time.time() - t), flush=True)
