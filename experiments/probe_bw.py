"""Probe: bandwidth schedule x kappa for single tree and forests (2-D expsin, 4-D nested)."""
import sys, time
import numpy as np
from cobweb_forest import Forest, designs as D

rng = np.random.default_rng(0)
fns = {
    "expsin": (2, lambda X: np.exp(np.sin(np.pi * X[:, 0]) + X[:, 1]**2), (0, 1), [0, 1]),
    "nested4": (4, lambda X: np.exp((np.sin(np.pi * (X[:, 0]**2 + X[:, 1]**2)) + np.sin(np.pi * (X[:, 2]**2 + X[:, 3]**2))) / 2),
                ([0, 1], [2, 3]), None),
}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
for fn, (n, f, graph, _) in fns.items():
    X = rng.uniform(-1, 1, (N, n)).astype(np.float32); Y = f(X)[:, None].astype(np.float32)
    Xt = rng.uniform(-1, 1, (500, n)).astype(np.float32); Yt = f(Xt)
    for kappa in [0.1, 0.25, 0.5, 1.0]:
        kw = dict(max_nodes=300, leaves_only=True, kappa=kappa)
        runs = [("single", D.single(n, N, **kw))]
        for mode in ["act", "hybrid", "label"]:
            runs.append((mode, D.ka(graph, mode, N, **kw)))
        for name, sp in runs:
            t = time.time(); F = Forest(sp, "out").fit(X, Y, protocol="online"); P = F.predict(Xt)[:, 0]
            print(fn, N, "kappa", kappa, name, "rmse %.4f" % np.sqrt(np.mean((P - Yt)**2)), "(std %.3f)" % Yt.std(),
                  "%.0fs" % (time.time() - t), flush=True)
