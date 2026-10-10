"""Probe: 4-D nested KAN function, single tree vs forests, training protocols."""
import sys, time
import numpy as np
from cobweb_forest import Forest, designs as D

rng = np.random.default_rng(0)
f = lambda X: np.exp((np.sin(np.pi * (X[:, 0]**2 + X[:, 1]**2)) + np.sin(np.pi * (X[:, 2]**2 + X[:, 3]**2))) / 2)
Xt = rng.uniform(-1, 1, (500, 4)).astype(np.float32); Yt = f(Xt)
kw = dict(max_nodes=200, leaves_only=True)
for N in [1000, 4000, 16000]:
    X = rng.uniform(-1, 1, (N, 4)).astype(np.float32); Y = f(X)[:, None].astype(np.float32)
    runs = [("single", "online", D.single(4, N, **kw))]
    for proto in ["online", "split"]:
        for mode in ["act", "hybrid", "label"]:
            runs.append((f"{mode}-pairs", proto, D.ka(([0, 1], [2, 3]), mode, N, **kw)))
        runs.append(("act-uni", proto, D.ka(((0, 1), (2, 3)), "act", N, **kw)))
    for name, proto, sp in runs:
        t = time.time(); F = Forest(sp, "out").fit(X, Y, protocol=proto); P = F.predict(Xt)[:, 0]
        print(N, name, proto, "rmse %.4f" % np.sqrt(np.mean((P - Yt)**2)), "(std %.3f)" % Yt.std(),
              "%.0fs" % (time.time() - t), flush=True)
