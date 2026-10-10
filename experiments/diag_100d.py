"""Diagnose the 100-D activation forest level by level: error of every tree's activation
against its exact conditional expectation m_S(x) = exp(s_S/100) * c^(100-|S|)."""
import sys
import numpy as np
from cobweb_forest import Forest, designs as D

n = 100
f = lambda X: np.exp(np.mean(np.sin(np.pi * X / 2) ** 2, 1))
u = np.sin(np.pi * np.random.default_rng(9).uniform(-1, 1, 10 ** 6) / 2) ** 2
c = np.mean(np.exp(u / 100))
N = int(sys.argv[1]); kappa = float(sys.argv[2]); M = int(sys.argv[3]) if len(sys.argv) > 3 else 100
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (N, n)).astype(np.float32); Y = f(X)[:, None].astype(np.float32)
Xt = rng.uniform(-1, 1, (300, n)).astype(np.float32); Yt = f(Xt)
specs = D.ka(D.balanced(range(n)), "act", N, kappa=kappa, max_nodes=M, leaves_only=True)
F = Forest(specs, "out").fit(X, Y, protocol="online")
# raw coordinates upstream of every tree
up = {}
for s in specs:
    up[s.name] = set(s.raw) | set().union(*[up[p] for p in s.act_from]) if s.act_from else set(s.raw)
M = [F.messages(x) for x in Xt]
P = F.predict(Xt)[:, 0]
lv = {s.name: F.level(s.name) for s in specs}
for L in sorted(set(lv.values())):
    errs, sig = [], []
    for s in specs:
        if lv[s.name] != L or s.name == "out":
            continue
        S = sorted(up[s.name])
        m = np.exp(np.sum(np.sin(np.pi * Xt[:, S] / 2) ** 2, 1) / 100) * c ** (n - len(S))
        a = np.array([mm[s.name]["act"][0] for mm in M])
        errs.append(np.sqrt(np.mean((a - m) ** 2))); sig.append(np.std(m))
    if errs:
        print(f"level {L}: trees {len(errs)}  mean |S|={len(S)}  act RMSE {np.mean(errs):.4f}  sd(m_S) {np.mean(sig):.4f}  ratio {np.mean(errs)/np.mean(sig):.2f}", flush=True)
print("output RMSE/sd", np.sqrt(np.mean((P - Yt) ** 2)) / Yt.std())
