import sys, math, heapq; sys.path.insert(0, '/tmp/referee_k7')
import numpy as np
from cw import Tree, cat_entropy
# 1. multiview: Z~N(0,1), X_v = Z + E_v, Var(E_v)=s2, m views. True posterior precision vs paper's formula
m, s2 = 3, 0.5; P0 = 1.0; Pv = P0 + 1/s2
print("1) true precision", P0 + m/s2, "| paper sum Pv - m P0 =", m*Pv - m*P0, "| sum Pv - (m-1) P0 =", m*Pv - (m-1)*P0)
# 2. CU numerator vs I(Z;C): two perfectly correlated binary attributes, concepts {(0,0)} and {(1,1)}, alpha->0
a = 1e-9; Hp = 2*cat_entropy(2, 2, 2, 2*(1+a)*math.log(1+a), a); Hc = 2*cat_entropy(1, 2, 1, (1+a)*math.log(1+a), a)
print("2) H(parent)-sum P(c)H(c) =", round(Hp - Hc, 6), " vs I(Z;C) = log 2 =", round(math.log(2), 6))
# 3. in-sample activation under N h^d -> infinity: self-weight of leaf i in leaves-only LIK at x_i
rng = np.random.default_rng(0); N, h = 1000, 0.05; X = rng.uniform(0, 1, N); Y = np.sin(6*X) + rng.normal(0, 0.5, N)
W = np.exp(-(X[:, None]-X[None])**2/(2*h*h)); sw = np.diag(W)/W.sum(1); act = (W@Y)/W.sum(1)
print(f"3) N={N}, h={h}: median self-weight {np.median(sw):.4f}; mean |act-y_i| {np.mean(np.abs(act-Y)):.3f} vs mean |act-m(x_i)| {np.mean(np.abs(act-np.sin(6*X))):.3f}")
# 4. leaves-only best-first expansion (LIK), budget M=1 -> cap 64 expansions: can the mixture be empty?
T = Tree(1, alpha=1.0, prior_var=1e-4, seed=1)
for x in rng.uniform(0, 1, 200): T.ifit({}, np.array([x]))
def leaves_reached(q, M=1):
    pq = [(-T.root.loglik({}, q), 0, T.root)]; cnt = 0; leaves = 0; tie = 1
    while pq and cnt < 64*M:
        s, _, n = heapq.heappop(pq); cnt += 1
        if not n.children:
            leaves += 1
            if leaves >= M: break
            continue
        for c in n.children: heapq.heappush(pq, (-c.loglik({}, q), tie, c)); tie += 1
    return leaves, cnt, sum(1 for n in T.nodes() if n.children)
for q in [0.5, 1.2, 3.0]:
    print("4) query", q, "-> (leaves in mixture, expansions, #internal nodes) =", leaves_reached(np.array([q])))
