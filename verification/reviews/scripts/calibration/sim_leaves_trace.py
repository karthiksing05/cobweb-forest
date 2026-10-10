import sys; sys.path.insert(0, '/tmp/referee_k7')
import numpy as np, cw
from cw import Tree
rng = np.random.default_rng(0); pts = rng.uniform(0, 10, size=(8, 1))
T = Tree(1, alpha=1.0, prior_var=0.01, seed=0)
ops = []; orig = cw.Node.best_op
def rec(self, *a):
    op = orig(self, *a); ops.append((op, len(self.children), self.count)); return op
cw.Node.best_op = rec
for i in range(60):
    x = pts[rng.integers(8)]; before = {round(float(l.mean[0]), 6) for l in T.leaves()}
    ops.clear(); nl0 = len(T.leaves()); T.ifit({}, x); nl1 = len(T.leaves())
    if nl1 > nl0 and round(float(x[0]), 6) in before:
        print(f"step {i}: instance {x[0]:.6f} already stored in a leaf, but a NEW leaf was created; operator path (op, #children, node count) = {ops}")
