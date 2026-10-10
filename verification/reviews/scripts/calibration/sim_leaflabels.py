import sys; sys.path.insert(0, '/tmp/referee_k7')
import numpy as np, math
from cw import Tree
def get_leaf(T, x):
    cur = T.root
    while cur.children:
        sc = [math.log(c.count / T.root.count) + c.loglik({}, x) for c in cur.children]
        cur = cur.children[int(np.argmax(sc))]
    return id(cur)
for N in [100, 200, 400, 800]:
    rng = np.random.default_rng(N)
    T = Tree(1, alpha=1.0, prior_var=1e-4, seed=N)
    for x in rng.uniform(0, 1, N): T.ifit({}, np.array([x]))           # chunk 1: label tree
    occ = len({get_leaf(T, np.array([x])) for x in rng.uniform(0, 1, N)})   # chunk 2: output chunk
    print(f"chunk size {N}: label-tree leaves {len(T.leaves())}, occupied leaf-label cells in output chunk {occ} ({occ/N:.2f} N)")
