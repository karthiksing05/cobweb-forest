import sys; sys.argv=['x']
exec(open('/tmp/referee_m5/bnb.py').read().split("found = 0")[0])   # reuse Node/loglik/all_scores/categorize
import math
def leaf_with_copies(inst, k):
    n = Node(inst=inst)
    for _ in range(k - 1):
        for a, v in inst.items(): n.av[a][v] += 1
        n.count += 1
    return n
for a_ in [0.1, 0.01]:
    alpha = a_
    globals()['alpha'] = a_
    curr = leaf_with_copies({0: 0}, 1)           # one instance with a=0
    d    = leaf_with_copies({0: 0}, 9)           # nine copies of a=0
    e    = leaf_with_copies({0: 1}, 76)          # 76 copies of a=1
    q    = Node(children=[d, e])                 # big, impure subtree holding the pure child d
    root = Node(children=[curr, q])
    attr_vals = {0: {0, 1}}
    query = {0: 0}
    sc = all_scores(root, query, attr_vals)
    names = {id(root): 'root', id(curr): 'curr', id(q): 'q', id(d): 'd', id(e): 'e'}
    for s, n in sc: print(f"alpha={alpha}: {names[id(n)]:5s} PATH score {s:8.4f}  loglik {loglik(n, query, attr_vals):8.4f}")
    b, bs = categorize(root, query, attr_vals)
    best = max(sc, key=lambda t: t[0])
    print(f"alpha={alpha}: categorize(use_best=True) returns {names[id(b)]} (score {bs:.4f}); true best is {names[id(best[1])]} (score {best[0]:.4f})\n")
