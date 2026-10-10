# Faithful re-implementation of CobwebHybridTree::expand + categorize(use_best=True, PATH)
# for purely NOMINAL trees, to test whether the global branch-and-bound stop is exact
# when every log-likelihood is <= 0.
import numpy as np, heapq, itertools
from math import log, exp
rng = np.random.default_rng(1)
alpha = 0.1

class Node:
    def __init__(self, children=None, inst=None):
        self.children = children or []
        self.av = {}   # attr -> {val: count}
        self.count = 0
        if inst is not None:
            self.count = 1
            for a, v in inst.items():
                self.av.setdefault(a, {}); self.av[a][v] = self.av[a].get(v, 0) + 1
        for c in self.children:
            self.count += c.count
            for a, d in c.av.items():
                self.av.setdefault(a, {})
                for v, n in d.items():
                    self.av[a][v] = self.av[a].get(v, 0) + n

def loglik(node, q, attr_vals):
    lp = 0.0
    for a, v in q.items():
        K = len(attr_vals[a])
        acount = (K + 1) * alpha + sum(node.av.get(a, {}).values())
        av = alpha + (node.av.get(a, {}).get(v, 0) if v in attr_vals[a] else 0)
        lp += log(av) - log(acount)
    return lp

def lse(xs):
    m = max(xs); return m + log(sum(exp(x - m) for x in xs))

def all_scores(root, q, attr_vals):
    out = [(loglik(root, q, attr_vals), root)]
    stack = [(root, 0.0)]
    while stack:
        cur, pll = stack.pop()
        if not cur.children: continue
        ll = [loglik(c, q, attr_vals) for c in cur.children]
        un = [l + log(c.count / cur.count) for l, c in zip(ll, cur.children)]
        Z = lse(un)
        for c, l, u in zip(cur.children, ll, un):
            cp = u - Z + pll
            out.append((cp + l, c)); stack.append((c, cp))
    return out

def categorize(root, q, attr_vals):
    tie = itertools.count()
    heap = [(-loglik(root, q, attr_vals), -0.0, next(tie), root)]
    best, best_score = root, -float('inf')
    while heap:
        ns, npll, _, cur = heapq.heappop(heap)
        score, pll = -ns, -npll
        if score > best_score: best, best_score = cur, score
        if best_score > pll: break          # the global stop of categorize(use_best=True)
        if not cur.children: continue
        ll = [loglik(c, q, attr_vals) for c in cur.children]
        un = [l + log(c.count / cur.count) for l, c in zip(ll, cur.children)]
        Z = lse(un)
        for c, l, u in zip(cur.children, ll, un):
            cp = u - Z + pll
            heapq.heappush(heap, (-(cp + l), -cp, next(tie), c))
    return best, best_score

def random_tree(insts):
    nodes = [Node(inst=x) for x in insts]
    while len(nodes) > 1:                    # random agglomeration into a random hierarchy
        k = int(rng.integers(2, min(4, len(nodes)) + 1))
        idx = sorted(rng.choice(len(nodes), size=k, replace=False), reverse=True)
        grp = [nodes.pop(i) for i in idx]
        nodes.append(Node(children=grp))
    return nodes[0]

found = 0
for trial in range(20000):
    nA, nV = 3, 3
    ninst = int(rng.integers(3, 12))
    insts = [{a: int(rng.integers(0, nV)) for a in range(nA)} for _ in range(ninst)]
    attr_vals = {a: set(x[a] for x in insts) for a in range(nA)}
    root = random_tree(insts)
    q = {a: int(rng.integers(0, nV)) for a in range(nA)}
    allsc = all_scores(root, q, attr_vals)
    assert all(loglik(n, q, attr_vals) <= 0 for _, n in allsc)
    smax = max(s for s, _ in allsc)
    b, bs = categorize(root, q, attr_vals)
    if bs < smax - 1e-9:
        found += 1
        if found == 1:
            print("trial", trial, "returned score", bs, "but best score", smax)
            print("instances:", insts); print("query:", q)
print("violations:", found, "of 20000 random nominal trees")
