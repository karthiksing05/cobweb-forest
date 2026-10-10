"""Mirror of CobwebHybridTree::expand + categorize(use_best=True, PATH) for 1-D
continuous trees, plus the two 'repairs' proposed in Rem. structure:
 (a) prune per subtree: when best_score > pi_c, do not push c's children, keep searching;
 (b) order the queue by path posterior (pi), same stop rule.
We report whether each returns the max-score concept."""
import heapq, math, random

def logN(x, mu, var):
    return -0.5 * (math.log(2 * math.pi * var) + (x - mu) ** 2 / var)

class Node:
    def __init__(self, pts):
        self.pts = list(pts); self.children = []
    @property
    def n(self): return len(self.pts)
    @property
    def mu(self): return sum(self.pts) / len(self.pts)
    @property
    def s2(self):
        m = self.mu; return sum((p - m) ** 2 for p in self.pts) / len(self.pts)

def ll(node, x, h2): return logN(x, node.mu, node.s2 + h2)

def all_scores(root, x, h2):
    out = []  # (score, pi, node)
    stack = [(root, 0.0, ll(root, x, h2))]
    while stack:
        nd, pi, sc = stack.pop(); out.append((sc, pi, nd))
        if nd.children:
            un = [ll(c, x, h2) + math.log(c.n / nd.n) for c in nd.children]
            m = max(un); lse = m + math.log(sum(math.exp(u - m) for u in un))
            for c, u in zip(nd.children, un):
                cpi = u - lse + pi
                stack.append((c, cpi, cpi + ll(c, x, h2)))
    return out

def categorize(root, x, h2, mode):
    """mode: 'code' (C++ rule), 'prune' (per-subtree pruning), 'order' (queue by pi)."""
    cnt = 0
    def key(score, pi):
        return -(pi if mode == 'order' else score)
    q = [(key(ll(root, x, h2), 0.0), 0, ll(root, x, h2), 0.0, root)]
    best, best_score = root, -math.inf
    while q:
        _, _, score, pi, nd = heapq.heappop(q)
        if score > best_score:
            best, best_score = nd, score
        if best_score > pi:
            if mode in ('code', 'order'):
                break            # C++: visit returns false -> whole search ends
            else:
                continue         # prune this subtree only
        if nd.children:
            un = [ll(c, x, h2) + math.log(c.n / nd.n) for c in nd.children]
            m = max(un); lse = m + math.log(sum(math.exp(u - m) for u in un))
            for c, u in zip(nd.children, un):
                cpi = u - lse + pi
                cnt += 1
                heapq.heappush(q, (key(cpi + ll(c, x, h2), cpi), cnt, cpi + ll(c, x, h2), cpi, c))
    return best, best_score

# --- Example 1: two points, small spread: l_root > 0 ---
h2 = 1e-6
leafA, leafB = Node([0.0]), Node([0.1])
root = Node([0.0, 0.1]); root.children = [leafA, leafB]
x = 0.0
sc = all_scores(root, x, h2)
true_best = max(sc, key=lambda t: t[0])
print('Example 1: points {0, 0.1}, h^2=1e-6, query 0')
print('  scores: ' + ', '.join(f'(mean={t[2].mu:.3f}, n={t[2].n}): score={t[0]:.3f}, pi={t[1]:.3f}' for t in sc))
for mode in ['code', 'prune', 'order']:
    b, bs = categorize(root, x, h2, mode)
    print(f'  {mode:6s} returns node mean={b.mu:.3f} n={b.n} score={bs:.3f}; true max score={true_best[0]:.3f}')

# --- Random search for failures with l_root < 0 (non-root failures) ---
random.seed(0)
def random_tree(pts, depth=0):
    nd = Node(pts)
    if len(pts) == 1 or depth > 6: return nd
    pts = sorted(pts)
    k = random.randint(2, min(4, len(pts)))
    cuts = sorted(random.sample(range(1, len(pts)), k - 1))
    parts = [pts[i:j] for i, j in zip([0] + cuts, cuts + [len(pts)])]
    nd.children = [random_tree(p, depth + 1) for p in parts]
    return nd

fails = {'code': 0, 'prune': 0, 'order': 0}; trials = 0; nonroot_ex = {}
for t in range(3000):
    pts = [random.uniform(-3, 3) for _ in range(random.randint(3, 12))]
    root = random_tree(pts)
    h2 = 10 ** random.uniform(-4, -1)
    x = random.uniform(-3, 3)
    if ll(root, x, h2) >= 0:   # skip the trivial root case
        continue
    trials += 1
    tb = max(t_[0] for t_ in all_scores(root, x, h2))
    for mode in fails:
        b, bs = categorize(root, x, h2, mode)
        if bs < tb - 1e-9:
            fails[mode] += 1
            nonroot_ex.setdefault(mode, (bs, tb, h2))
print(f'Random trees with l_root<0: {trials} trials; failures (returned score < max score):', fails)
print('first failure per mode (returned, true max, h^2):', nonroot_ex)
