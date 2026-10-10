"""Constructed trees in 1-D: depth-one grid tree (Thm single / Prop prior) and a
balanced binary hierarchy over the same grid.  Scores follow Def. predict
(and the C++ expand()): root scored by its likelihood; non-root c:
  LIK: l_c, PRIOR: l_c + log(n_c/n_root), PATH: pi_c + l_c, POSTERIOR: pi_c.
Full expansion, all-concept and leaves-only mixtures."""
import numpy as np

def logN(x, mu, var):
    return -0.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)

def build_binary(points, ys, h2, depth_one=False):
    # nodes: list of dict(n, mu, s2, ymean, parent, children)
    nodes = []
    def make(idx):
        p = points[idx]; y = ys[idx]
        return dict(n=len(idx), mu=p.mean(), s2=p.var(), ymean=y.mean(), children=[], idx=idx)
    G = len(points)
    root = make(np.arange(G)); nodes.append(root)
    if depth_one:
        for i in range(G):
            c = make(np.array([i])); nodes.append(c); root['children'].append(len(nodes) - 1)
        return nodes
    stack = [0]
    while stack:
        k = stack.pop()
        idx = nodes[k]['idx']
        if len(idx) == 1:
            continue
        half = len(idx) // 2
        for part in (idx[:half], idx[half:]):
            c = make(part); nodes.append(c); nodes[k]['children'].append(len(nodes) - 1)
            stack.append(len(nodes) - 1)
    return nodes

def predict(nodes, x, h2, weighting, leaves_only):
    # returns mixture prediction at array x
    n_root = nodes[0]['n']
    out_num = np.zeros_like(x); out_den = np.zeros_like(x)
    # log-sum-exp safe accumulation: collect scores
    scores = []; vals = []
    ll = {}
    pi = {0: np.zeros_like(x)}
    for k, nd in enumerate(nodes):
        ll[k] = logN(x, nd['mu'], nd['s2'] + h2)
    # path posteriors
    order = [0]
    while order:
        k = order.pop()
        ch = nodes[k]['children']
        if not ch:
            continue
        un = np.array([ll[c] + np.log(nodes[c]['n'] / nodes[k]['n']) for c in ch])
        m = un.max(axis=0)
        lse = m + np.log(np.exp(un - m).sum(axis=0))
        for j, c in enumerate(ch):
            pi[c] = un[j] - lse + pi[k]
            order.append(c)
    for k, nd in enumerate(nodes):
        is_leaf = not nd['children']
        if leaves_only and not is_leaf:
            continue
        if k == 0:
            s = ll[0]
        elif weighting == 'lik':
            s = ll[k]
        elif weighting == 'prior':
            s = ll[k] + np.log(nd['n'] / n_root)
        elif weighting == 'path':
            s = pi[k] + ll[k]
        elif weighting == 'posterior':
            s = pi[k]
        scores.append(s); vals.append(nd['ymean'])
    S = np.array(scores); V = np.array(vals)[:, None]
    m = S.max(axis=0)
    W = np.exp(S - m)
    return (W * V).sum(axis=0) / W.sum(axis=0)

def run(g, Ds, beta=1.0, depth_one=False):
    res = {}
    xt = np.linspace(0.1, 0.9, 4001)  # interior queries
    for D in Ds:
        G = 2 ** D
        delta = 1.0 / G
        pts = (np.arange(G) + 0.5) * delta
        ys = g(pts)
        h = beta * delta; h2 = h * h
        nodes = build_binary(pts, ys, h2, depth_one=depth_one)
        row = {}
        for wt in ['lik', 'path', 'prior', 'posterior']:
            for lo in [False, True]:
                yhat = predict(nodes, xt, h2, wt, lo)
                err = np.sqrt(np.mean((yhat - g(xt)) ** 2))
                row[(wt, lo)] = err
        res[D] = row
    return res

if __name__ == '__main__':
    g = lambda x: np.sin(np.pi * (2 * x - 1))   # sin(pi u) on u in [-1,1]
    for depth_one in [True, False]:
        print('=== depth-one grid tree' if depth_one else '=== balanced binary hierarchy (depth D)')
        Ds = [4, 6, 8, 10, 12] if depth_one else [4, 6, 8, 10, 12, 14]
        res = run(g, Ds, depth_one=depth_one)
        hdr = 'D  G     ' + '  '.join(f'{w[:4]}{"/L" if lo else "/A"}' for w in ['lik','path','prior','posterior'] for lo in [False, True])
        print(hdr)
        for D in Ds:
            r = res[D]
            print(f'{D:2d} {2**D:6d} ' + '  '.join(f'{r[(w, lo)]:.4f}' for w in ['lik','path','prior','posterior'] for lo in [False, True]))
        if not depth_one:
            print('prior/all error * D :', [round(res[D][('prior', False)] * D, 3) for D in Ds])
            print('posterior/all error * D :', [round(res[D][('posterior', False)] * D, 3) for D in Ds])
