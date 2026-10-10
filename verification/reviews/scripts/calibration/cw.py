# Faithful port of cobweb_hybrid_node.cpp / cobweb_hybrid_tree.cpp (ifit + scoring), weight_attr=False,
# no attr_weights, cont_weight=1.  is_close_to_zero assumed = max|v|<=tol (helper.h not readable).
import math, numpy as np

def cat_entropy(N, K, m, snl, alpha):
    n0 = K - m
    zero = n0 * alpha * math.log(alpha) if (alpha > 0 and n0 > 0) else 0.0
    den = N + K * alpha
    if den <= 0: return 0.0
    return -((snl + zero) / den - math.log(den))

def gauss_entropy(var):
    if len(var) == 0: return 0.0
    return 0.5 * float(np.sum(np.log(2 * math.pi * math.e * var)))

class Node:
    def __init__(self, tree, other=None):
        self.tree = tree; self.parent = None; self.children = []
        self.count = 0.0; self.a_count = {}; self.av = {}; self.snl = {}
        self.mean = np.zeros(tree.size); self.ss = np.zeros(tree.size)
        if other is not None:
            self.parent = other.parent
            self.update_from(other)
            for c in other.children:
                cp = Node(tree, c); cp.parent = self; self.children.append(cp)

    def inc(self, disc, cont):
        a = self.tree.alpha; self.count += 1
        for attr, vm in disc.items():
            for val, cnt in vm.items():
                self.a_count[attr] = self.a_count.get(attr, 0.0) + cnt
                if attr >= 0 and attr in self.av and val in self.av[attr]:
                    tf = self.av[attr][val] + a; self.snl[attr] = self.snl.get(attr, 0.0) - tf * math.log(tf)
                self.av.setdefault(attr, {}); self.av[attr][val] = self.av[attr].get(val, 0.0) + cnt
                if attr >= 0:
                    tf = self.av[attr][val] + a; self.snl[attr] = self.snl.get(attr, 0.0) + tf * math.log(tf)
        if self.tree.size > 0:
            d = cont - self.mean; self.mean = self.mean + d / self.count; self.ss = self.ss + d * (cont - self.mean)

    def update_from(self, o):
        a = self.tree.alpha
        if self.tree.size > 0 and o.count > 0:
            if self.count > 0:
                n1, n2 = self.count, o.count; d = o.mean - self.mean
                self.ss = self.ss + o.ss + d * d * (n1 * n2 / (n1 + n2)); self.mean = (n1 * self.mean + n2 * o.mean) / (n1 + n2)
            else:
                self.mean = o.mean.copy(); self.ss = o.ss.copy()
        self.count += o.count
        for attr, vm in o.av.items():
            self.a_count[attr] = self.a_count.get(attr, 0.0) + o.a_count[attr]
            for val, cnt in vm.items():
                if attr >= 0 and attr in self.av and val in self.av[attr]:
                    tf = self.av[attr][val] + a; self.snl[attr] = self.snl.get(attr, 0.0) - tf * math.log(tf)
                self.av.setdefault(attr, {}); self.av[attr][val] = self.av[attr].get(val, 0.0) + cnt
                if attr >= 0:
                    tf = self.av[attr][val] + a; self.snl[attr] = self.snl.get(attr, 0.0) + tf * math.log(tf)

    # ---- discrete entropies
    def ent_attr(self, attr):
        T = self.tree
        if attr < 0 or attr not in T.attr_vals: return 0.0
        K = len(T.attr_vals[attr]); N = 0.0; m = 0
        if attr in self.av: N = self.a_count[attr]; m = len(self.av[attr])
        if N == 0 and T.alpha == 0: return 0.0
        return cat_entropy(N, K, m, self.snl.get(attr, 0.0), T.alpha)

    def ent_attr_insert(self, attr, disc):
        T = self.tree; a = T.alpha
        if attr < 0 or attr not in T.attr_vals: return 0.0
        K = len(T.attr_vals[attr]); N = 0.0; m = 0
        if attr in self.av: N = self.a_count[attr]; m = len(self.av[attr])
        snl = self.snl.get(attr, 0.0)
        if attr in disc:
            for val, cnt in disc[attr].items():
                N += cnt; prior = 0.0
                if attr in self.av and val in self.av[attr]:
                    prior = self.av[attr][val]; tf = prior + a; snl -= tf * math.log(tf)
                else: m += 1
                tf = prior + cnt + a; snl += tf * math.log(tf)
        if N == 0 and a == 0: return 0.0
        return cat_entropy(N, K, m, snl, a)

    def ent_attr_merge(self, attr, other, disc):
        T = self.tree; a = T.alpha
        if attr < 0 or attr not in T.attr_vals: return 0.0
        K = len(T.attr_vals[attr]); N = 0.0; m = 0
        if attr in self.av: N = self.a_count[attr]; m = len(self.av[attr])
        snl = self.snl.get(attr, 0.0)
        ov = other.av.get(attr); dv = disc.get(attr)
        if ov is not None:
            for val, oc in ov.items():
                ic = dv.get(val, 0.0) if dv is not None else 0.0
                N += oc + ic; prior = 0.0
                if attr in self.av and val in self.av[attr]:
                    prior = self.av[attr][val]; tf = prior + a; snl -= tf * math.log(tf)
                else: m += 1
                tf = prior + oc + ic + a; snl += tf * math.log(tf)
        if dv is not None:
            for val, ic in dv.items():
                if ov is not None and val in ov: continue
                N += ic; prior = 0.0
                if attr in self.av and val in self.av[attr]:
                    prior = self.av[attr][val]; tf = prior + a; snl -= tf * math.log(tf)
                else: m += 1
                tf = prior + ic + a; snl += tf * math.log(tf)
        if N == 0 and a == 0: return 0.0
        return cat_entropy(N, K, m, snl, a)

    # ---- continuous
    def var_model(self):
        pv = self.tree.prior_var
        return self.ss / self.count + pv if self.count > 0 else np.full(self.tree.size, pv)
    def var_insert(self, cont):
        n = self.count + 1; d = cont - self.mean; nm = self.mean + d / n
        return (self.ss + d * (cont - nm)) / n + self.tree.prior_var
    def var_merge(self, o, cont):
        n1, n2 = self.count, o.count
        if n1 + n2 > 0:
            d = o.mean - self.mean; ss = self.ss + o.ss + d * d * (n1 * n2 / (n1 + n2)); m = (n1 * self.mean + n2 * o.mean) / (n1 + n2)
        else:
            ss = np.zeros(self.tree.size); m = np.zeros(self.tree.size)
        n = n1 + n2
        if len(cont) > 0:
            n += 1; d = cont - m; m = m + d / n; ss = ss + d * (cont - m)
        if n <= 0: return np.full(self.tree.size, self.tree.prior_var)
        return ss / n + self.tree.prior_var

    def entropy(self):
        T = self.tree
        h = sum(self.ent_attr(a) for a in T.attr_vals if a >= 0)
        if T.size > 0: h += gauss_entropy(self.var_model())
        return h
    def entropy_insert(self, disc, cont):
        T = self.tree
        h = sum(self.ent_attr_insert(a, disc) for a in T.attr_vals if a >= 0)
        if T.size > 0: h += gauss_entropy(self.var_insert(cont))
        return h
    def entropy_merge(self, o, disc, cont):
        T = self.tree
        h = sum(self.ent_attr_merge(a, o, disc) for a in T.attr_vals if a >= 0)
        if T.size > 0: h += gauss_entropy(self.var_merge(o, cont))
        return h

    def pu_insert(self, child, disc, cont):
        ce = 0.0
        for c in self.children:
            if c is child: ce += (c.count + 1.0) / (self.count + 1.0) * c.entropy_insert(disc, cont)
            else: ce += c.count / (self.count + 1.0) * c.entropy()
        return (self.entropy_insert(disc, cont) - ce) / len(self.children)
    def pu_new(self, disc, cont):
        nc = Node(self.tree); nc.parent = self; nc.inc(disc, cont)
        ce = 1.0 / (self.count + 1.0) * nc.entropy()
        for c in self.children: ce += c.count / (self.count + 1.0) * c.entropy()
        return (self.entropy_insert(disc, cont) - ce) / (len(self.children) + 1)
    def pu_merge(self, b1, b2, disc, cont):
        ce = 0.0
        for c in self.children:
            if c is b1 or c is b2: continue
            ce += c.count / (self.count + 1.0) * c.entropy()
        ce += (b1.count + b2.count + 1.0) / (self.count + 1.0) * b1.entropy_merge(b2, disc, cont)
        return (self.entropy_insert(disc, cont) - ce) / (len(self.children) - 1)
    def pu_split(self, best):
        ce = 0.0
        for c in self.children:
            if c is best: continue
            ce += c.count / self.count * c.entropy()
        for c in best.children: ce += c.count / self.count * c.entropy()
        return (self.entropy() - ce) / (len(self.children) - 1 + len(best.children))

    def two_best(self, disc, cont):
        rng = self.tree.rng
        rel = [((c.count * c.entropy()) - ((c.count + 1) * c.entropy_insert(disc, cont)), c.count, rng.random(), i)
               for i, c in enumerate(self.children)]
        rel.sort(reverse=True)
        b1 = self.children[rel[0][3]]; b2 = self.children[rel[1][3]] if len(rel) > 1 else None
        return self.pu_insert(b1, disc, cont), b1, b2

    def best_op(self, disc, cont, b1, b2, b1pu):
        T = self.tree; rng = T.rng
        ops = [(b1pu, rng.random(), 'BEST')]
        if T.add_only and not T.allow_new: return 'BEST'
        if len(self.children) < T.branch_max:
            ops.append((self.pu_new(disc, cont), rng.random(), 'NEW'))
            if T.add_only:
                ops.sort(reverse=True); return ops[0][2]
            if len(self.children) > 2 and b2 is not None:
                ops.append((self.pu_merge(b1, b2, disc, cont), rng.random(), 'MERGE'))
            if len(b1.children) > 1 and (len(self.children) - 1 + len(b1.children)) <= T.branch_max:
                ops.append((self.pu_split(b1), rng.random(), 'SPLIT'))
        ops.sort(reverse=True); return ops[0][2]

    def exact(self, disc, cont, tol=1e-5):
        attrs = set(disc) | set(self.av)
        for attr in attrs:
            if attr < 0: continue
            ii = attr in disc; ic = attr in self.av
            if ii != ic: return False
            if not ii: continue
            tot = sum(disc[attr].values())
            for val in set(disc[attr]) | set(self.av[attr]):
                if (val in disc[attr]) != (val in self.av[attr]): return False
                if abs(disc[attr][val] / tot - self.av[attr][val] / self.a_count[attr]) > tol: return False
        if self.tree.size > 0 and len(cont) > 0:
            if self.count > 0 and np.max(np.abs(self.ss / self.count)) > tol: return False
            if np.max(np.abs(cont - self.mean)) > tol: return False
        return True

    def loglik(self, disc, cont):
        T = self.tree; a = T.alpha; lp = 0.0
        for attr, vm in disc.items():
            if attr < 0 or attr not in T.attr_vals: continue
            K = len(T.attr_vals[attr]); ac = (K + 1) * a + self.a_count.get(attr, 0.0)
            for val, cnt in vm.items():
                av = a
                if val in T.attr_vals[attr] and attr in self.av and val in self.av[attr]: av += self.av[attr][val]
                lp += cnt * (math.log(av) - math.log(ac))
        if T.size > 0 and len(cont) > 0:
            v = self.var_model()
            for d in range(len(cont)):
                if math.isnan(cont[d]): continue
                dd = cont[d] - self.mean[d]; lp += -0.5 * (math.log(v[d]) + math.log(2 * math.pi) + dd * dd / v[d])
        return lp

class Tree:
    def __init__(self, size, alpha=1.0, prior_var=1.0, branch_max=10**9, depth_max=1000, seed=0, add_only=False, allow_new=False):
        self.size = size; self.alpha = alpha; self.prior_var = prior_var; self.branch_max = branch_max
        self.depth_max = depth_max; self.add_only = add_only; self.allow_new = allow_new
        self.attr_vals = {}; self.rng = np.random.default_rng(seed); self.root = Node(self)

    def ifit(self, disc, cont):
        cont = np.asarray(cont, float)
        for attr, vm in disc.items():
            for val in vm: self.attr_vals.setdefault(attr, set()).add(val)
        cur = self.root; depth = 1; placed = False
        while depth < self.depth_max:
            if not cur.children and (cur.count == 0 or cur.exact(disc, cont)):
                cur.inc(disc, cont); placed = True; break
            elif not cur.children:
                nn = Node(self, cur); cur.parent = nn; nn.children.append(cur)
                if nn.parent is None: self.root = nn
                else:
                    p = nn.parent; p.children = [c for c in p.children if c is not cur]; p.children.append(nn)
                nn.inc(disc, cont)
                nc = Node(self); nc.parent = nn; nc.inc(disc, cont); nn.children.append(nc)
                cur = nc; placed = True; break
            else:
                b1pu, b1, b2 = cur.two_best(disc, cont)
                op = cur.best_op(disc, cont, b1, b2, b1pu)
                if op == 'BEST': cur.inc(disc, cont); cur = b1
                elif op == 'NEW':
                    cur.inc(disc, cont); nc = Node(self); nc.parent = cur; nc.inc(disc, cont); cur.children.append(nc)
                    cur = nc; placed = True; break
                elif op == 'MERGE':
                    cur.inc(disc, cont); nc = Node(self); nc.parent = cur
                    nc.update_from(b1); nc.update_from(b2); b1.parent = nc; b2.parent = nc
                    nc.children += [b1, b2]
                    cur.children = [c for c in cur.children if c is not b1 and c is not b2]; cur.children.append(nc)
                    cur = nc
                elif op == 'SPLIT':
                    cur.children = [c for c in cur.children if c is not b1]
                    for c in b1.children: c.parent = cur; cur.children.append(c)
                    b1.children = []
            depth += 1
        if not placed:
            cur.inc(disc, cont)
            if cur.children:
                nl = Node(self); nl.parent = cur; nl.inc(disc, cont); cur.children.append(nl); cur = nl
        return cur

    def nodes(self):
        out = []; st = [self.root]
        while st:
            n = st.pop(); out.append(n); st += n.children
        return out
    def leaves(self):
        return [n for n in self.nodes() if not n.children]
