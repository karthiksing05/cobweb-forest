"""Cobweb Forest: a DAG of hybrid Cobweb trees whose representations feed one another.

A tree ``tau`` sees an instance made of
  * continuous inputs   -- raw coordinates and the *activations* of its parent trees,
  * discrete inputs     -- the *labels* of its parent trees (one attribute per parent,
                           value = the uid of a concept, or a bag {uid: weight}),
  * task attributes     -- variables the tree models jointly with its inputs and
                           predicts by pattern completion (a continuous target ``y`` or
                           a discrete class ``label``).  Inner trees may have no task.

Every tree learns only with Cobweb's own ``ifit`` (category utility, the four
operators).  There is no gradient and no outer optimiser: the forest is trained in
topological order (each tree sees its parents' messages computed by the already
trained parents -- the protocol of Convolutional Cobweb).

A tree emits two kinds of message:
  * ``act``   -- its multi-node prediction of its continuous task attributes (the
                 mixture sum_c w_c(q) mu_c with the task dimensions masked), and
  * ``label`` -- the concept the query sorts to at a given depth (``label_depth``;
                 ``None`` = the leaf) of the root-to-leaf descent by log P(c|x).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ._core import CobwebHybridTree

NAN = np.float32(np.nan)


def raw_outcome(task):
    """Indices of a cross-view outcome "raw:j" or "raw:j1+j2+...", else None."""
    if isinstance(task, str) and task.startswith("raw:"):
        return [int(j) for j in task.split(":", 1)[1].split("+")]
    return None
EMPTY = np.zeros(0, dtype=np.float32)


@dataclass
class TreeSpec:
    """One tree of a forest.

    raw:        indices of raw input coordinates this tree reads (continuous)
    act_from:   parent trees whose activation messages it reads (continuous)
    label_from: parent trees whose label messages it reads (discrete, one attr each)
    raw_disc:   indices of raw *discrete* input columns (e.g. tokens), one attr each
    task:       the tree's outcome (Def. "cognitive task"): None (categorization only),
                "y" (the continuous target, may be multi-dim), "class" (the discrete
                target), or "raw:j" (the raw coordinate j, which the tree must not
                perceive -- a cross-view, self-supervised outcome)
    emits:      subset of {"act", "label"}
    """
    name: str
    raw: Sequence[int] = ()
    act_from: Sequence[str] = ()
    label_from: Sequence[str] = ()
    raw_disc: Sequence[int] = ()
    task: Optional[str] = None
    emits: Sequence[str] = ("act",)
    label_depth: Optional[object] = None   # int depth, None = leaf, "basic" = basic level
    alpha: float = 0.1
    prior_var: float = 1e-3
    cont_weight: float = 1.0
    max_nodes: int = 200
    weighting: str = "path"
    leaves_only: bool = False
    add_only: bool = False
    seed: int = 0
    task_weight: float = 1.0               # attr weight of a discrete task attribute


class Tree:
    """A hybrid Cobweb tree with a fixed instance layout."""

    def __init__(self, spec: TreeSpec, y_dim: int = 1, n_act: Optional[int] = None,
                 n_classes: int = 0):
        self.spec = spec
        self.n_raw = len(spec.raw)
        self.n_act = len(spec.act_from) if n_act is None else n_act
        self.n_classes = n_classes
        js = raw_outcome(spec.task)
        if js is not None and set(js) & set(spec.raw):
            raise ValueError(f"tree {spec.name}: an outcome coordinate is also one of its cues")
        self.y_dim = y_dim if spec.task == "y" else (len(js) if js is not None else 0)
        self.size = self.n_raw + self.n_act + self.y_dim
        self.tree = CobwebHybridTree(self.size, alpha=spec.alpha, prior_var=spec.prior_var,
                                     cont_weight=spec.cont_weight)
        self.tree.set_random_seed(spec.seed)
        self.tree.add_only = spec.add_only
        self.class_attr = len(spec.label_from) + len(spec.raw_disc)
        # Per-dimension scale of the continuous block (Cobweb/3's acuity is
        # relative to an attribute's spread): instances are standardised with
        # the training statistics, so prior_var is a variance in units of each
        # attribute's standard deviation.  Set by Forest.fit.
        self.mu = np.zeros(self.size, dtype=np.float32)
        self.sd = np.ones(self.size, dtype=np.float32)
        if spec.task == "class" and spec.task_weight != 1.0:
            self.tree.set_attr_weight(self.class_attr, spec.task_weight)

    # ---- instance layout ------------------------------------------------
    def _cont(self, x_raw, acts, y=None):
        parts = [np.asarray(x_raw, dtype=np.float32)[list(self.spec.raw)]]
        if self.n_act:
            parts.append(np.asarray(acts, dtype=np.float32))
        if self.y_dim:
            parts.append(np.full(self.y_dim, NAN, dtype=np.float32) if y is None
                         else np.atleast_1d(np.asarray(y, dtype=np.float32)))
        return np.concatenate(parts) if self.size else EMPTY

    def _std(self, c):
        return (c - self.mu) / self.sd if self.size else c

    def _disc(self, labels, x_disc, cls=None):
        d = {}
        for i, lab in enumerate(labels):
            d[i] = lab if isinstance(lab, dict) else {int(lab): 1.0}
        off = len(labels)
        for j, col in enumerate(self.spec.raw_disc):
            d[off + j] = {int(x_disc[col]): 1.0}
        if cls is not None:
            d[self.class_attr] = {int(cls): 1.0}
        return d

    # ---- learning / inference --------------------------------------------
    def ifit(self, x_raw, acts, labels, x_disc, y=None, cls=None):
        return self.tree.ifit(self._disc(labels, x_disc, cls), self._std(self._cont(x_raw, acts, y)))

    def label(self, x_raw, acts, labels, x_disc):
        """uid of the concept the query sorts to (root-to-leaf descent, log P(c|x))."""
        q = self._std(self._cont(x_raw, acts))
        leaf = self.tree.get_leaf(self._disc(labels, x_disc), q)
        depth = self.spec.label_depth
        if depth is None:
            return leaf.uid
        if depth == "basic":
            # the basic level of the branch: the concept on the leaf's path with the
            # largest expected pointwise mutual information against the root
            return leaf.get_basic().uid
        path = []
        n = leaf
        while n is not None:
            path.append(n)
            n = n.parent
        path.reverse()                        # root ... leaf
        return path[min(depth, len(path) - 1)].uid

    def act(self, x_raw, acts, labels, x_disc):
        """The activation: the tree's mixture prediction of its task attributes -- the
        continuous task dims, or for a discrete task the predicted class distribution
        as a vector of length n_classes."""
        if self.spec.task == "class":
            p = self.predict_class(x_raw, acts, labels, x_disc)
            v = np.zeros(self.n_classes, dtype=np.float32)
            for c, pc in p.items():
                if 0 <= c < self.n_classes:
                    v[c] = pc
            return v
        q = self._std(self._cont(x_raw, acts))
        m = self.tree.predict_mean(self._disc(labels, x_disc), q, self.spec.max_nodes, False,
                                   self.spec.weighting, self.spec.leaves_only)
        k = self.n_raw + self.n_act
        return np.asarray(m)[k:] * self.sd[k:] + self.mu[k:]

    def predict_class(self, x_raw, acts, labels, x_disc):
        q = self._std(self._cont(x_raw, acts)) if self.size else EMPTY
        p = self.tree.predict_attr(self._disc(labels, x_disc), q, self.class_attr,
                                   self.spec.max_nodes, False, self.spec.weighting)
        return p


@dataclass
class Forest:
    """A DAG of trees; ``output`` names the designated output tree."""
    specs: List[TreeSpec]
    output: str
    y_dim: int = 1
    n_classes: int = 0
    raw_scale: str = "per_dim"     # "per_dim" or "global" (one acuity for all raw inputs, e.g. pixels)
    trees: Dict[str, Tree] = field(default_factory=dict)

    def __post_init__(self):
        self.by_name = {s.name: s for s in self.specs}
        self.order = self._topo()
        for name in self.order:
            s = self.by_name[name]
            n_act = sum(self.act_dim(p) for p in s.act_from)
            self.trees[name] = Tree(s, self.y_dim, n_act=n_act, n_classes=self.n_classes)

    def act_dim(self, name):
        task = self.by_name[name].task
        if raw_outcome(task) is not None:
            return len(raw_outcome(task))
        return self.y_dim if task == "y" else (self.n_classes if task == "class" else 0)

    def _topo(self):
        seen, order = set(), []

        def visit(n, stack=()):
            if n in seen:
                return
            if n in stack:
                raise ValueError(f"cycle through {n}")
            s = self.by_name[n]
            for p in list(s.act_from) + list(s.label_from):
                visit(p, stack + (n,))
            seen.add(n)
            order.append(n)
        for s in self.specs:
            visit(s.name)
        return order

    # ---- messages ---------------------------------------------------------
    def _inputs(self, name, x_raw, x_disc, msgs):
        s = self.by_name[name]
        acts = np.concatenate([msgs[p]["act"] for p in s.act_from]) if s.act_from else EMPTY
        labels = [msgs[p]["label"] for p in s.label_from]
        return x_raw, acts, labels, x_disc

    def ancestors(self, name):
        s = self.by_name[name]
        out = set()
        for p in list(s.act_from) + list(s.label_from):
            out |= {p} | self.ancestors(p)
        return out

    def messages(self, x_raw, x_disc=None, upto=None, only_ancestors_of=None):
        """Forward pass: every tree's messages for one query (no learning)."""
        keep = self.ancestors(only_ancestors_of) if only_ancestors_of else None
        msgs = {}
        for name in self.order:
            if name == upto:
                break
            if keep is not None and name not in keep:
                continue
            msgs[name] = self._emit(name, self._inputs(name, x_raw, x_disc, msgs))
        return msgs

    # ---- training ------------------------------------------------------------
    def _set_scales(self, X, Y):
        """Acuity scales: raw coordinates and y by their training spread; an
        activation dimension estimates y, so it is measured in y's units."""
        xm = np.nanmean(X, 0) if X is not None and len(X) else None
        xs = np.maximum(np.nanstd(X, 0), 1e-6) if xm is not None else None
        if xm is not None and self.raw_scale == "global":
            xm = np.zeros_like(xm)
            xs = np.full_like(xs, max(float(np.nanstd(X)), 1e-6))
        ym = np.nanmean(Y, 0) if Y is not None else np.zeros(self.y_dim)
        ys = np.maximum(np.nanstd(Y, 0), 1e-6) if Y is not None else np.ones(self.y_dim)
        for t in self.trees.values():
            mu, sd = [], []
            if t.n_raw:
                mu.append(xm[list(t.spec.raw)]); sd.append(xs[list(t.spec.raw)])
            for p in t.spec.act_from:
                if self.by_name[p].task == "class":
                    # a class distribution is measured on the scale of the one-hot class
                    k = self.n_classes
                    mu.append(np.full(k, 1.0 / k)); sd.append(np.full(k, np.sqrt((1.0 / k) * (1 - 1.0 / k))))
                elif raw_outcome(self.by_name[p].task) is not None:
                    js = raw_outcome(self.by_name[p].task)      # measured in the x_j's units
                    mu.append(xm[js]); sd.append(xs[js])
                else:
                    mu.append(ym); sd.append(ys)
            if t.y_dim:
                js = raw_outcome(t.spec.task)
                if js is not None:
                    mu.append(xm[js]); sd.append(xs[js])
                else:
                    mu.append(ym); sd.append(ys)
            if mu:
                t.mu = np.concatenate(mu).astype(np.float32)
                t.sd = np.concatenate(sd).astype(np.float32)

    def _emit(self, name, inp):
        t = self.trees[name]
        m = {}
        if "act" in t.spec.emits and self.act_dim(name):
            m["act"] = t.act(*inp)
        if "label" in t.spec.emits:
            m["label"] = t.label(*inp)
        return m

    def level(self, name):
        s = self.by_name[name]
        parents = list(s.act_from) + list(s.label_from)
        return 0 if not parents else 1 + max(self.level(p) for p in parents)

    def fit(self, X, Y=None, C=None, X_disc=None, protocol="online", verbose=False):
        """Train every tree with Cobweb's ``ifit`` only.

        X: (N, n) raw continuous inputs; Y: (N, y_dim) continuous targets (optional);
        C: (N,) class labels (optional); X_disc: (N, k) raw discrete inputs (optional).
        Trees whose task is "y"/"class" model the target jointly with their inputs
        (the target funnels through them); trees with task=None are unsupervised.

        protocol
          "online"  one pass over the stream; for each instance every tree, in
                    topological order, first emits its message (from what it has
                    learned so far) and then learns the instance.  Messages are
                    therefore always out-of-sample, as at test time
                    (predict-then-learn, the Cobweb-LLM protocol).
          "split"   the stream is cut into L+1 consecutive chunks (L = depth of
                    the DAG); a tree at level k learns only chunk k, reading the
                    messages of its (already trained, frozen) parents -- the
                    sample-splitting protocol of the consistency theorems.
          "stage"   stage-wise on the full stream (parents' messages computed
                    in-sample; leaks the target through memorising trees --
                    kept as a documented negative control).
        """
        N = len(X) if X is not None else len(X_disc)
        self._set_scales(X, Y)

        def row(i):
            return (X[i] if X is not None else EMPTY,
                    X_disc[i] if X_disc is not None else None)

        def learn(name, i, msgs):
            t = self.trees[name]
            xr, xd = row(i)
            inp = self._inputs(name, xr, xd, msgs)
            js = raw_outcome(t.spec.task)
            if js is not None:
                y = np.asarray(xr)[js]               # cross-view outcome: raw coordinates
            else:
                y = Y[i] if (Y is not None and t.spec.task == "y") else None
            c = C[i] if (C is not None and t.spec.task == "class") else None
            t.ifit(*inp, y=y, cls=c)

        if protocol == "online":
            for i in range(N):
                msgs = {}
                xr, xd = row(i)
                for name in self.order:
                    if name != self.output:
                        msgs[name] = self._emit(name, self._inputs(name, xr, xd, msgs))
                for name in self.order:
                    learn(name, i, msgs)
        elif protocol == "split":
            L = max(self.level(n) for n in self.order)
            bounds = np.linspace(0, N, L + 2).astype(int)
            for name in sorted(self.order, key=self.level):
                k = self.level(name)
                for i in range(bounds[k], bounds[k + 1]):
                    xr, xd = row(i)
                    msgs = self.messages(xr, xd, upto=None, only_ancestors_of=name)
                    learn(name, i, msgs)
        elif protocol == "stage":
            cache = [dict() for _ in range(N)]
            for name in self.order:
                for i in range(N):
                    learn(name, i, cache[i])
                if name != self.output:
                    for i in range(N):
                        xr, xd = row(i)
                        cache[i][name] = self._emit(name, self._inputs(name, xr, xd, cache[i]))
        else:
            raise ValueError(protocol)
        if verbose:
            print("  concepts:", self.num_concepts(), flush=True)
        return self

    def predict(self, X, X_disc=None):
        out = []
        for i in range(len(X) if X is not None else len(X_disc)):
            xr = X[i] if X is not None else EMPTY
            xd = X_disc[i] if X_disc is not None else None
            msgs = self.messages(xr, xd, upto=self.output)
            inp = self._inputs(self.output, xr, xd, msgs)
            t = self.trees[self.output]
            if t.spec.task == "class":
                p = t.predict_class(*inp)
                out.append(max(p, key=p.get) if p else -1)
            else:
                out.append(t.act(*inp))
        return np.array(out)

    def predict_proba(self, X, X_disc=None):
        """Class distributions of the output tree, one dict per query."""
        out = []
        for i in range(len(X) if X is not None else len(X_disc)):
            xr = X[i] if X is not None else EMPTY
            xd = X_disc[i] if X_disc is not None else None
            msgs = self.messages(xr, xd, upto=self.output)
            out.append(self.trees[self.output].predict_class(*self._inputs(self.output, xr, xd, msgs)))
        return out

    def num_concepts(self):
        return {n: t.tree.num_concepts() for n, t in self.trees.items()}
