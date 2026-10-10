"""Build a Cobweb tree with a prescribed structure (for the constructions in the proofs).

A structure is given as nested lists whose leaves are instances (disc, cont):
    spec = [leaf_a, [leaf_b, leaf_c]]       # root with children: leaf_a and an internal node
Every node's statistics are the pooled statistics of the instances below it (the additivity
of Def. "Cobweb tree"), computed exactly here and loaded through the C++ JSON reader.
"""
from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np

from ._core import CobwebHybridTree


def _is_instance(x):
    """An instance is (disc, cont) or (disc, cont, count) -- the latter a leaf holding `count`
    copies of the instance."""
    return isinstance(x, tuple) and len(x) in (2, 3) and isinstance(x[0], dict)


def _instances(node):
    if _is_instance(node):
        return [node[:2]] * (node[2] if len(node) == 3 else 1)
    out = []
    for c in node:
        out.extend(_instances(c))
    return out


def _stats(insts, size, alpha):
    n = len(insts)
    a_count, av_count = {}, {}
    for disc, _ in insts:
        for attr, vals in disc.items():
            for v, w in vals.items():
                a_count[attr] = a_count.get(attr, 0.0) + w
                av_count.setdefault(attr, {})
                av_count[attr][v] = av_count[attr].get(v, 0.0) + w
    snl = {a: sum((c + alpha) * math.log(c + alpha) for c in vs.values()) for a, vs in av_count.items()}
    if size:
        C = np.array([c for _, c in insts], dtype=np.float64).reshape(n, size)
        mean = C.mean(0)
        sum_sq = ((C - mean) ** 2).sum(0)
    else:
        mean = sum_sq = np.zeros(0)
    return n, a_count, av_count, snl, mean, sum_sq


def build(spec, size, alpha=1.0, prior_var=1.0, cont_weight=1.0, weight_attr=True):
    """Return a CobwebHybridTree whose concept hierarchy is exactly ``spec``."""
    nodes, attr_vals = [], {}

    def walk(node, parent_id):
        my_id = len(nodes)
        nodes.append(None)
        insts = _instances(node)
        n, a_count, av_count, snl, mean, sum_sq = _stats(insts, size, alpha)
        for disc, _ in insts:
            for attr, vals in disc.items():
                attr_vals.setdefault(attr, set()).update(vals)
        nodes[my_id] = {
            "id": my_id, "parent_id": parent_id, "concept_id": f"c{my_id}", "count": float(n),
            "a_count": {str(a): float(c) for a, c in a_count.items()},
            "sum_n_logn": {str(a): float(v) for a, v in snl.items()},
            "av_count": {str(a): {str(v): float(c) for v, c in vs.items()} for a, vs in av_count.items()},
            "mean": [float(x) for x in mean], "sum_sq": [float(x) for x in sum_sq]}
        if not _is_instance(node):
            for child in node:
                walk(child, my_id)

    walk(spec, None)
    doc = {"tree_parameters": {"size": size, "alpha": alpha, "prior_var": prior_var,
                               "cont_weight": cont_weight, "weight_attr": weight_attr,
                               "depth_max": 1000, "branch_max": 1000},
           "attr_weights": {}, "attr_vals": {str(a): sorted(int(v) for v in vs) for a, vs in attr_vals.items()},
           "nodes": nodes}
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    tree = CobwebHybridTree(size, alpha=alpha, prior_var=prior_var, cont_weight=cont_weight,
                            weight_attr=weight_attr)
    tree.load_json(path)
    os.remove(path)
    return tree


def leaves_of(tree):
    out, stack = [], [tree.root]
    while stack:
        n = stack.pop()
        if n.children:
            stack.extend(n.children)
        else:
            out.append(n)
    return out


def nodes_of(tree):
    out, stack = [], [tree.root]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children)
    return out
