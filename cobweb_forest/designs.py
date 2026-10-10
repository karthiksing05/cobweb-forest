"""Forest designs used by the theory and the experiments.

``single``    one tree over every raw input and the target (the flat baseline, Thm 1a)
``ka``        a forest shaped by a Kolmogorov-Arnold graph: a nested tuple of raw input
              indices, e.g. ((0, 1), (2, 3)).  An int is a univariate tree over that
              coordinate, a list [i, j] one tree over several raw coordinates, and a
              tuple a tree reading its children's messages.  The root is the output
              tree.
Message modes for ``ka``:
  "act"     every tree models its inputs jointly with y and passes its activation
            E_hat[y | inputs] (the target funnels through every tree)
  "label"   inner trees are unsupervised (no task) and pass labels; only the output
            tree sees y (Convolutional Cobweb's protocol)
  "hybrid"  every tree models y jointly and passes activation AND label
"""
from __future__ import annotations

from typing import List

from .forest import TreeSpec


def bandwidth(n_train: int, d: int, kappa: float, rate: str = "regression") -> float:
    """prior_var for a tree with ``d`` continuous inputs, in units of each attribute's
    standard deviation: (kappa * N^(-1/e))^2 with
      e = d      ("interpolation": the bandwidth tracks the spacing of N points; right
                  for noise-free targets, the existence constructions of Thm 1), or
      e = d + 4  ("regression": the MSE-optimal order for twice-differentiable
                  targets, which satisfies N h^d -> infinity as Thms 2a/2b require).
    """
    d = max(d, 1)
    e = d if rate == "interpolation" else d + 4
    return float((kappa * n_train ** (-1.0 / e)) ** 2)


def single(n_inputs: int, n_train: int, kappa: float = 1.0, rate: str = "regression",
           **kw) -> List[TreeSpec]:
    pv = bandwidth(n_train, n_inputs, kappa, rate)
    return [TreeSpec("out", raw=list(range(n_inputs)), task="y", prior_var=pv, **kw)]


def ka(graph, mode: str, n_train: int, kappa: float = 1.0, label_depth: int = 3,
       rate: str = "regression", **kw) -> List[TreeSpec]:
    specs: List[TreeSpec] = []
    counter = [0]

    def build(node, is_root):
        name = "out" if is_root else f"t{counter[0]}"
        counter[0] += 1
        if isinstance(node, int):
            raw, kids = [node], []
        elif isinstance(node, list):
            raw, kids = list(node), []
        else:
            raw, kids = [], [build(c, False) for c in node]
        supervised = is_root or mode in ("act", "hybrid")
        act_from = kids if mode in ("act", "hybrid") else []
        label_from = kids if mode in ("label", "hybrid") else []
        emits = {"act": ["act"], "label": ["label"], "hybrid": ["act", "label"]}[mode]
        d = len(raw) + len(act_from)
        specs.append(TreeSpec(name, raw=raw, act_from=act_from, label_from=label_from,
                              task="y" if supervised else None, emits=emits,
                              label_depth=label_depth,
                              prior_var=bandwidth(n_train, d, kappa, rate), **kw))
        return name

    build(graph, True)
    return specs


def balanced(indices, arity: int = 2):
    """A balanced KA graph over the given raw coordinates (pairs, then pairs of pairs)."""
    nodes = list(indices)
    while len(nodes) > 1:
        nodes = [tuple(nodes[i:i + arity]) if len(nodes[i:i + arity]) > 1 else nodes[i]
                 for i in range(0, len(nodes), arity)]
    return nodes[0]
