"""The forest iteration of the hybrid tree reproduces upstream cobweb_hybrid exactly
on fully observed data, and NaN-masked queries equal scoring the observed block alone."""
import sys

import numpy as np
import pytest

from cobweb_forest import CobwebHybridTree

UPSTREAM = "/coc/flash10/ksingara3/cobweb-private/lib/current"


def _shape(node):
    return (round(node.count, 6), tuple(np.round(node.mean, 5)),
            tuple(sorted((_shape(c) for c in node.children), key=str)))


def _data(n=150, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n, 1)).astype(np.float32)
    y = np.sin(3 * x).astype(np.float32)
    lab = (x[:, 0] > 0).astype(int)
    return [({0: {int(l): 1.0}}, np.concatenate([a, b])) for a, b, l in zip(x, y, lab)]


@pytest.mark.parametrize("add_only", [False, True])
def test_matches_upstream(add_only):
    sys.path.insert(0, UPSTREAM)
    import cobweb_hybrid as up
    data = _data()
    a = CobwebHybridTree(2, alpha=0.1, prior_var=0.01)
    b = up.CobwebHybridTree(2, alpha=0.1, prior_var=0.01)
    a.set_random_seed(3)
    b.set_random_seed(3)
    a.add_only = b.add_only = add_only
    for d, c in data:
        a.ifit(d, c)
        b.ifit(d, c)
    assert _shape(a.root) == _shape(b.root)
    for d, c in data[:20]:
        for w in ["path", "prior", "posterior", "likelihood"]:
            pa, ma = a.predict(d, c, 50, False, w)
            pb, mb = b.predict(d, c, 50, False, w)
            np.testing.assert_allclose(ma, mb, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(pa[0][1], pb[0][1], rtol=1e-4, atol=1e-6)


def test_nan_mask_equals_marginal():
    data = _data()
    t = CobwebHybridTree(2, alpha=0.1, prior_var=0.01)
    t.set_random_seed(0)
    for d, c in data:
        t.ifit(d, c)
    x = np.array([0.3, np.nan], dtype=np.float32)
    for node in [t.root] + list(t.root.children):
        full = node.log_prob_instance({}, np.array([0.3, 0.0], dtype=np.float32))
        var = node.variance()
        y_term = -0.5 * (np.log(var[1]) + np.log(2 * np.pi) + (0.0 - node.mean[1]) ** 2 / var[1])
        assert node.log_prob_instance({}, x) == pytest.approx(full - y_term, rel=1e-5)
    # prediction of the hidden dimension tracks sin(3x)
    m = t.predict_mean({}, x, 1000, False, "prior", True)
    assert abs(m[1] - np.sin(0.9)) < 0.1


def test_uids_unique_and_leaf_stable():
    t = CobwebHybridTree(1, prior_var=0.01)
    leaf = t.ifit({}, np.array([0.0], dtype=np.float32))
    uid = leaf.uid
    for v in np.linspace(-1, 1, 30):
        t.ifit({}, np.array([v], dtype=np.float32))
    seen, stack = set(), [t.root]
    while stack:
        n = stack.pop()
        assert n.uid not in seen
        seen.add(n.uid)
        stack.extend(n.children)
    assert uid in seen
