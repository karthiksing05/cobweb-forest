"""KAN-style interpretability: plot what each univariate tree of an activation forest
learned against the exact conditional expectation it estimates (Prop. sufficient class).
f(x, y) = exp(sin(pi x) + y^2): tree 1 should learn c1 * exp(sin(pi x)), tree 2 c2 * exp(y^2)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cobweb_forest import Forest, designs as D

rng = np.random.default_rng(0)
f = lambda X: np.exp(np.sin(np.pi * X[:, 0]) + X[:, 1] ** 2)
N = 4000
X = rng.uniform(-1, 1, (N, 2)).astype(np.float32); Y = f(X)[:, None].astype(np.float32)
F = Forest(D.ka((0, 1), "act", N, kappa=0.25, max_nodes=300, leaves_only=True), "out").fit(X, Y, protocol="online")
g = np.linspace(-1, 1, 201).astype(np.float32)
u = rng.uniform(-1, 1, 10 ** 6)
c1, c2 = np.mean(np.exp(u ** 2)), np.mean(np.exp(np.sin(np.pi * u)))
names = [s.name for s in F.specs if s.name != "out"]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
for ax, name, truth, lab in [(axes[0], names[0], c1 * np.exp(np.sin(np.pi * g)), r"$\mathbb{E}[f\mid x_1]=c_1e^{\sin\pi x_1}$"),
                             (axes[1], names[1], c2 * np.exp(g ** 2), r"$\mathbb{E}[f\mid x_2]=c_2e^{x_2^2}$")]:
    i = 0 if ax is axes[0] else 1
    act = []
    for v in g:
        x = np.zeros(2, np.float32); x[i] = v
        act.append(F.messages(x)[name]["act"][0])
    ax.plot(g, truth, color="#c3c2b7", lw=3, label="exact")
    ax.plot(g, act, color="#2a78d6", lw=1.4, label="tree activation")
    ax.set_title(f"univariate tree on x{i + 1}: " + lab, fontsize=7)
    ax.legend(fontsize=6, frameon=False)
# output tree: activation surface as a function of the two messages
A = np.array([[F.messages(np.array([a, b], np.float32))[n]["act"][0] for n in names] for a, b in rng.uniform(-1, 1, (400, 2))])
P = F.predict(rng.uniform(-1, 1, (400, 2)).astype(np.float32))
sc = axes[2].scatter(np.log(A[:, 0] / c1), np.log(A[:, 1] / c2), c=[F.trees["out"].act(np.zeros(0), a, [], None)[0] for a in A], s=6, cmap="Blues")
axes[2].set_xlabel(r"$\log(a_1/c_1)\approx\sin\pi x_1$"); axes[2].set_ylabel(r"$\log(a_2/c_2)\approx x_2^2$")
axes[2].set_title("output tree on the two activations", fontsize=7)
fig.colorbar(sc, ax=axes[2])
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("theory/figs/interpret.pdf"); fig.savefig("theory/figs/interpret.png", dpi=160)
