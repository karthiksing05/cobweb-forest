"""Figures and the prediction-vs-measurement table for theory/main.tex (v8) from results/theory
and theory/verification.json.  Works on partial results.

    PYTHONPATH=experiments:. python experiments/theory_figs.py
"""
import collections
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_figures import COL, FIG, GRID, INK, TAB, load   # shared style

R = "results/theory"
DCOL = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eb6834", 8: "#1baf7a"}


def slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def one(path):
    p = f"{R}/{path}.json"
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def guide(ax, x, y0, s, **kw):
    x = np.asarray(x, float)
    ax.plot(x, y0 * (x / x[0]) ** s, ls="--", lw=0.9, **kw)


def fig_compose():
    lip, smooth = one("r2_lip"), one("r2")
    src = lip or smooth
    if not src:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for ax, n in zip(axes, ("2", "4", "8")):
        v = src[n]
        F = [(r["G_total"], r["sup_err"]) for r in v["forest"]]
        S = [(r["G_total"], r["sup_err"]) for r in v["single"]]
        ax.loglog(*zip(*F), "s-", color=COL["act"], ms=3.5, label="forest of 2-input trees")
        ax.loglog(*zip(*S), "o-", color=COL["single"], ms=3.5, label="one tree")
        guide(ax, [a for a, _ in F], F[0][1], -0.5, color=COL["act"])
        guide(ax, [a for a, _ in S], S[0][1], -1 / int(n), color=COL["single"])
        if lip and smooth:
            Fs = [(r["G_total"], r["sup_err"]) for r in smooth[n]["forest"]]
            ax.loglog(*zip(*Fs), ":", color=COL["act"], lw=1.0, alpha=0.7, label="forest, smooth maps")
        ax.set_title(f"$n={n}$", color=INK)
        ax.set_xlabel("total leaves")
    axes[0].set_ylabel("sup error")
    axes[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/compose.pdf"); fig.savefig(f"{FIG}/compose.png", dpi=160)
    plt.close(fig)


def l1_table():
    L = collections.defaultdict(list)
    for f in glob.glob(f"{R}/l1_*.json"):
        d = json.load(open(f, encoding="utf-8"))
        L[(d["d"], d["N"])].append(d["rmse_interior"])
    return L


def l2_table():
    L = collections.defaultdict(list)
    for f in glob.glob(f"{R}/l2_*.json"):
        d = json.load(open(f, encoding="utf-8"))
        for m, r in d["results"].items():
            L[(d["n"], m, d["N"])].append(r["rmse_interior"] / d["sd_f"])
    return L


def fig_learn():
    L1, L2 = l1_table(), l2_table()
    if not L1:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    for d in (1, 2, 3):
        Ns = sorted(N for (x, N) in L1 if x == d)
        if not Ns:
            continue
        e = [np.mean(L1[(d, N)]) for N in Ns]
        ax.loglog(Ns, e, "o-", color=DCOL[d], ms=3.5, label=f"$d={d}$")
        guide(ax, Ns, e[0], -2 / (d + 4), color=DCOL[d])
    ax.set_xlabel("$N$"); ax.set_ylabel("interior RMSE"); ax.legend(fontsize=7)
    ax.set_title("one learned tree", color=INK)
    ax = axes[1]
    for n in (2, 4, 8):
        for m, mk, ls in (("single", "o", "-"), ("forest", "s", "--")):
            Ns = sorted(N for (x, mm, N) in L2 if x == n and mm == m)
            if not Ns:
                continue
            e = [np.mean(L2[(n, m, N)]) for N in Ns]
            ax.loglog(Ns, e, marker=mk, ls=ls, color=DCOL[n] if n != 2 else "#2a78d6", ms=3.5,
                      label=f"$n={n}$, " + ("one tree" if m == "single" else "forest"))
    ax.set_xlabel("$N$"); ax.set_ylabel("interior RMSE / sd$(f)$"); ax.legend(fontsize=6)
    ax.set_title("learned forest vs one tree", color=INK)
    fig.tight_layout()
    fig.savefig(f"{FIG}/learn.pdf"); fig.savefig(f"{FIG}/learn.png", dpi=160)
    plt.close(fig)


def fig_continual():
    D = load("results/continual/s0.json")
    if not D:
        return
    d = D[0]
    x, y = np.array(d["x"]), np.array(d["y"])
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.5), sharey=True)
    for k, ax in enumerate(axes):
        ax.plot(x, y, color=GRID, lw=2.5)
        ax.plot(x, d["methods"]["cobweb"][k], color=COL["single"], lw=1.2)
        ax.axvspan(-1 + 0.4 * k, -1 + 0.4 * (k + 1), color="#f1efe8", zorder=0)
        ax.set_ylim(-0.5, 1.5); ax.grid(False)
        ax.set_title(f"after phase {k + 1}", color=INK, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/continual_cobweb.pdf"); fig.savefig(f"{FIG}/continual_cobweb.png", dpi=160)
    plt.close(fig)


def tab_equiv():
    V = {c["check"]: c for c in json.load(open("theory/verification.json", encoding="utf-8"))["checks"]}
    rows = []

    def row(ref, quantity, pred, meas):
        rows.append(f"{ref} & {quantity} & {pred} & {meas}\\\\")
    r1 = one("r1")
    if r1:
        for d in ("1", "2", "3"):
            row(r"\cref{prop:rate}" if d == "1" else "", f"one constructed tree, $d={d}$: rate in $G$",
                f"${-1 / int(d):.3f}$", f"${r1[d]['measured_slope']:.3f}$")
    for key, name in (("r2_lip", "Lipschitz maps"), ("r2", "smooth maps")):
        r2 = one(key)
        if not r2:
            continue
        for n in ("2", "4", "8"):
            v = r2[n]
            row(r"\cref{thm:compose}" if (n == "2" and key == "r2_lip") else "",
                f"{name}, $n={n}$: forest / one tree, rate in $G$",
                (f"$\\le{-0.5:.3f}$ / $\\le{-1 / int(n):.3f}$" if key == "r2" else f"${-0.5:.3f}$ / ${-1 / int(n):.3f}$"),
                f"${v['measured_slope_forest']:.3f}$ / ${v['measured_slope_single']:.3f}$")
    c = V.get("check_tree_is_nw")
    if c:
        row(r"\cref{thm:learn-tree}", "learned tree $=$ Nadaraya--Watson: identity", "$0$",
            f"${float(c['max_abs_diff_lik_vs_NW']):.1e}$")
    L1 = l1_table()
    for d in (1, 2, 3):
        Ns = sorted(N for (x, N) in L1 if x == d)
        if len(Ns) >= 3:
            row("", f"one learned tree, $d={d}$: interior rate in $N$", f"${-2 / (d + 4):.3f}$",
                f"${slope(Ns, [np.mean(L1[(d, N)]) for N in Ns]):.3f}$")
    c = V.get("check_interweave")
    if c:
        row(r"\cref{prop:interweave}", "merged trees $=$ mixture: identity", "$0$", f"${float(c['max_abs_diff']):.1e}$")
    c = V.get("check_single_tree")
    if c:
        pr = max(r["sup_err"]["prior"] for r in c["rows"]["d1"][-1:])
        row(r"\cref{prop:prior}", "\\textsc{prior} rule, flat tree: error floor", "$>0$", f"${pr:.2f}$")
    L3 = [json.load(open(f, encoding="utf-8")) for f in glob.glob(f"{R}/l3_*.json")]
    if L3:
        Nmax = max(d["N"] for d in L3)
        top = [d for d in L3 if d["N"] == Nmax]
        m = lambda k: np.mean([d["xy"]["measured"][k]["ratio"] for d in top])
        row(r"\cref{prop:tasks}(b)", f"funnel forest on $x_1x_2$: floor ($N={Nmax}$)", "$1$", f"${m('funnel'):.3f}$")
        row(r"\cref{prop:tasks}(c)", f"label forest on $x_1x_2$: floor ($N={Nmax}$)", "$\\to0$", f"${m('label'):.3f}$")
        row(r"\cref{thm:learn-tree}", f"one tree on $x_1x_2$ ($N={Nmax}$)", "$\\to0$", f"${m('single'):.3f}$")
        row(r"\cref{prop:tasks}(d)", f"cross-view forest, view-specific target: floor ($N={Nmax}$)",
            f"${top[0]['multiview']['predicted_ratio']:.3f}$",
            f"${np.mean([d['multiview']['measured_ratio'] for d in top]):.3f}$")
    L2 = l2_table()
    for n in (4, 8):
        out = []
        for m in ("forest", "single"):
            Ns = sorted(N for (x, mm, N) in L2 if x == n and mm == m)
            out.append(slope(Ns, [np.mean(L2[(n, m, N)]) for N in Ns]) if len(Ns) >= 3 else float("nan"))
        if np.isfinite(out).all():
            row(r"\cref{rem:crossover}" if n == 4 else "", f"learned forest / one tree, $n={n}$: rate in $N$ (heuristic)",
                f"${-1 / 3:.3f}$ / ${-2 / (n + 4):.3f}$", f"${out[0]:.3f}$ / ${out[1]:.3f}$")
    row(r"\cref{prop:local}", "law of $(x_2,Y)$ in the two worlds", "equal", "equal (exact; Lean)")
    lines = [r"\begin{tabular}{@{}llll@{}}\toprule", r"result & quantity & predicted & measured\\\midrule"]
    lines += rows + [r"\bottomrule\end{tabular}"]
    open(f"{TAB}/equivariance.tex", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    fig_compose()
    fig_learn()
    fig_continual()
    tab_equiv()
