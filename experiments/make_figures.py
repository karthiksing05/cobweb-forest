"""Figures and tables for theory/main.tex from results/ (works on partial results).

    python experiments/make_figures.py
Writes theory/figs/*.pdf and theory/tables/*.tex, and prints a plain-text summary.
"""
import collections
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG, TAB = "theory/figs", "theory/tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# Reference categorical palette (fixed order; colour follows the method, never its rank).
COL = {"single": "#2a78d6", "act": "#eb6834", "label": "#1baf7a", "hybrid": "#eda100",
       "act-split": "#e87ba4", "mlp": "#008300", "kan": "#4a3aa7"}
MARK = {"single": "o", "act": "s", "label": "^", "hybrid": "D", "act-split": "v", "mlp": "P", "kan": "X"}
NAME = {"single": "one tree", "act": "forest: activations", "label": "forest: labels",
        "hybrid": "forest: hybrid", "act-split": "forest: act. (split)", "mlp": "MLP", "kan": "KAN"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6})


def load(pattern):
    return [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(pattern))]


def base(m):
    return m.split(":")[0]


def msd(v):
    v = np.asarray(v, dtype=float)
    return float(np.mean(v)), float(np.std(v))


# ---------------------------------------------------------------- weighting (Thm 1, Prop 2)
def fig_weighting():
    for fn in ("sin", "j0"):
        R = collections.defaultdict(list)
        for d in load(f"results/weighting/{fn}_s*.json"):
            for r in d["rows"]:
                if r["sched"] == "interpolation":
                    R[(r["weighting"], r["leaves_only"], r["N"])].append(r["rmse_interior"])
        if not R:
            continue
        Ns = sorted({k[2] for k in R})
        fig, ax = plt.subplots(figsize=(3.3, 2.5))
        styles = [("likelihood", True, "#2a78d6", "o", "leaves only (lik = prior)"),
                  ("path", False, "#eb6834", "s", "path, all concepts"),
                  ("likelihood", False, "#1baf7a", "^", "lik, all concepts"),
                  ("prior", False, "#eda100", "D", "prior, all concepts"),
                  ("posterior", False, "#e87ba4", "v", "posterior, all concepts")]
        for w, lo, c, mk, lab in styles:
            y = [np.mean(R[(w, lo, N)]) for N in Ns]
            ax.loglog(Ns, y, color=c, marker=mk, ms=4, label=lab)
        ref = np.array(Ns, dtype=float)
        y0 = np.mean(R[("likelihood", True, Ns[0])])
        ax.loglog(ref, y0 * (ref / ref[0]) ** -1.0, ls="--", color=MUTED, lw=1, label=r"slope $-1$ ($G^{-1/d}$, $d=1$)")
        ax.set_xlabel("training points N (= leaves)")
        ax.set_ylabel("test RMSE (interior)")
        ax.set_title(f"{'sin(πx)' if fn == 'sin' else 'J0(20x)'}: one learned tree", color=INK)
        ax.legend(fontsize=6.5)
        fig.tight_layout()
        fig.savefig(f"{FIG}/weighting_{fn}.pdf"); fig.savefig(f"{FIG}/weighting_{fn}.png", dpi=160)
        plt.close(fig)


# ---------------------------------------------------------------- toy scaling
def toy_table():
    rows = load("results/toys/*.json")
    for m in ("single", "act", "label", "hybrid", "act-split", "mlp", "kan"):
        for d in load(f"results/toys/exp100_parts/{m}/*.json"):
            rows.append(d)
    T = collections.defaultdict(list)          # (fn, N, method) -> [test rmse / sd_f]
    for d in rows:
        for m, r in d["results"].items():
            T[(d["fn"], d["N"], m)].append(r["test"] / d["sd_f"])
    return T


def fig_toys(T):
    fns = ["sin", "j0", "expsin", "xy", "nested4", "exp100"]
    titles = {"sin": "sin(πx)", "j0": "J0(20x)", "expsin": "exp(sin πx + y²)", "xy": "xy",
              "nested4": "4-D nested", "exp100": "100-D exp(mean sin²)"}
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4))
    for ax, fn in zip(axes.ravel(), fns):
        meths = sorted({k[2] for k in T if k[0] == fn})
        for m in meths:
            b = base(m)
            if b not in COL:
                continue
            Ns = sorted({k[1] for k in T if k[0] == fn and k[2] == m})
            mu = [np.mean(T[(fn, N, m)]) for N in Ns]
            ls = ":" if m.endswith(":uni") and fn == "nested4" else "-"
            lab = NAME[b] + (" (uni graph)" if ls == ":" else "")
            ax.loglog(Ns, mu, color=COL[b], marker=MARK[b], ms=3.5, ls=ls, label=lab)
        ax.set_title(titles[fn], color=INK)
        ax.set_xlabel("N")
    axes[0, 0].set_ylabel("test RMSE / sd(f)")
    axes[1, 0].set_ylabel("test RMSE / sd(f)")
    h, l = [], []
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(f"{FIG}/toys.pdf"); fig.savefig(f"{FIG}/toys.png", dpi=160)
    plt.close(fig)


def tab_toys(T):
    fns = ["sin", "j0", "expsin", "xy", "nested4", "exp100"]
    meths = ["single", "act", "label", "hybrid", "act-split", "mlp", "kan"]
    lines = [r"\begin{tabular}{@{}ll" + "r" * len(meths) + r"@{}}\toprule",
             "fn & N & " + " & ".join(NAME[m].replace("forest: ", "") for m in meths) + r"\\\midrule"]
    for fn in fns:
        for N in sorted({k[1] for k in T if k[0] == fn}):
            cells = []
            for m in meths:
                ks = [k for k in T if k[0] == fn and k[1] == N and base(k[2]) == m]
                if not ks:
                    cells.append("--")
                    continue
                k = min(ks, key=lambda k: np.mean(T[k]))     # best graph for this method
                mu, sd = msd(T[k])
                cells.append(f"{mu:.3f}" + ("$^{u}$" if k[2].endswith(":uni") and fn == "nested4" else ""))
            lines.append(f"{fn} & {N} & " + " & ".join(cells) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    open(f"{TAB}/toys.tex", "w", encoding="utf-8").write("\n".join(lines))


# ---------------------------------------------------------------- continual (Lemma leaves)
def fig_continual():
    D = load("results/continual/s0.json")
    if not D:
        return
    d = D[0]
    x, y = np.array(d["x"]), np.array(d["y"])
    fig, axes = plt.subplots(3, 5, figsize=(7.0, 3.6), sharex=True, sharey=True)
    for i, m in enumerate(["cobweb", "mlp", "kan"]):
        for k in range(5):
            ax = axes[i, k]
            ax.plot(x, y, color=GRID, lw=2.5)
            ax.plot(x, d["methods"][m][k], color=COL["single" if m == "cobweb" else m], lw=1.2)
            ax.axvspan(-1 + 0.4 * k, -1 + 0.4 * (k + 1), color="#f1efe8", zorder=0)
            ax.set_ylim(-0.5, 1.5)
            ax.grid(False)
            if k == 0:
                ax.set_ylabel({"cobweb": "Cobweb", "mlp": "MLP", "kan": "KAN"}[m])
            if i == 0:
                ax.set_title(f"after phase {k + 1}", color=INK)
    fig.tight_layout()
    fig.savefig(f"{FIG}/continual.pdf"); fig.savefig(f"{FIG}/continual.png", dpi=160)
    plt.close(fig)
    S = load("results/continual/s*.json")
    lines = [r"\begin{tabular}{@{}l" + "r" * 5 + r"@{}}\toprule",
             r"method & \multicolumn{5}{c}{RMSE on all data seen so far, after phase 1..5}\\\midrule"]
    for m in ["cobweb", "mlp", "kan"]:
        v = np.mean([s["summary"][m]["rmse_seen_after_phase"] for s in S], 0)
        lines.append(m + " & " + " & ".join(f"{a:.3f}" for a in v) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    open(f"{TAB}/continual.tex", "w", encoding="utf-8").write("\n".join(lines))


# ---------------------------------------------------------------- label cells
def fig_cells():
    R = collections.defaultdict(list)
    for d in load("results/label_cells/s*.json"):
        for r in d["rows"]:
            R[(r["d"], r["depth"], r["N"])].append(r["mean_spread"])
    if not R:
        return
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3))
    for ax, dim in zip(axes, (1, 2)):
        depths = sorted({k[1] for k in R if k[0] == dim}, key=lambda z: 99 if z == -1 else z)
        cmap = plt.get_cmap("Blues")
        for j, dp in enumerate(depths):
            Ns = sorted({k[2] for k in R if k[0] == dim and k[1] == dp})
            ax.loglog(Ns, [np.mean(R[(dim, dp, N)]) for N in Ns], marker="o", ms=3,
                      color=cmap(0.35 + 0.65 * j / max(1, len(depths) - 1)),
                      label="leaf" if dp == -1 else f"depth {dp}")
        ax.set_title(f"unsupervised tree, d = {dim}", color=INK)
        ax.set_xlabel("N")
    axes[0].set_ylabel("RMS spread of the label cell")
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{FIG}/label_cells.pdf"); fig.savefig(f"{FIG}/label_cells.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- real-world tables
def tab_real():
    out = []
    # housing
    H = collections.defaultdict(list)
    for d in load("results/housing/*.json"):
        for m, r in d["results"].items():
            H[(d["N"], m)].append(r["test"])
    if H:
        meths = sorted({k[1] for k in H}, key=lambda m: list(COL).index(base(m)) if base(m) in COL else 99)
        lines = [r"\begin{tabular}{@{}l" + "r" * len(meths) + r"@{}}\toprule",
                 "N & " + " & ".join(m.replace("_", r"\_") for m in meths) + r"\\\midrule"]
        for N in sorted({k[0] for k in H}):
            lines.append(f"{N} & " + " & ".join(
                (lambda v: f"{v[0]:.3f}$\\pm${v[1]:.3f}")(msd(H[(N, m)])) if H[(N, m)] else "--" for m in meths) + r"\\")
        lines.append(r"\bottomrule\end{tabular}")
        open(f"{TAB}/housing.tex", "w", encoding="utf-8").write("\n".join(lines))
        out.append(("housing (test RMSE of log value)", H))
    # mnist
    M = collections.defaultdict(list)
    for d in load("results/mnist/*.json"):
        for m, r in d["results"].items():
            M[(d["N"], m)].append(r["test"])
    if M:
        meths = [m for m in ("single", "label", "act", "hybrid", "mlp", "kan") if any(k[1] == m for k in M)]
        lines = [r"\begin{tabular}{@{}l" + "r" * len(meths) + r"@{}}\toprule",
                 "N & " + " & ".join(meths) + r"\\\midrule"]
        for N in sorted({k[0] for k in M}):
            lines.append(f"{N} & " + " & ".join(
                (lambda v: f"{100 * v[0]:.1f}$\\pm${100 * v[1]:.1f}")(msd(M[(N, m)])) if M[(N, m)] else "--"
                for m in meths) + r"\\")
        lines.append(r"\bottomrule\end{tabular}")
        open(f"{TAB}/mnist.tex", "w", encoding="utf-8").write("\n".join(lines))
        out.append(("mnist (test accuracy)", M))
    # owt
    O = collections.defaultdict(list)
    for d in load("results/owt/*.json"):
        for m, r in d["results"].items():
            O[(d["N"], m)].append((r["test_acc"], r["test_nll"]))
    if O:
        meths = [m for m in ("ngram", "mlp", "single", "label-unsup", "label-funnel") if any(k[1] == m for k in O)]
        lines = [r"\begin{tabular}{@{}l" + "r" * len(meths) + r"@{}}\toprule",
                 "N & " + " & ".join(meths) + r"\\\midrule"]
        for N in sorted({k[0] for k in O}):
            lines.append(f"{N} & " + " & ".join(
                f"{100 * O[(N, m)][0][0]:.1f} / {O[(N, m)][0][1]:.2f}" if O[(N, m)] else "--" for m in meths) + r"\\")
        lines.append(r"\bottomrule\end{tabular}")
        open(f"{TAB}/owt.tex", "w", encoding="utf-8").write("\n".join(lines))
        out.append(("owt (acc, nll)", O))
    # feynman
    Fy = collections.defaultdict(list)
    for d in load("results/feynman/*.json"):
        for m, r in d["results"].items():
            Fy[(d["eq"], d["N"], m)].append(r["test"] / d["sd_f"])
    if Fy:
        meths = ["single", "act", "label", "hybrid", "mlp", "kan"]
        for N in sorted({k[1] for k in Fy}):
            lines = [r"\begin{tabular}{@{}lr" + "r" * len(meths) + r"@{}}\toprule",
                     "equation & $n$ & " + " & ".join(meths) + r"\\\midrule"]
            eqs = sorted({k[0] for k in Fy if k[1] == N})
            for eq in eqs:
                n = next(d["n"] for d in load(f"results/feynman/{eq}_N{N}_s0.json")) if os.path.exists(f"results/feynman/{eq}_N{N}_s0.json") else "?"
                vals = [np.mean(Fy[(eq, N, m)]) if Fy[(eq, N, m)] else None for m in meths]
                best = min(v for v in vals if v is not None)
                lines.append(f"{eq} & {n} & " + " & ".join(
                    "--" if v is None else (f"\\textbf{{{v:.3f}}}" if v == best else f"{v:.3f}") for v in vals) + r"\\")
            lines.append(r"\bottomrule\end{tabular}")
            open(f"{TAB}/feynman_N{N}.tex", "w", encoding="utf-8").write("\n".join(lines))
        out.append(("feynman (RMSE/sd)", Fy))
    return out


def main():
    fig_weighting()
    T = toy_table()
    if T:
        fig_toys(T)
        tab_toys(T)
    fig_continual()
    fig_cells()
    real = tab_real()
    print("== toys (RMSE / sd(f), mean over seeds)")
    for fn in ["sin", "j0", "expsin", "xy", "nested4", "exp100"]:
        for N in sorted({k[1] for k in T if k[0] == fn}):
            print(f"  {fn:8s} {N:6d}", "  ".join(f"{m}={np.mean(T[(fn, N, m)]):.3f}(n{len(T[(fn, N, m)])})"
                                                for m in sorted({k[2] for k in T if k[0] == fn and k[1] == N})))
    for name, R in real:
        print("==", name)
        for k in sorted(R, key=str):
            v = R[k]
            print("  ", k, np.round(np.mean(v, 0), 4), f"n={len(v)}")


if __name__ == "__main__":
    main()
