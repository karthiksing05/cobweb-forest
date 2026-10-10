# Cobweb Forests — a Kolmogorov–Arnold theory for Cobweb

Can a network of Cobweb trees represent any continuous function (Q1)? Can Cobweb's own
learning rule, with category utility and no gradients, learn it (Q2)? The theory is in
[`theory/main.tex`](theory/main.tex) (build: `tectonic -X compile theory/main.tex` with the
`tex` conda env). The experiments test each statement, in the style of the KAN paper.

## Layout

| path | what |
|---|---|
| `csrc/` | the hybrid Cobweb of `cobweb-private` (commit in `csrc/UPSTREAM_COMMIT`), forest iteration: NaN-masked continuous queries, integer node `uid`s (labels), `expanded()`, `predict_mean`, `leaves_only` mixtures. `tests/test_core_equivalence.py` checks it reproduces upstream trees and predictions exactly |
| `cobweb_forest/forest.py` | `TreeSpec`, `Tree`, `Forest`: trees that pass **activations** (multi-node predictions of their task attribute) and **labels** (concept uids at a depth); `online` / `split` / `stage` training protocols |
| `cobweb_forest/designs.py` | single tree; KA-graph forests (`act`, `label`, `hybrid` messages); bandwidth schedules |
| `cobweb_forest/baselines.py` | MLP and B-spline KAN baselines |
| `experiments/` | one driver per claim (see below); `make_figures.py` builds `theory/figs`, `theory/tables` |
| `scripts/` | `cpu.sbatch` (overcap CPU), `snapshot.sh` (each sweep runs from a frozen copy in `runs/`), `submit_toys.sh` |

## Experiments → claims

| driver | claim tested |
|---|---|
| `weighting.py` | Thm 1 rate; Prop 2 (prior/posterior all-concept mixtures are inconsistent) |
| `toys.py` | KAN toys (sin, J0, exp(sin+y²), xy, 4-D, 100-D): Thm CKA, Thm forest consistency, Prop xy, the crossover corollary |
| `continual.py` | leaf preservation → no forgetting (KAN Fig. 3.4 task) |
| `label_cells.py` | assumption (a) of the label-forest consistency theorem |
| `interpret.py` | the univariate trees learn the KA inner functions |
| `feynman.py`, `housing.py`, `mnist.py`, `owt.py` | real-world-style benchmarks |

Build the extension with `cmake -S . -B build -G Ninja -DPython_EXECUTABLE=<env>/bin/python && cmake --build build`
(Eigen 3.4 and rapidjson are vendored under `third_party/`). Do not rebuild while jobs are running.
