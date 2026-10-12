# CobwebForest (Lean 4 + Mathlib)

Formal proofs of the statements of `theory/main.tex`. Build:

    export ELAN_HOME=/coc/flash10/ksingara3/.elan PATH=$ELAN_HOME/bin:$PATH
    cd lean && lake exe cache get && lake build

`verification/checks.py::check_lean` rebuilds, prints `#print axioms` for every theorem
below and fails on `sorryAx`, `sorry`, `admit` or a new `axiom`. `Audit.lean` prints the same
list. Every theorem depends only on `propext`, `Classical.choice` and `Quot.sound`.

Without the Mathlib cache (for example in a sandbox where `cache.mathlib.org` is blocked),
`lake build` compiles Mathlib from source; the toolchain also installs from the GitHub
release `lean-4.35.0-rc4-linux.tar.zst` unpacked into
`$ELAN_HOME/toolchains/leanprover--lean4---v4.35.0-rc4`.

## Modelling choices

- **Leaf mixture.** `leafMixture S γ y n h x` is eq. (1): leaves `s ∈ S` hold `n s` copies of
  `(γ s, y s)`, Gaussian weights `exp(-d(x, γ s)² / 2h²)`, any (pseudo)metric space. A
  `k`-input tree uses `EuclideanSpace ℝ (Fin k)` (`IsCobwebTree k G φ`: nonempty leaves, at
  most `G` of them, positive counts, `h > 0`).
- **Forests and compositions.** `Net n` is a syntax tree: `raw i` reads coordinate `i`;
  `node k args g` applies `g : (Fin k → ℝ) → ℝ` to `k` sub-forests. A DAG is its unfolding
  (shared vertices repeated, which changes no value and no depth). `Fin k → ℝ` carries the
  max-norm, so `cthickening r B` is the paper's `B^r`. `depth` counts non-source vertices on
  the longest path; `numTrees` counts non-source vertices.
- **Probability.** The learning statements are proved on finite probability spaces (`Dist`,
  `condExp` of `Cues.lean`). All probability sits in the hypotheses, as in the paper's
  deterministic forms.
- **Cited theorems** enter as hypotheses: the Kolmogorov–Arnold theorem (`cobweb_ka`),
  Nadaraya–Watson consistency and rates (`learned_trees_inherit`), the uniform kernel-mass laws
  (`step2_bound`), the single-tree uniform rate (`learn_rate`, hypothesis (iii)).

## Paper statement → Lean

| paper | Lean | file |
|---|---|---|
| Thm. univ-tree (one tree is universal; any counts in `[nmin, nmax]`) | `univ_tree`, `univ_tree_counts` | UnivTree.lean |
| Prop. rate (grid rate, `C(d,β) = 2d e^{1/2β²}/(1-e^{-1/2β²})²`) | `grid_rate` | Rate.lean |
| Lem. stable (stability of composition) | `Net.stable` | Forest.lean |
| Thm. univ-forest (fan-in 2 forests are universal) | `Net.univ_forest` | Forest.lean |
| Thm. compose (core induction; full form with grid trees and McShane extensions) | `Net.compose`, `Net.compose_grid` | Compose.lean, ComposeGrid.lean |
| Thm. separation (a) (no single tree beats `G^{-1/n}/4`; the bump is in `𝓑_n`) | `separation_a`, `bump_mem_B` | Separation.lean |
| Thm. separation (b) (`≤ 2^D - 1 ≤ 8n - 1` trees, error `C D (1+2r) G^{-1/2}`) | `separation_b`, `Net.numTrees_le` | ComposeGrid.lean, Separation.lean |
| Thm. cka (KA theorem as hypothesis; exactly `4N² + 4N` trees) | `cobweb_ka`, `Net.numTrees_kaNet` | KA.lean |
| Thm. learn-tree (a) (learned tree = Nadaraya–Watson, eq. (1)); (b), (c) cited | `learned_tree_eq_nw`, `learned_trees_inherit` | LearnTree.lean |
| Thm. ideal, Step 2 (perturbed design: kernel ratios, far points, averages) | `gauss_ratio`, `gauss_far_perturbed`, `wavg_perturb`, `step2_bound` | Perturb.lean |
| Thm. learn-rate (deterministic form: `e_out ≤ r ∑_{l≤D} (2L)^l`) | `learn_rate`, `condExp_tower` | LearnRate.lean |
| Prop. tasks (b) = Prop. xy (funnel tasks can fail: `F ≡ 0`, `Δ = Var Y`) | `funnel_xy` | Tasks.lean |
| sufficiency criterion (last step of Prop. multiview, Prop. suffclass) | `condExp_eq_of_factor` | Tasks.lean |
| Prop. xy, Cor. parity (uninformative input gives a constant message) | `validity_nonneg`, `condExp_const_of_validity_zero` | Cues.lean |
| (tool only, not in paper) blocking | `condExp_blocked`, `conditional_validity_blocked` | Cues.lean |
| Thm. locality (i)-(ii), sign skeleton | `sign_worlds_local_law`, `local_rule_same`, `sign_worlds_differ` | Locality.lean |
| Cor. parity (J = all bits) | `parity_fibre_balanced` | Locality.lean |
| Thm. consistency (leaf mixture = Nadaraya-Watson) | `leaf_sum_eq`, `leaf_mixture_eq_nw` | Leaves.lean |
| Prop. interweave | `merged_prediction`, `merged_weights` | Mixture.lean |
| Prop. fan-in | `Chain.fanin_one` | Structure.lean |
| Thm. CKA tree count | `ka_tree_count` | Structure.lean |

Not formalised: the Kolmogorov–Arnold theorem itself, the probabilistic laws cited above,
the Gaussian computation of Prop. multiview (its last step is `condExp_eq_of_factor`), the
independence computation of Prop. suffclass, and Thm. labelcons (cited partitioning estimate).

`cobweb_forest/cues.py` is a design tool outside the paper; `check_cues` tests it.
