# Formalization handoff: Cobweb forests are universal approximators

This document is for an autoformalization session (Lean 4 + Mathlib). It lists the objects,
the exact statements, the proof steps, and the Mathlib facts each step needs, in dependency
order. Paper labels are those of `theory/main.tex`. Statements marked **[core]** are the
universality results. Statements marked **[cited]** rely on external theorems; formalize the
reduction and state the cited theorem as a hypothesis.

The existing Lean project `lean/` (Mathlib v4.35.0-rc4; build with
`ELAN_HOME=/coc/flash10/ksingara3/.elan lake build`) already proves:

- `leaf_sum_eq` and `leaf_mixture_eq_nw` (the leaf mixture equals Nadaraya–Watson);
- `merged_prediction` and `merged_weights` (merged trees give a mixture);
- `Chain.fanin_one` (fan-in one is not enough);
- `ka_tree_count`;
- `sign_worlds_*` and `parity_fibre_balanced` (locality);
- `condExp_const_of_validity_zero`.

Reuse them.

---

## 0. Objects

**Leaf mixture (Cobweb tree prediction, eq. (1)).** A finite set `S` of leaves, a point
`γ s ∈ E` and a value `y s ∈ ℝ` for each leaf, counts `n s > 0`, and a bandwidth `h > 0`:

    ŷ(x) = (∑ s∈S, n s · exp(-‖x - γ s‖² / (2h²)) · y s) / (∑ s∈S, n s · exp(-‖x - γ s‖² / (2h²)))

- `E` is `EuclideanSpace ℝ (Fin d)`. Any metric space works for 1.1, with `dist`.
- This is exactly the count-weighted leaves-only prediction of a depth-one Cobweb tree whose
  leaves hold exact copies. The Gaussian normalising constants are equal and cancel.
- Faithfulness to the C++ code is checked by the executable verifier, not by Lean.

**Forest of fan-in ≤ 2.** An inductive syntax with constructors `raw i | const c | node1 t φ |
node2 t₁ t₂ φ`, with `eval` defined by recursion. A "Cobweb forest" is a forest whose every
node map is a leaf mixture (one or two inputs).

**k-composition of depth D.** A DAG (or the inductive syntax with fan-in ≤ k) whose node maps
are `g_v : (Fin k → ℝ) → ℝ`. Depth is the number of non-source vertices on the longest
source-to-output path.

---

## 1. Representation

### 1.1 `thm:univ-tree` **[core]**: one tree is a universal approximator

**Statement.**

    ∀ K compact ⊆ E, ∀ g ContinuousOn K, ∀ ε > 0,
      ∃ S : Finset E, S ⊆ K, ∃ h > 0, ∀ x ∈ K, |ŷ_{S, g, 1, h}(x) - g x| < ε

Counts are all 1 and `y = g`.

**Variant.** Arbitrary counts with `0 < n_min ≤ n s ≤ n_max`. The same proof works, with
`|S|` replaced by `|S|·n_max/n_min`.

**Proof steps.**
1. `M := sup_K |g|`, finite. Mathlib: `IsCompact.exists_bound_of_continuousOn`.
2. Uniform continuity on `K` gives `ρ > 0` with `x, z ∈ K`, `dist x z < ρ` ⇒ `|g x - g z| < ε/2`.
   Mathlib: `IsCompact.uniformContinuousOn_of_continuous`, then `Metric.uniformContinuousOn_iff`.
3. Finite `ρ/2`-net `S ⊆ K`. Mathlib: `IsCompact.finite_cover_balls` (or
   `finite_approx_of_totallyBounded` with `IsCompact.totallyBounded`).
4. Fix `x ∈ K`, and `s₀ ∈ S` with `dist x s₀ < ρ/2`.
   - Write `a s = exp(-‖x-s‖²/(2h²))`. Then `A := ∑ a ≥ a s₀ > exp(-ρ²/(8h²))`.
   - A far leaf (`dist ≥ ρ`) has `a s ≤ exp(-ρ²/(2h²))`.
   - So the far share is `≤ |S| · exp(-3ρ²/(8h²))`. Use `Real.exp_le_exp` and `Real.exp_add`.
5. Since the weights `w s = a s / A` sum to 1,
   `|ŷ - g x| ≤ ∑ w s |g s - g x|`.
   - Split the sum into near and far leaves (`Finset.sum_filter_add_sum_filter_not`).
   - Near terms are `< ε/2` (step 2). Far terms are `≤ 2M`.
6. Choose `h` with `2M|S|exp(-3ρ²/(8h²)) ≤ ε/2`. Such an `h` exists because
   `exp(-c/h²) → 0` as `h → 0⁺`, e.g. `h = ρ·sqrt(3/(8·log(4M|S|/ε + 2)))`.
   - Edge case: `M = 0` (any `h` works).
   - Edge case: `K = ∅` (vacuous).

### 1.2 `prop:rate`: grid rate for Lipschitz `g`

For formalization, take the cleaner statement used by `thm:compose`. Leaves hold
`(γ, g̃ γ)`, where `g̃` is the McShane extension of `g` from `K` to the cube
(`LipschitzWith.extend_real` in Mathlib); prediction is the count-weighted leaf mixture:

    g L-Lipschitz on K ⊆ cube Q of side w, grid of m^d cell centres, h = β·w/m
      ⇒ ∀ x ∈ Q, |ŷ(x) - g̃ x| ≤ C(d, β) · L · w · m^{-1}

**Proof** (appendix `thm:single`):
- `|ŷ - g x| ≤ L ∑ w_γ ‖γ - x‖`.
- Bound `∑ e^{-‖x-γ‖²/2h²} ‖x-γ‖` above, and `∑ e^{-‖x-γ‖²/2h²}` below by the nearest grid
  point. Both are comparisons with Gaussian sums over a lattice of spacing `β` in units of `h`.

This is the most tedious part of the formalization. An acceptable weaker version: any
explicit `C(d, β)`, proved by splitting into annuli.

### 1.3 `lem:stable` (appendix `lem:dag`)

For a DAG on a compact `K`, where each `g_v` is continuous on an open set containing a closed
neighbourhood of its input set `B_v`: for every `ε` there are tolerances `ρ_v, η_v` such that
replacing each `g_v` by any `ĝ_v` that is `η_v`-close on the closed `ρ_v`-neighbourhood of `B_v`
changes the output by `≤ ε` on `K`. Downstream, use a tolerance slightly below `ε/2` to get
strict inequalities.

**Proof.** Choose tolerances in reverse topological order. At each node use uniform
continuity of `g_v` on the compact `K_v + closedBall 0 ρ_v`.

**Formalization tip.** Do induction on the inductive forest syntax. Strengthen the hypothesis
to

    ∀ ε > 0, ∃ δ > 0, ∃ R, ∀ forest t' of the same shape, (node maps δ-close on boxes of radius R)
      → sup_K |eval t' - eval t| < ε

### 1.4 `thm:univ-forest` **[core]**: fan-in-2 forests are universal

**Statement.**

    ∀ K compact ⊆ (Fin n → ℝ), ∀ f ContinuousOn K, ∀ ε > 0,
      ∃ forest t with fan-in ≤ 2 whose node maps are leaf mixtures,
        ∀ x ∈ K, |eval t x - f x| < ε

**Proof.**
1. **Stone–Weierstrass.** Polynomials are dense in `C(K, ℝ)`. Mathlib:
   `ContinuousMap.subalgebra_topologicalClosure_eq_top_of_separatesPoints`, applied to the
   subalgebra generated by the coordinate functions (they separate points). Alternatively,
   define the subalgebra of functions computed by exact `+`/`×`/const/raw forests directly;
   it is closed under `+`, `×` and constants by construction.
2. Pick an exact forest `t` with `sup_K |eval t - f| < ε/2`.
3. Apply 1.3 with a tolerance below `ε/2`. Then apply 1.1 at each node. The node map is
   `(a,b) ↦ a+b`, `a·b`, or `a ↦ a²`. The last is needed when both arguments are the same
   value, because a forest tree reads each parent once. Each is continuous on the compact box,
   so a leaf mixture is `η_v`-close there.
   - A constant node is `const`.
   - If `p` is a coordinate or a constant, add one identity node.
4. Use the triangle inequality.

### 1.5 `prop:fanin-main`

Already in Lean (`Chain.fanin_one`).

### 1.6 `rem:ka-main`

Optional; it needs the Kolmogorov–Arnold theorem, which is not in Mathlib. The tree count is
in Lean (`ka_tree_count`).

### 1.7 `thm:compose` **[core]**: composition beats dimension

**Definitions.**
- Vertex `v` has `k_v ≤ k` inputs.
- The depth `d(v)` is the number of non-source vertices on the longest source-to-`v` path;
  `D = max d(v)`.
- Neighbourhoods are in the max-norm: `B^r = {u | infDist_∞(u, B) ≤ r}`.

**Statement** (restructured for Lean after review round 6).
- Hypotheses:
  - each `g_v` is `L`-Lipschitz (max-norm) on `B_v^r ⊆ Q_v`, a cube of side `w`;
  - the tree `T_v` is the `m^{k_v}` grid tree of 1.2 for the McShane extension `g̃_v` on
    `Q_v`, so `sup_{Q_v} |T_v - g̃_v| ≤ η := C L w / m`;
  - every vertex error stays `≤ r`. This is implied by `η ∑_{l<D} L^l ≤ r`, and can also be
    checked a posteriori.
- Conclusion: `∀ v, sup_K |û_v - u_v| ≤ η ∑_{l<d(v)} L^l`.

**Proof.** Induction on `d(v)`:

    |T_v(û) - g_v(u)| ≤ |T_v(û) - g̃_v(û)| + |g̃_v(û) - g̃_v(u)| ≤ η + L·e

Here `û ∈ B_v^r` because each coordinate moved by at most `e ≤ r` (max-norm). Pure arithmetic.

**Note.** Only the exponent is free of `n`. The constant `∑ L^l` grows with the depth, which
is at least `log_k n`.

---

## 2. Learning

### 2.1 `thm:learn-tree`

- **(a) Identity.** Already in Lean (`leaf_mixture_eq_nw`). Formalize the bridge statement:
  "Cobweb never deletes a leaf and leaves hold copies" is about the C++ algorithm (appendix
  `lem:leaves`). State it as the hypothesis `hx : ∀ i, X i = xl (leaf i)`.
- **Hypotheses** (after review round 6): i.i.d. examples, bounded `Y`, exact matching, and no
  depth limit (`depth_max = ∞`).
- **(b) Consistency.** **[cited]** Strong L1 consistency of Nadaraya–Watson
  (Devroye & Krzyżak 1989). State it as a hypothesis and conclude by (a).
- **(c) Rate.** **[cited]** `N^{-2/(d+4)}` on a compact `I` interior to the support. It needs
  `m ∈ C²` with bounded second derivatives, and a `C¹` density bounded below on a neighbourhood
  of `I`. The bias is `O(h²)` and the variance `O(1/(N h^d))`, uniformly on `I`. State it as a
  hypothesis.

### 2.2 `thm:learn-forest` (appendix `thm:ideal`)

Uses assumption (A5), the uniform kernel law up to the boundary, which is stated as an
assumption. The deterministic Step 2 (kernel-ratio bounds) is formalizable; the
probabilistic parts are the cited laws.

### 2.3 `thm:learn-rate` **[core of learning]**: learned forests converge at the fan-in rate

**Statement.** Use the deterministic form; it isolates all probability in the hypotheses.
- Hypotheses:
  - (i) `E[V_τ | X_{R(τ)}] = ψ_τ(U_τ)`;
  - (ii) `ψ_τ` is `L`-Lipschitz (max-norm);
  - (iii) on an event `Ω_N`, every tree satisfies
    `sup_{û ∈ Û_τ(Q)} |ψ̂_τ(û) - φ_τ(û)| ≤ r_N`, where `φ_τ(û) = E[V_τ | Û_τ = û]`.
- Conclusion: on `Ω_N`, `e_out ≤ r_N ∑_{l=0}^{D} (2L)^l`.

**Proof.**
1. **Tower property:** `φ_τ(û) = E[ψ_τ(U_τ) | Û_τ = û]`. Mathlib: `MeasureTheory.condexp_condexp_of_le`;
   `Û` is `σ(X_R)`-measurable.
2. Hence `|φ - ψ| ≤ L·e` pointwise, because `|U - Û| ≤ e` surely.
3. Apply the triangle inequality, then induction on level.

To avoid conditional-expectation API friction, the finite-probability-space version works
like `Cues.lean`. The general version needs `condexp` monotonicity
(`MeasureTheory.condexp_mono`) and the bound `|E[Z|G]| ≤ E[|Z| | G]`.

### 2.4 `prop:tasks`

- (a) appendix `prop:suffclass`: an algebraic induction with strictly monotone `H_S`.
  `StrictMono.injective` and the continuity of the inverse of a continuous strictly monotone
  function on an interval are needed (search Mathlib for the order-isomorphism of a strictly
  monotone continuous surjection onto an interval).
- (b) appendix `prop:xy`: `E[X₁X₂|X₁] = X₁ E X₂ = 0` by independence.
- (c) **[cited]**: uses the partitioning-estimate theorem (Györfi Thm 4.2), with bounded `Y`,
  exact matching and a count-weighted output tree.
- (d): only the `Δ = 0` half (multi-view redundancy) is a theorem. Review round 6 showed that
  "`Δ > 0` otherwise" is false in general: it fails when the views are as small as the latent.
  Do not formalize a `Δ > 0` claim.

### 2.5 `prop:local`

It is stated for a tree whose **only** input is `X_2`. Already in Lean in its discrete form (`sign_worlds_local_law`). The continuous form reduces
to it because the magnitudes are independent of the signs.

---

## 3. Suggested order

1. 1.1 `thm:univ-tree`
2. 1.3 `lem:stable` (syntax version)
3. 1.4 `thm:univ-forest`
4. 1.7 `thm:compose`
5. 2.3 `thm:learn-rate` (deterministic form)
6. 1.2 `prop:rate`

Items 1–5 are the paper's headline. Item 6 is the most technical (lattice Gaussian sums) and
can be stated with an unspecified constant first.
