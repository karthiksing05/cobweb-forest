/-
Copyright (c) 2026 Cobweb Forest authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Cobweb Forest authors
-/
import CobwebForest.Leaves
import CobwebForest.UnivTree

/-!
# A learned tree is the Nadaraya–Watson estimate (theory: Thm. learn-tree, Thm. consistency)

Cobweb never deletes a leaf, and with exact matching and no depth limit every example ends in a
leaf that holds only copies of it (Lem. leaf preservation, a statement about the C++ algorithm,
checked by the executable verifier).  In Lean this is the hypothesis that example `i` is
stored in the leaf `leaf i`, which holds `(xl c, yl c)`: `X i = xl (leaf i)`, `Y i = yl (leaf i)`.

* `learned_tree_eq_nw` (Thm. learn-tree (a)): the count-weighted leaf mixture of eq. (1) of the
  learned tree equals the Nadaraya–Watson estimate with the Gaussian kernel on all examples
  seen, whatever hierarchy category utility built and whatever the order of the examples.
* `learned_trees_inherit` (Thm. learn-tree (b), (c)): every statement about the sequence of
  Nadaraya–Watson estimates -- the cited strong `L₁` consistency (Devroye & Krzyżak 1989) under
  `h_N → 0`, `N h_N^d → ∞`, or the cited interior rate `N^{-2/(d+4)}` -- holds for the sequence
  of learned trees, which are the same functions.  The cited theorems enter as the hypothesis.
-/

open Finset

namespace CobwebForest

variable {ι L E : Type*} [Fintype ι] [Fintype L] [DecidableEq L] [PseudoMetricSpace E]

/-- **Thm. learn-tree (a).**  If example `i` lies in leaf `leaf i` and every leaf `c` holds
copies of one example `(xl c, yl c)`, the count-weighted leaf mixture (eq. (1)) of the learned
tree is the Nadaraya–Watson estimate with the Gaussian kernel on all the examples. -/
theorem learned_tree_eq_nw (leaf : ι → L) (X : ι → E) (Y : ι → ℝ) (xl : L → E) (yl : L → ℝ)
    (hx : ∀ i, X i = xl (leaf i)) (hy : ∀ i, Y i = yl (leaf i)) (h : ℝ) (x : E) :
    leafMixture univ xl yl (fun c => (count leaf c : ℝ)) h x =
      (∑ i, gauss h (dist x (X i)) * Y i) / ∑ i, gauss h (dist x (X i)) := by
  have key := leaf_mixture_eq_nw leaf X Y xl yl hx hy (fun z => gauss h (dist x z))
  unfold leafMixture
  simp only [mul_assoc]
  exact key

/-- **Thm. learn-tree (b), (c).**  For every sample size `N` let the learned tree store the
examples `(X N i, Y N i)` as in (a).  Every property `Prop` of the sequence of Nadaraya–Watson
estimates (the cited consistency or rate) holds for the sequence of learned trees. -/
theorem learned_trees_inherit {ιN LN : ℕ → Type*} [∀ N, Fintype (ιN N)] [∀ N, Fintype (LN N)]
    [∀ N, DecidableEq (LN N)] (leaf : ∀ N, ιN N → LN N) (X : ∀ N, ιN N → E)
    (Y : ∀ N, ιN N → ℝ) (xl : ∀ N, LN N → E) (yl : ∀ N, LN N → ℝ)
    (hx : ∀ N i, X N i = xl N (leaf N i)) (hy : ∀ N i, Y N i = yl N (leaf N i)) (h : ℕ → ℝ)
    (Good : (ℕ → E → ℝ) → Prop)
    (hNW : Good fun N x =>
      (∑ i, gauss (h N) (dist x (X N i)) * Y N i) / ∑ i, gauss (h N) (dist x (X N i))) :
    Good fun N x => leafMixture univ (xl N) (yl N) (fun c => (count (leaf N) c : ℝ)) (h N) x := by
  have heq : (fun N x => leafMixture univ (xl N) (yl N) (fun c => (count (leaf N) c : ℝ)) (h N) x)
      = fun N x =>
        (∑ i, gauss (h N) (dist x (X N i)) * Y N i) / ∑ i, gauss (h N) (dist x (X N i)) := by
    funext N x
    exact learned_tree_eq_nw (leaf N) (X N) (Y N) (xl N) (yl N) (hx N) (hy N) (h N) x
  rw [heq]
  exact hNW

end CobwebForest
