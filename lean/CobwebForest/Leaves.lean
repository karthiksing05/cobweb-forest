/-
Copyright (c) 2026 Cobweb Forest authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Cobweb Forest authors
-/
import Mathlib

/-!
# The count-weighted leaf mixture is a sum over all examples (theory: Thm. consistency)

Cobweb never deletes a leaf and, with exact matching, a leaf holds `n_c` copies of one
example (Lemma "leaf preservation").  Write `leaf i` for the leaf that holds example `i`
and `xl c` for the example stored in leaf `c`.  Then any count-weighted sum over leaves
equals the sum over the examples themselves (`leaf_sum_eq`), and so the count-weighted
leaf mixture equals the Nadaraya--Watson estimate on all examples seen (`leaf_mixture_eq_nw`),
whatever tree category utility built.
-/

open Finset BigOperators

namespace CobwebForest

variable {ι L α : Type*} [Fintype ι] [Fintype L] [DecidableEq L]

/-- Number of examples held by leaf `c`. -/
def count (leaf : ι → L) (c : L) : ℕ := (univ.filter (fun i => leaf i = c)).card

/-- A count-weighted sum over leaves is the sum over the examples. -/
theorem leaf_sum_eq (leaf : ι → L) (xs : ι → α) (xl : L → α)
    (hx : ∀ i, xs i = xl (leaf i)) (F : α → ℝ) :
    ∑ c, (count leaf c : ℝ) * F (xl c) = ∑ i, F (xs i) := by
  rw [← Finset.sum_fiberwise univ leaf (fun i => F (xs i))]
  apply Finset.sum_congr rfl
  intro c _
  have h : ∀ i ∈ univ.filter (fun i => leaf i = c), F (xs i) = F (xl c) := by
    intro i hi
    rw [hx i, (Finset.mem_filter.mp hi).2]
  rw [Finset.sum_congr rfl h, Finset.sum_const, nsmul_eq_mul, count]

/-- **The count-weighted leaf mixture is the Nadaraya--Watson estimate.**  For any kernel
`K`, query `x`, and examples `(X i, Y i)` stored as copies in leaves holding `(xl c, yl c)`,
`∑ n_c K(x, xl c) yl c / ∑ n_c K(x, xl c) = ∑ K(x, X i) Y i / ∑ K(x, X i)`. -/
theorem leaf_mixture_eq_nw {β : Type*} (leaf : ι → L) (X : ι → β) (Y : ι → ℝ)
    (xl : L → β) (yl : L → ℝ) (hx : ∀ i, X i = xl (leaf i)) (hy : ∀ i, Y i = yl (leaf i))
    (K : β → ℝ) :
    (∑ c, (count leaf c : ℝ) * (K (xl c) * yl c)) / (∑ c, (count leaf c : ℝ) * K (xl c)) =
      (∑ i, K (X i) * Y i) / (∑ i, K (X i)) := by
  have hxy : ∀ i, (X i, Y i) = (xl (leaf i), yl (leaf i)) := fun i => by rw [hx i, hy i]
  have num := leaf_sum_eq leaf (fun i => (X i, Y i)) (fun c => (xl c, yl c)) hxy
    (fun p => K p.1 * p.2)
  have den := leaf_sum_eq leaf X xl hx K
  simp only at num
  rw [num, den]

end CobwebForest
