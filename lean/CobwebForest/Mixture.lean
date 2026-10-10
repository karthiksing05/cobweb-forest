/-
Copyright (c) 2026 Cobweb Forest authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Cobweb Forest authors
-/
import Mathlib

/-!
# Merging trees under a common root gives a mixture (theory: Prop. interweaving)

A tree's prediction is a normalised mixture `(∑ w_c μ_c) / (∑ w_c)` of its concepts' means.
Merge `m` trees under a new root of weight `w₀` and mean `μ₀`; under the `lik` weighting every
concept keeps its weight (under `path` every weight of tree `j` is multiplied by the same
factor, absorbed into `Z j` below).  The merged prediction is then the convex combination of the
root's mean and the subtrees' own predictions -- a mixture, never a composition.
-/

open Finset BigOperators

namespace CobwebForest

/-- **Interweaving.**  With `Z j > 0` the total weight of tree `j` and `M j` its prediction, the
merged tree predicts `λ₀ μ₀ + ∑ λ_j M_j` with `λ₀ = w₀ / T`, `λ_j = Z j / T`,
`T = w₀ + ∑ Z j`. -/
theorem merged_prediction {m : ℕ} (w₀ μ₀ : ℝ) (Z M : Fin m → ℝ) (hw₀ : 0 < w₀)
    (hZ : ∀ j, 0 < Z j) :
    (w₀ * μ₀ + ∑ j, Z j * M j) / (w₀ + ∑ j, Z j) =
      (w₀ / (w₀ + ∑ j, Z j)) * μ₀ + ∑ j, (Z j / (w₀ + ∑ j, Z j)) * M j := by
  have hT : 0 < w₀ + ∑ j, Z j :=
    add_pos_of_pos_of_nonneg hw₀ (Finset.sum_nonneg fun j _ => (hZ j).le)
  rw [add_div, Finset.sum_div]
  congr 1
  · ring
  · apply Finset.sum_congr rfl; intro j _; ring

/-- The weights of the mixture are nonnegative and sum to one. -/
theorem merged_weights {m : ℕ} (w₀ : ℝ) (Z : Fin m → ℝ) (hw₀ : 0 < w₀) (hZ : ∀ j, 0 < Z j) :
    0 ≤ w₀ / (w₀ + ∑ j, Z j) ∧ (∀ j, 0 ≤ Z j / (w₀ + ∑ j, Z j)) ∧
    w₀ / (w₀ + ∑ j, Z j) + ∑ j, Z j / (w₀ + ∑ j, Z j) = 1 := by
  have hT : 0 < w₀ + ∑ j, Z j :=
    add_pos_of_pos_of_nonneg hw₀ (Finset.sum_nonneg fun j _ => (hZ j).le)
  refine ⟨div_nonneg hw₀.le hT.le, fun j => div_nonneg (hZ j).le hT.le, ?_⟩
  rw [← Finset.sum_div, ← add_div, div_self (ne_of_gt hT)]

end CobwebForest
