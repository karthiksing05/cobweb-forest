/-
Copyright (c) 2026 Cobweb Forest authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Cobweb Forest authors
-/
import Mathlib

/-!
# Structural statements about Cobweb forests

* `fanin_one` (theory: Prop. "two inputs are necessary"): a forest in which every tree reads
  at most one input computes a function of at most one raw coordinate.
* `ka_tree_count` (theory: Thm. Cobweb–Kolmogorov–Arnold): the KA forest has `4n² + 4n` trees.
-/

namespace CobwebForest

/-- A forest in which every tree has fan-in at most one, over inputs `x : Fin n → ℝ`.
`const` reads nothing, `raw i g` reads coordinate `i`, `comp t g` reads the message of the
tree `t`; `g` is the map the tree realises. -/
inductive Chain (n : ℕ) where
  | const (c : ℝ)
  | raw (i : Fin n) (g : ℝ → ℝ)
  | comp (t : Chain n) (g : ℝ → ℝ)

namespace Chain

/-- The message the forest computes on input `x`. -/
def eval {n : ℕ} : Chain n → (Fin n → ℝ) → ℝ
  | const c, _ => c
  | raw i g, x => g (x i)
  | comp t g, x => g (eval t x)

/-- **Fan-in one is not enough.**  Every fan-in-one forest is a function of one coordinate. -/
theorem fanin_one {n : ℕ} (hn : 0 < n) (t : Chain n) :
    ∃ i : Fin n, ∃ h : ℝ → ℝ, ∀ x, eval t x = h (x i) := by
  induction t with
  | const c => exact ⟨⟨0, hn⟩, fun _ => c, fun x => rfl⟩
  | raw i g => exact ⟨i, g, fun x => rfl⟩
  | comp t g ih =>
    obtain ⟨i, h, hh⟩ := ih
    exact ⟨i, g ∘ h, fun x => by simp [eval, hh x]⟩

end Chain

/-- Tree count of the Cobweb–Kolmogorov–Arnold forest:
`n(2n+1)` inner trees, `(2n+1)(n-1)` inner adders, `2n+1` outer trees, `2n` outer adders. -/
theorem ka_tree_count (n : ℕ) (hn : 1 ≤ n) :
    n * (2 * n + 1) + (2 * n + 1) * (n - 1) + (2 * n + 1) + 2 * n = 4 * n ^ 2 + 4 * n := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  simp only [Nat.add_sub_cancel]
  ring

end CobwebForest
