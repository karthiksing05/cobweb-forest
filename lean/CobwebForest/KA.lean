/-
Copyright (c) 2026 Cobweb Forest authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Cobweb Forest authors
-/
import CobwebForest.ComposeGrid

/-!
# The Cobweb–Kolmogorov–Arnold forest (theory: Thm. cka, Rem. ka-main)

The Kolmogorov–Arnold theorem -- every continuous `f` on `[0,1]^N` is
`∑_{q=0}^{2N} Φ_q(∑_{p=1}^N φ_{q,p}(x_p))` with continuous `φ_{q,p}`, `Φ_q` -- is not in
Mathlib.  It enters `cobweb_ka` as a hypothesis (a cited theorem).  The rest of Thm. cka is
proved: the representation is a forest (`kaNet`) of `N(2N+1)` univariate inner trees,
`(2N+1)(N-1)` two-input adders for the inner sums, `2N+1` univariate outer trees and `2N`
adders for the outer sum, `4N² + 4N` trees in all (`numTrees_kaNet`), and Lem. stable with
Thm. univ-tree turns it into a Cobweb forest with exactly these trees, within `ε` of `f`.
-/

open Finset

namespace CobwebForest

namespace Net

variable {n : ℕ}

/-- A chain of `k` two-input adders over `k + 1` summands. -/
def chainSum : (k : ℕ) → (Fin (k + 1) → Net n) → Net n
  | 0, f => f 0
  | k + 1, f => node 2 ![chainSum k (fun i => f i.castSucc), f (Fin.last (k + 1))] addMap

theorem eval_chainSum (x : Fin n → ℝ) :
    ∀ (k : ℕ) (f : Fin (k + 1) → Net n), eval (chainSum k f) x = ∑ i, eval (f i) x
  | 0, f => by simp [chainSum]
  | k + 1, f => by
    rw [chainSum, eval_node, Fin.sum_univ_castSucc, ← eval_chainSum x k]
    simp [addMap]

theorem numTrees_chainSum :
    ∀ (k : ℕ) (f : Fin (k + 1) → Net n), numTrees (chainSum k f) = k + ∑ i, numTrees (f i)
  | 0, f => by simp [chainSum]
  | k + 1, f => by
    rw [chainSum, numTrees_node, Fin.sum_univ_two, Fin.sum_univ_castSucc]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    rw [numTrees_chainSum k]
    ring

theorem allCont_chainSum :
    ∀ (k : ℕ) (f : Fin (k + 1) → Net n), (∀ i, AllCont (f i)) → AllCont (chainSum k f)
  | 0, _, hf => hf 0
  | k + 1, _, hf => .node continuous_addMap
      (Fin.forall_fin_two.mpr ⟨allCont_chainSum k _ fun _ => hf _, hf _⟩)

theorem fanIn_chainSum :
    ∀ (k : ℕ) (f : Fin (k + 1) → Net n), (∀ i, FanInLE 2 (f i)) → FanInLE 2 (chainSum k f)
  | 0, _, hf => hf 0
  | k + 1, _, hf => .node le_rfl
      (Fin.forall_fin_two.mpr ⟨fanIn_chainSum k _ fun _ => hf _, hf _⟩)

/-- The Kolmogorov–Arnold forest on `N = n + 1` inputs: inner trees `φ q p` on the coordinates,
chains of adders for the `2N + 1` inner sums, outer trees `Φ q`, a chain of adders for the outer
sum. -/
def kaNet (φ : Fin (2 * (n + 1) + 1) → Fin (n + 1) → ℝ → ℝ)
    (Φ : Fin (2 * (n + 1) + 1) → ℝ → ℝ) : Net (n + 1) :=
  chainSum (2 * (n + 1)) fun q =>
    node 1 ![chainSum n fun p => node 1 ![raw p] fun u => φ q p (u 0)] fun u => Φ q (u 0)

theorem eval_kaNet (φ : Fin (2 * (n + 1) + 1) → Fin (n + 1) → ℝ → ℝ)
    (Φ : Fin (2 * (n + 1) + 1) → ℝ → ℝ) (x : Fin (n + 1) → ℝ) :
    eval (kaNet φ Φ) x = ∑ q, Φ q (∑ p, φ q p (x p)) := by
  rw [kaNet, eval_chainSum]
  refine Finset.sum_congr rfl fun q _ => ?_
  rw [eval_node, inVec_apply, Matrix.cons_val_zero, eval_chainSum]
  congr 1

/-- **Tree count** (`ka_tree_count` in `Structure.lean`): `4N² + 4N` trees. -/
theorem numTrees_kaNet (φ : Fin (2 * (n + 1) + 1) → Fin (n + 1) → ℝ → ℝ)
    (Φ : Fin (2 * (n + 1) + 1) → ℝ → ℝ) :
    numTrees (kaNet φ Φ) = 4 * (n + 1) ^ 2 + 4 * (n + 1) := by
  rw [kaNet, numTrees_chainSum]
  have hinner : ∀ q : Fin (2 * (n + 1) + 1), numTrees
      (node 1 ![chainSum n fun p => node 1 ![raw p] fun u => φ q p (u 0)] fun u => Φ q (u 0)) =
      2 * n + 2 := by
    intro q
    rw [numTrees_node, Fin.sum_univ_one, Matrix.cons_val_zero, numTrees_chainSum]
    have : ∀ p : Fin (n + 1), numTrees (node 1 ![raw p] fun u => φ q p (u 0)) = 1 := by
      intro p
      rw [numTrees_node, Fin.sum_univ_one, Matrix.cons_val_zero]
      rfl
    simp only [this, Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul, mul_one]
    ring
  simp only [hinner, Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
  ring

end Net

open Net

/-- **Thm. cka (Cobweb–Kolmogorov–Arnold), with the Kolmogorov–Arnold theorem as hypothesis.**
Let `f` have a Kolmogorov–Arnold representation on `[0,1]^N`, `N = n + 1`, with continuous
inner and outer functions (the KA theorem provides one for every continuous `f`).  For every
`ε > 0` there is a Cobweb forest in which every tree has at most two inputs, consisting of
exactly `4N² + 4N` trees, whose output is within `ε` of `f` on `[0,1]^N`.  The number of trees
depends neither on `f` nor on `ε`; only the trees' leaves do. -/
theorem cobweb_ka {n : ℕ} {f : (Fin (n + 1) → ℝ) → ℝ}
    (hKA : ∃ (φ : Fin (2 * (n + 1) + 1) → Fin (n + 1) → ℝ → ℝ)
      (Φ : Fin (2 * (n + 1) + 1) → ℝ → ℝ), (∀ q p, Continuous (φ q p)) ∧
        (∀ q, Continuous (Φ q)) ∧
        ∀ x ∈ Set.Icc (0 : Fin (n + 1) → ℝ) 1, f x = ∑ q, Φ q (∑ p, φ q p (x p)))
    {ε : ℝ} (hε : 0 < ε) :
    ∃ t : Net (n + 1), (∃ G, IsCobweb G t) ∧ FanInLE 2 t ∧
      numTrees t = 4 * (n + 1) ^ 2 + 4 * (n + 1) ∧
      ∀ x ∈ Set.Icc (0 : Fin (n + 1) → ℝ) 1, |eval t x - f x| < ε := by
  obtain ⟨φ, Φ, hφ, hΦ, hf⟩ := hKA
  -- the exact forest
  have hinner : ∀ q, AllCont (chainSum n fun p => node 1 ![raw p] fun u => φ q p (u 0)) :=
    fun q => allCont_chainSum n _ fun p =>
      .node ((hφ q p).comp (continuous_apply 0)) (Fin.forall_fin_one.mpr (.raw p))
  have hcont : AllCont (kaNet φ Φ) :=
    allCont_chainSum _ _ fun q =>
      .node ((hΦ q).comp (continuous_apply 0)) (Fin.forall_fin_one.mpr (hinner q))
  have hfan : FanInLE 2 (kaNet φ Φ) :=
    fanIn_chainSum _ _ fun q => .node (by norm_num) (Fin.forall_fin_one.mpr
      (fanIn_chainSum _ _ fun p => .node (by norm_num) (Fin.forall_fin_one.mpr (.raw p))))
  -- stability, then a Cobweb tree at every vertex
  have hK : IsCompact (Set.Icc (0 : Fin (n + 1) → ℝ) 1) := isCompact_Icc
  obtain ⟨η, hη, hstab⟩ := stable hK one_pos (hcont.contNear _ 1) hε
  obtain ⟨t, G, htc, hclose⟩ := exists_cobweb_close hK hη _ (hcont.contNear _ 1)
  refine ⟨t, ⟨G, htc⟩, hclose.fanInLE hfan, ?_, fun x hx => ?_⟩
  · rw [hclose.numTrees_eq, numTrees_kaNet]
  · rw [hf x hx, ← eval_kaNet φ Φ x]
    exact hstab t hclose x hx

end CobwebForest
