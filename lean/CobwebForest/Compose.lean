/-
Copyright (c) 2026 Cobweb Forest authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Cobweb Forest authors
-/
import CobwebForest.Forest

/-!
# Composition beats dimension (theory: Thm. compose)

`f` is a *`k`-composition of depth `D`* when it is computed by a forest `t` (a `Net`) whose
every vertex applies a map `g_v` of `k_v ≤ k` inputs.  The depth `d(v)` of a vertex is the
number of non-source vertices on the longest source-to-`v` path (`Net.depth`), and
`D = depth t`.  Neighbourhoods are taken in the max-norm, which is the norm of `Fin k → ℝ`:
`B^r = cthickening r B`.

* `compose` (Thm. compose): if every `g_v` is `L`-Lipschitz on `B_v^r` and the forest `t'`
  replaces every `g_v` by a map within `η` of it on `B_v^r` (`Close K r η t' t`), and the margin
  `η ∑_{l<D} L^l ≤ r` holds, then `t'` is within `η ∑_{l<D} L^l` of `t` on `K`.  The bound for
  every vertex `v` is the same statement applied to the sub-forest rooted at `v`.

The trees of Prop. rate give `η = C L w / m` (`Rate.lean`).  Only the exponent is free of `n`:
the constant `∑_{l<D} L^l` grows with the depth.
-/

open Metric Finset

namespace CobwebForest

namespace Net

variable {n : ℕ}

/-- Depth: the number of non-source vertices on the longest path from a source to the output. -/
def depth : Net n → ℕ
  | raw _ => 0
  | node _ args _ => (Finset.univ.sup fun j => depth (args j)) + 1

theorem depth_node {k : ℕ} (args : Fin k → Net n) (g : (Fin k → ℝ) → ℝ) :
    depth (node k args g) = (Finset.univ.sup fun j => depth (args j)) + 1 := rfl

theorem depth_arg_le {k : ℕ} (args : Fin k → Net n) (j : Fin k) :
    depth (args j) ≤ Finset.univ.sup fun j => depth (args j) :=
  Finset.le_sup (f := fun j => depth (args j)) (Finset.mem_univ j)

theorem depth_node_le {k : ℕ} {args : Fin k → Net n} {g : (Fin k → ℝ) → ℝ} {d : ℕ}
    (h : ∀ j, depth (args j) ≤ d) : depth (node k args g) ≤ d + 1 := by
  rw [depth_node]
  exact Nat.add_le_add_right (Finset.sup_le fun j _ => h j) 1

/-- Every vertex map is `L`-Lipschitz (max-norm) on the closed `r`-neighbourhood of its input
set. -/
inductive LipNear (K : Set (Fin n → ℝ)) (r : ℝ) (L : NNReal) : Net n → Prop
  | raw (i : Fin n) : LipNear K r L (raw i)
  | node {k : ℕ} {args : Fin k → Net n} {g : (Fin k → ℝ) → ℝ}
      (hg : LipschitzOnWith L g (cthickening r (inSet K args)))
      (hargs : ∀ j, LipNear K r L (args j)) : LipNear K r L (node k args g)

/-- `1 + q ∑_{l<d} q^l = ∑_{l<d+1} q^l`. -/
theorem geom_succ (q : ℝ) (d : ℕ) :
    1 + q * ∑ l ∈ range d, q ^ l = ∑ l ∈ range (d + 1), q ^ l := by
  rw [Finset.sum_range_succ', pow_zero, Finset.mul_sum]
  have : ∀ l ∈ range d, q * q ^ l = q ^ (l + 1) := fun l _ => by ring
  rw [Finset.sum_congr rfl this]
  ring

theorem geom_mono {q : ℝ} (hq : 0 ≤ q) {a b : ℕ} (hab : a ≤ b) :
    ∑ l ∈ range a, q ^ l ≤ ∑ l ∈ range b, q ^ l :=
  Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono hab) fun _ _ _ => by positivity

theorem compose_aux {K : Set (Fin n → ℝ)} {r η : ℝ} {L : NNReal} (hη : 0 ≤ η) :
    ∀ t : Net n, LipNear K r L t → ∀ t' : Net n, Close K r η t' t →
      η * ∑ l ∈ range (depth t), (L : ℝ) ^ l ≤ r →
        ∀ x ∈ K, |eval t' x - eval t x| ≤ η * ∑ l ∈ range (depth t), (L : ℝ) ^ l := by
  intro t
  induction t with
  | raw i =>
    intro _ t' hc _ x _
    cases hc
    simp [depth]
  | node k args g ih =>
    intro hL t' hc hmargin x hx
    rcases hL with _ | ⟨hg, hargs⟩
    rcases hc with _ | @⟨_, args', _, g', _, hg', hargs'⟩
    set d := Finset.univ.sup fun j => depth (args j) with hd
    rw [depth_node] at hmargin ⊢
    have hL0 : (0 : ℝ) ≤ L := L.coe_nonneg
    set e := η * ∑ l ∈ range d, (L : ℝ) ^ l with he
    have he0 : 0 ≤ e := mul_nonneg hη (Finset.sum_nonneg fun _ _ => by positivity)
    have he_le : e ≤ r :=
      (mul_le_mul_of_nonneg_left (geom_mono hL0 (Nat.le_succ d)) hη).trans hmargin
    -- by induction, every input deviates by at most `e`
    have hdev : ∀ j, dist (inVec args' x j) (inVec args x j) ≤ e := by
      intro j
      rw [inVec_apply, inVec_apply, Real.dist_eq]
      have hsum := mul_le_mul_of_nonneg_left (geom_mono hL0 (depth_arg_le args j)) hη
      exact (ih j (hargs j) (args' j) (hargs' j) (hsum.trans he_le) x hx).trans hsum
    have hdist : dist (inVec args' x) (inVec args x) ≤ e := (dist_pi_le_iff he0).mpr hdev
    have hu : inVec args x ∈ cthickening r (inSet K args) :=
      self_subset_cthickening _ (inVec_mem_inSet args hx)
    have hu' : inVec args' x ∈ cthickening r (inSet K args) :=
      mem_cthickening_of_dist_le _ _ _ _ (inVec_mem_inSet args hx) (hdist.trans he_le)
    have h1 := hg' _ hu'
    have h2 := hg.dist_le_mul _ hu' _ hu
    rw [Real.dist_eq] at h2
    rw [eval_node, eval_node]
    calc |g' (inVec args' x) - g (inVec args x)|
        ≤ |g' (inVec args' x) - g (inVec args' x)| +
            |g (inVec args' x) - g (inVec args x)| := abs_sub_le _ _ _
      _ ≤ η + L * e := add_le_add h1 (h2.trans (mul_le_mul_of_nonneg_left hdist hL0))
      _ = η * ∑ l ∈ range (d + 1), (L : ℝ) ^ l := by
        rw [← geom_succ, he]
        ring

/-- **Thm. compose: composition beats dimension.**  Let `t` compute a composition on `K` whose
vertex maps are `L`-Lipschitz in the max-norm on the closed `r`-neighbourhoods of their input
sets, and let `t'` replace every vertex map by one within `η ≥ 0` of it there.  If the margin
`η ∑_{l<D} L^l ≤ r` holds (`D` the depth), then on `K`
`|t'(x) - t(x)| ≤ η ∑_{l<D} L^l`. -/
theorem compose {K : Set (Fin n → ℝ)} {r η : ℝ} {L : NNReal} (hη : 0 ≤ η) {t t' : Net n}
    (hL : LipNear K r L t) (hc : Close K r η t' t)
    (hmargin : η * ∑ l ∈ range (depth t), (L : ℝ) ^ l ≤ r) :
    ∀ x ∈ K, |eval t' x - eval t x| ≤ η * ∑ l ∈ range (depth t), (L : ℝ) ^ l :=
  compose_aux hη t hL t' hc hmargin

end Net

end CobwebForest
