/-
Copyright (c) 2026 Cobweb Forest authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Cobweb Forest authors
-/
import CobwebForest.Cues
import CobwebForest.Compose

/-!
# Learned forests converge at the rate of their fan-in (theory: Thm. learn-rate)

The deterministic form of the theorem: all probability sits in the hypotheses.  The
distribution of the raw input `X` and of the outcomes is a finite probability space `P` on `Ω`
(as in `Cues.lean`; `condExp P V A ω = E[V | A = A ω]`), and the frozen learned trees are fixed
maps.  The support of `X` is `Q = {X ω | P ω > 0}`.

For every tree `τ` of the ideal forest `t` (maps `ψ_τ`) and of the learned forest `t'` (maps
`ψ̂_τ`, same shape), with ideal inputs `U_τ`, learned inputs `Û_τ` and outcome `V_τ`:

* (i)   sufficiency: `E[V_τ | X_{R(τ)}] = ψ_τ(U_τ)`, where `R(τ)` is the set of raw coordinates
        upstream of `τ` (`Net.rawSet`) and `X_R` the cue that reveals exactly them;
* (ii)  smoothness: `ψ_τ` is `L`-Lipschitz (max-norm) on the set of ideal and learned inputs;
* (iii) single-tree rate: `|ψ̂_τ(û) - E[V_τ | Û_τ = û]| ≤ r` for `û ∈ Û_τ(Q)`.

`learn_rate`: then the output error is at most `r ∑_{l=0}^{D} (2L)^l` on `Q`, `D + 1` the depth.

The steps are the paper's: the tower property `E[E[V | X_R] | Û] = E[V | Û]` (`condExp_tower`,
`Û` is a function of `X_R` because the parents are frozen), so by (i)
`E[V | Û = û] = E[ψ(U) | Û = û]`, which is within `L e` of `ψ(û)` because `‖U - Û‖ ≤ e` surely
(`abs_condExp_sub_le`); then the triangle inequality and induction on the depth.
-/

open Finset

namespace CobwebForest

variable {Ω : Type*} [Fintype Ω] (P : Dist Ω)

/-! ## Conditional expectations on a finite space -/

/-- On a union `T` of fibres of `A`, summing `p · E[V | A]` gives `∑ p · V`. -/
theorem sum_mul_condExp_of_saturated {α : Type*} [DecidableEq α] (V : Ω → ℝ) (A : Ω → α)
    (T : Finset Ω) (hT : ∀ ω₁ ω₂, A ω₁ = A ω₂ → (ω₁ ∈ T ↔ ω₂ ∈ T)) :
    ∑ ω ∈ T, P.p ω * condExp P V A ω = ∑ ω ∈ T, P.p ω * V ω := by
  have hmaps : ∀ ω ∈ T, A ω ∈ T.image A := fun ω hω => Finset.mem_image_of_mem A hω
  rw [← Finset.sum_fiberwise_of_maps_to hmaps, ← Finset.sum_fiberwise_of_maps_to hmaps]
  refine Finset.sum_congr rfl fun a ha => ?_
  obtain ⟨ω₀, hω₀T, rfl⟩ := Finset.mem_image.mp ha
  -- the part of `T` in the fibre through `ω₀` is the whole fibre
  have hfib : {ω ∈ T | A ω = A ω₀} = {ω ∈ (univ : Finset Ω) | A ω = A ω₀} := by
    ext ω
    simp only [mem_filter, mem_univ, true_and]
    exact ⟨fun h => h.2, fun h => ⟨(hT ω ω₀ h).mpr hω₀T, h⟩⟩
  rw [hfib]
  -- on the fibre, `E[V | A]` is constant
  have hconst : ∀ ω ∈ {ω ∈ (univ : Finset Ω) | A ω = A ω₀},
      P.p ω * condExp P V A ω = P.p ω * condExp P V A ω₀ := by
    intro ω hω
    simp only [mem_filter, mem_univ, true_and] at hω
    simp only [condExp, fibreMass, hω]
  rw [Finset.sum_congr rfl hconst, ← Finset.sum_mul]
  have hM : fibreMass P A ω₀ = ∑ ω ∈ {ω ∈ (univ : Finset Ω) | A ω = A ω₀}, P.p ω := rfl
  rw [← hM]
  by_cases h0 : fibreMass P A ω₀ = 0
  · have hp0 : ∀ ω ∈ {ω ∈ (univ : Finset Ω) | A ω = A ω₀}, P.p ω = 0 :=
      (Finset.sum_eq_zero_iff_of_nonneg (fun ω _ => P.nonneg ω)).mp (hM ▸ h0)
    rw [h0, zero_mul]
    exact (Finset.sum_eq_zero fun ω hω => by rw [hp0 ω hω, zero_mul]).symm
  · unfold condExp
    rw [mul_div_cancel₀ _ h0]

/-- **Tower property.**  If the cue `B` is a function of the cue `A` (`B = F ∘ A`), then
`E[E[V | A] | B] = E[V | B]`. -/
theorem condExp_tower {α β : Type*} [DecidableEq α] [DecidableEq β] (V : Ω → ℝ) (A : Ω → α)
    (B : Ω → β) (F : α → β) (hB : ∀ ω, B ω = F (A ω)) (ω : Ω) :
    condExp P (condExp P V A) B ω = condExp P V B ω := by
  have h := sum_mul_condExp_of_saturated P V A {ω' ∈ (univ : Finset Ω) | B ω' = B ω}
    (fun ω₁ ω₂ h12 => by simp [hB, h12])
  change (∑ ω' ∈ {ω' ∈ (univ : Finset Ω) | B ω' = B ω}, P.p ω' * condExp P V A ω') /
      fibreMass P B ω = (∑ ω' ∈ {ω' ∈ (univ : Finset Ω) | B ω' = B ω}, P.p ω' * V ω') /
      fibreMass P B ω
  rw [h]

/-- `E[V | A]` depends only on the values of `V` on the support. -/
theorem condExp_congr {α : Type*} [DecidableEq α] {V W : Ω → ℝ}
    (hVW : ∀ ω, 0 < P.p ω → V ω = W ω) (A : Ω → α) (ω : Ω) :
    condExp P V A ω = condExp P W A ω := by
  unfold condExp
  congr 1
  refine Finset.sum_congr rfl fun ω' _ => ?_
  rcases (P.nonneg ω').lt_or_eq with h | h
  · rw [hVW ω' h]
  · rw [← h, zero_mul, zero_mul]

/-- **Averaging.**  If `|Z - c| ≤ a` on the support of the fibre of `A` through a point of
positive probability, then `|E[Z | A] - c| ≤ a` there. -/
theorem abs_condExp_sub_le {α : Type*} [DecidableEq α] (Z : Ω → ℝ) (A : Ω → α) {ω : Ω}
    (hω : 0 < P.p ω) (c a : ℝ) (h : ∀ ω', A ω' = A ω → 0 < P.p ω' → |Z ω' - c| ≤ a) :
    |condExp P Z A ω - c| ≤ a := by
  have hmem : ω ∈ {ω' ∈ (univ : Finset Ω) | A ω' = A ω} := by simp
  have hM : 0 < ∑ ω' ∈ {ω' ∈ (univ : Finset Ω) | A ω' = A ω}, P.p ω' :=
    hω.trans_le (Finset.single_le_sum (fun ω' _ => P.nonneg ω') hmem)
  have hwavg := abs_wavg_sub_le _ P.p Z c (fun ω' _ => P.nonneg ω') hM
  refine hwavg.trans ?_
  rw [div_le_iff₀ hM, Finset.mul_sum]
  refine Finset.sum_le_sum fun ω' hω' => ?_
  simp only [mem_filter, mem_univ, true_and] at hω'
  rcases (P.nonneg ω').lt_or_eq with hp | hp
  · rw [mul_comm a]
    exact mul_le_mul_of_nonneg_left (h ω' hω' hp) (P.nonneg ω')
  · rw [← hp, zero_mul, mul_zero]

/-! ## Upstream coordinates -/

namespace Net

variable {n : ℕ}

/-- The raw coordinates upstream of the output, `R(τ)`. -/
def rawSet : Net n → Finset (Fin n)
  | raw i => {i}
  | node _ args _ => Finset.univ.biUnion fun j => rawSet (args j)

/-- The cue `X_R`: the raw input with the coordinates outside `R` hidden. -/
def maskTo (R : Finset (Fin n)) (x : Fin n → ℝ) : Fin n → ℝ := fun i => if i ∈ R then x i else 0

theorem eval_maskTo : ∀ (t : Net n) (R : Finset (Fin n)), rawSet t ⊆ R →
    ∀ x, eval t (maskTo R x) = eval t x := by
  intro t
  induction t with
  | raw i =>
    intro R hR x
    have hi : i ∈ R := hR (Finset.mem_singleton_self i)
    simp [maskTo, hi]
  | node k args g ih =>
    intro R hR x
    rw [eval_node, eval_node]
    congr 1
    funext j
    rw [inVec_apply, inVec_apply]
    exact ih j R ((Finset.subset_biUnion_of_mem (fun j => rawSet (args j))
      (Finset.mem_univ j)).trans hR) x

end Net

/-! ## The learned forest next to its ideal forest -/

variable {n : ℕ} (X : Ω → (Fin n → ℝ))

open Net in
/-- `Trained P X L r t' t`: the learned forest `t'` (maps `ψ̂`) has the shape of the ideal
forest `t` (maps `ψ`), and every tree, with outcome `V`, satisfies (i) sufficiency,
(ii) smoothness and (iii) the single-tree rate `r` of Thm. learn-rate. -/
inductive Trained (L : NNReal) (r : ℝ) : Net n → Net n → Prop
  | raw (i : Fin n) : Trained L r (raw i) (raw i)
  | node {k : ℕ} {args' args : Fin k → Net n} {ψ' ψ : (Fin k → ℝ) → ℝ} (V : Ω → ℝ)
      (hargs : ∀ j, Trained L r (args' j) (args j))
      (suff : ∀ ω, 0 < P.p ω →
        condExp P V (fun ω => maskTo (rawSet (node k args ψ)) (X ω)) ω = ψ (inVec args (X ω)))
      (lip : LipschitzOnWith L ψ
        {u | ∃ ω, 0 < P.p ω ∧ (u = inVec args (X ω) ∨ u = inVec args' (X ω))})
      (rate : ∀ ω, 0 < P.p ω →
        |ψ' (inVec args' (X ω)) - condExp P V (fun ω => inVec args' (X ω)) ω| ≤ r) :
      Trained L r (node k args' ψ') (node k args ψ)

theorem Trained.rawSet_eq {L : NNReal} {r : ℝ} {t' t : Net n} (h : Trained P X L r t' t) :
    Net.rawSet t' = Net.rawSet t := by
  induction h with
  | raw i => rfl
  | node V _ _ _ _ ih => simp only [Net.rawSet, ih]

open Net in
theorem learn_rate_aux {L : NNReal} {r : ℝ} :
    ∀ t : Net n, ∀ t' : Net n, Trained P X L r t' t →
      ∀ ω, 0 < P.p ω → |eval t' (X ω) - eval t (X ω)| ≤
        r * ∑ l ∈ range (depth t), (2 * (L : ℝ)) ^ l := by
  intro t
  induction t with
  | raw i =>
    intro t' h ω _
    cases h
    simp [depth]
  | node k args ψ ih =>
    intro t' h ω hω
    rcases h with _ | @⟨_, args', _, ψ', _, V, hargs, suff, lip, rate⟩
    have hr : 0 ≤ r := (abs_nonneg _).trans (rate ω hω)
    have hL0 : (0 : ℝ) ≤ L := L.coe_nonneg
    set d := Finset.univ.sup fun j => depth (args j) with hd
    rw [depth_node]
    set e := r * ∑ l ∈ range d, (2 * (L : ℝ)) ^ l with he
    -- learned and ideal inputs are within `e` of each other on the support
    have hdev : ∀ ω', 0 < P.p ω' → dist (inVec args' (X ω')) (inVec args (X ω')) ≤ e := by
      intro ω' hω'
      have he0 : 0 ≤ e := mul_nonneg hr (Finset.sum_nonneg fun _ _ => by positivity)
      refine (dist_pi_le_iff he0).mpr fun j => ?_
      rw [inVec_apply, inVec_apply, Real.dist_eq]
      exact (ih j (args' j) (hargs j) ω' hω').trans (mul_le_mul_of_nonneg_left
        (geom_mono (by positivity) (depth_arg_le args j)) hr)
    -- the learned input is a function of the upstream raw coordinates
    have hraw : ∀ j, rawSet (args' j) ⊆ rawSet (node k args ψ) := fun j => by
      rw [(hargs j).rawSet_eq P X]
      exact Finset.subset_biUnion_of_mem (fun j => rawSet (args j)) (Finset.mem_univ j)
    have hfac : ∀ ω', inVec args' (X ω') =
        inVec args' (maskTo (rawSet (node k args ψ)) (X ω')) := fun ω' => by
      funext j
      rw [inVec_apply, inVec_apply, eval_maskTo _ _ (hraw j)]
    -- step 1: (iii)
    have h1 := rate ω hω
    -- step 2: tower property, (i) and averaging give `|E[V | Û] - ψ(Û)| ≤ L e`
    have h2 : |condExp P V (fun ω' => inVec args' (X ω')) ω - ψ (inVec args' (X ω))| ≤ L * e := by
      rw [← condExp_tower P V (fun ω' => maskTo (rawSet (node k args ψ)) (X ω'))
        (fun ω' => inVec args' (X ω')) (inVec args') hfac, condExp_congr P suff _ ω]
      refine abs_condExp_sub_le P _ _ hω _ _ fun ω' hω'eq hω' => ?_
      have hÛ : inVec args' (X ω') = inVec args' (X ω) := hω'eq
      change |ψ (inVec args (X ω')) - ψ (inVec args' (X ω))| ≤ L * e
      rw [← hÛ]
      have hl := lip.dist_le_mul (inVec args (X ω')) ⟨ω', hω', Or.inl rfl⟩
        (inVec args' (X ω')) ⟨ω', hω', Or.inr rfl⟩
      rw [Real.dist_eq, dist_comm] at hl
      exact hl.trans (mul_le_mul_of_nonneg_left (hdev ω' hω') hL0)
    -- step 3: (ii)
    have h3 : |ψ (inVec args' (X ω)) - ψ (inVec args (X ω))| ≤ L * e := by
      have hl := lip.dist_le_mul (inVec args' (X ω)) ⟨ω, hω, Or.inr rfl⟩
        (inVec args (X ω)) ⟨ω, hω, Or.inl rfl⟩
      rw [Real.dist_eq] at hl
      exact hl.trans (mul_le_mul_of_nonneg_left (hdev ω hω) hL0)
    rw [eval_node, eval_node]
    have t1 := abs_sub_le (ψ' (inVec args' (X ω)))
      (condExp P V (fun ω' => inVec args' (X ω')) ω) (ψ (inVec args (X ω)))
    have t2 := abs_sub_le (condExp P V (fun ω' => inVec args' (X ω')) ω)
      (ψ (inVec args' (X ω))) (ψ (inVec args (X ω)))
    calc |ψ' (inVec args' (X ω)) - ψ (inVec args (X ω))|
        ≤ r + L * e + L * e := by linarith
      _ = r * ∑ l ∈ range (d + 1), (2 * (L : ℝ)) ^ l := by
        rw [← geom_succ, he]
        ring

/-- **Thm. learn-rate (deterministic form): learned forests converge at the rate of their
fan-in.**  If every tree of the learned forest `t'` satisfies sufficiency (i), smoothness (ii)
with constant `L` and the single-tree rate (iii) with error `r` relative to the ideal forest
`t`, then on the support of `X` the output error is at most `r ∑_{l=0}^{D} (2L)^l`, where
`D + 1 = depth t` (the paper's `D` levels below the output). -/
theorem learn_rate {L : NNReal} {r : ℝ} {t t' : Net n} (h : Trained P X L r t' t) :
    ∀ ω, 0 < P.p ω → |Net.eval t' (X ω) - Net.eval t (X ω)| ≤
      r * ∑ l ∈ range (Net.depth t), (2 * (L : ℝ)) ^ l :=
  learn_rate_aux P X t t' h

end CobwebForest
