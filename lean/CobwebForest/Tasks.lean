/-
Copyright (c) 2026 Cobweb Forest authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Cobweb Forest authors
-/
import CobwebForest.LearnRate

/-!
# Which outcomes recover the target (theory: Prop. tasks (b), Prop. xy; sufficiency)

On finite probability spaces (as in `Cues.lean`):

* `condExp_eq_of_factor` (the sufficiency criterion, last step of Prop. multiview and of
  Prop. suffclass): if `f = E[Y | X]` is a function of a cue `M = G ∘ X`, then
  `E[Y | M] = E[Y | X]` on the support; the forest whose output tree reads `M` is sufficient.
* `funnel_xy` (Prop. xy, Prop. tasks (b)): for independent `X₁, X₂` with mean zero and
  `Y = X₁ X₂`, the funnel forest over two one-input trees has ideal messages
  `m₁ = E[Y | X₁] ≡ 0` and `m₂ = E[Y | X₂] ≡ 0`, ideal output `F ≡ 0`, and deficit
  `Δ = E (f - F)² = Var Y`: it learns nothing.  (Symmetric inputs have mean zero.)
-/

open Finset

namespace CobwebForest

section General

variable {Ω : Type*} [Fintype Ω] (P : Dist Ω)

/-- If `V` is a function of the cue `A`, then `E[V | A] = V` at points of positive
probability. -/
theorem condExp_of_fn {α : Type*} [DecidableEq α] (V : Ω → ℝ) (A : Ω → α) (F : α → ℝ)
    (hV : ∀ ω, V ω = F (A ω)) {ω : Ω} (hω : 0 < P.p ω) : condExp P V A ω = V ω := by
  have h := abs_condExp_sub_le P V A hω (V ω) 0 fun ω' hω'eq _ => by
    rw [hV ω', hV ω, hω'eq, sub_self, abs_zero]
  exact sub_eq_zero.mp (abs_nonpos_iff.mp h)

/-- Conditioning on a constant cue gives the expectation. -/
theorem condExp_const_cue {α : Type*} [DecidableEq α] (V : Ω → ℝ) (a : α) (ω : Ω) :
    condExp P V (fun _ => a) ω = expect P V := by
  unfold condExp fibreMass expect
  simp [P.sum_one]

open Filter Topology in
/-- **Cor. iff, first half.**  If the learned outputs `F̂_N` converge to the ideal output `F`,
their mean squared error for `f` converges to the deficit `Δ = E (F - f)²`. -/
theorem tendsto_mse (Fhat : ℕ → Ω → ℝ) (F f : Ω → ℝ)
    (hconv : ∀ ω, Tendsto (fun N => Fhat N ω) atTop (𝓝 (F ω))) :
    Tendsto (fun N => expect P fun ω => (Fhat N ω - f ω) ^ 2) atTop
      (𝓝 (expect P fun ω => (F ω - f ω) ^ 2)) := by
  unfold expect
  refine tendsto_finsetSum _ fun ω _ => ?_
  exact (((hconv ω).sub_const (f ω)).pow 2).const_mul (P.p ω)

open Filter Topology in
/-- **Cor. iff (consistency if and only if sufficiency).**  If the learned outputs converge to
the ideal output `F = E[Y | U_o]`, the forest is consistent for `f = E[Y | X]` (mean squared
error tending to `0`) if and only if `F = f` on the support, i.e. its task assignment is
sufficient (`Δ = 0`). -/
theorem consistent_iff_sufficient (Fhat : ℕ → Ω → ℝ) (F f : Ω → ℝ)
    (hconv : ∀ ω, Tendsto (fun N => Fhat N ω) atTop (𝓝 (F ω))) :
    Tendsto (fun N => expect P fun ω => (Fhat N ω - f ω) ^ 2) atTop (𝓝 0) ↔
      ∀ ω, 0 < P.p ω → F ω = f ω := by
  constructor
  · intro h0 ω hω
    have hΔ : (expect P fun ω => (F ω - f ω) ^ 2) = 0 :=
      tendsto_nhds_unique (tendsto_mse P Fhat F f hconv) h0
    unfold expect at hΔ
    have hterm := (Finset.sum_eq_zero_iff_of_nonneg fun ω _ =>
      mul_nonneg (P.nonneg ω) (sq_nonneg (F ω - f ω))).mp hΔ ω (Finset.mem_univ ω)
    rcases mul_eq_zero.mp hterm with h | h
    · exact absurd h hω.ne'
    · exact sub_eq_zero.mp (pow_eq_zero_iff two_ne_zero |>.mp h)
  · intro hsuff
    have hΔ : (expect P fun ω => (F ω - f ω) ^ 2) = 0 := by
      unfold expect
      refine Finset.sum_eq_zero fun ω _ => ?_
      rcases (P.nonneg ω).lt_or_eq with h | h
      · beta_reduce
        rw [hsuff ω h, sub_self]
        ring
      · rw [← h, zero_mul]
    rw [← hΔ]
    exact tendsto_mse P Fhat F f hconv

/-- **Sufficiency criterion.**  If `E[Y | X]` is a function `φ ∘ G` of a cue `G ∘ X` computed
from `X`, then `E[Y | G ∘ X] = E[Y | X]` at points of positive probability. -/
theorem condExp_eq_of_factor {α β : Type*} [DecidableEq α] [DecidableEq β] (Y : Ω → ℝ)
    (X : Ω → α) (G : α → β) (φ : β → ℝ)
    (hf : ∀ ω, 0 < P.p ω → condExp P Y X ω = φ (G (X ω))) {ω : Ω} (hω : 0 < P.p ω) :
    condExp P Y (fun ω => G (X ω)) ω = condExp P Y X ω := by
  rw [← condExp_tower P Y X (fun ω => G (X ω)) G (fun _ => rfl), condExp_congr P hf,
    condExp_of_fn P _ _ φ (fun _ => rfl) hω, hf ω hω]

end General

/-! ## Products of finite probability spaces -/

section Product

variable {Ω₁ Ω₂ : Type*} [Fintype Ω₁] [Fintype Ω₂]

/-- The product of two finite probability spaces: independent components. -/
noncomputable def Dist.prod (P₁ : Dist Ω₁) (P₂ : Dist Ω₂) : Dist (Ω₁ × Ω₂) where
  p ω := P₁.p ω.1 * P₂.p ω.2
  nonneg ω := mul_nonneg (P₁.nonneg _) (P₂.nonneg _)
  sum_one := by
    rw [Fintype.sum_prod_type]
    simp_rw [← Finset.mul_sum, P₂.sum_one, mul_one, P₁.sum_one]

variable (P₁ : Dist Ω₁) (P₂ : Dist Ω₂)

theorem expect_prod_mul (Z₁ : Ω₁ → ℝ) (Z₂ : Ω₂ → ℝ) :
    expect (P₁.prod P₂) (fun ω => Z₁ ω.1 * Z₂ ω.2) = expect P₁ Z₁ * expect P₂ Z₂ := by
  unfold expect
  rw [Fintype.sum_prod_type, Finset.sum_mul]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun b _ => ?_
  simp only [Dist.prod]
  ring

/-- Independence: conditioning a product on a cue of the first component. -/
theorem condExp_prod_left {α : Type*} [DecidableEq α] (A : Ω₁ → α) (Z₁ : Ω₁ → ℝ)
    (Z₂ : Ω₂ → ℝ) (ω : Ω₁ × Ω₂) :
    condExp (P₁.prod P₂) (fun ω => Z₁ ω.1 * Z₂ ω.2) (fun ω => A ω.1) ω =
      condExp P₁ Z₁ A ω.1 * expect P₂ Z₂ := by
  have hfib : (univ.filter fun ω' : Ω₁ × Ω₂ => A ω'.1 = A ω.1) =
      (univ.filter fun a => A a = A ω.1) ×ˢ (univ : Finset Ω₂) := by
    ext ⟨a, b⟩
    simp
  unfold condExp fibreMass expect
  rw [hfib, Finset.sum_product, Finset.sum_product]
  have h1 : ∀ a, ∑ b, (P₁.prod P₂).p (a, b) * (Z₁ (a, b).1 * Z₂ (a, b).2) =
      P₁.p a * Z₁ a * ∑ b, P₂.p b * Z₂ b := fun a => by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun b _ => ?_
    simp only [Dist.prod]
    ring
  have h2 : ∀ a, ∑ b, (P₁.prod P₂).p (a, b) = P₁.p a := fun a => by
    simp only [Dist.prod]
    rw [← Finset.mul_sum, P₂.sum_one, mul_one]
  simp_rw [h1, h2]
  rw [← Finset.sum_mul, mul_div_right_comm]

/-- Independence: conditioning a product on a cue of the second component. -/
theorem condExp_prod_right {α : Type*} [DecidableEq α] (A : Ω₂ → α) (Z₁ : Ω₁ → ℝ)
    (Z₂ : Ω₂ → ℝ) (ω : Ω₁ × Ω₂) :
    condExp (P₁.prod P₂) (fun ω => Z₁ ω.1 * Z₂ ω.2) (fun ω => A ω.2) ω =
      expect P₁ Z₁ * condExp P₂ Z₂ A ω.2 := by
  have hfib : (univ.filter fun ω' : Ω₁ × Ω₂ => A ω'.2 = A ω.2) =
      (univ : Finset Ω₁) ×ˢ (univ.filter fun b => A b = A ω.2) := by
    ext ⟨a, b⟩
    simp
  unfold condExp fibreMass expect
  rw [hfib, Finset.sum_product, Finset.sum_product]
  have h1 : ∀ a, ∑ b ∈ univ.filter (fun b => A b = A ω.2),
      (P₁.prod P₂).p (a, b) * (Z₁ (a, b).1 * Z₂ (a, b).2) =
        P₁.p a * Z₁ a * ∑ b ∈ univ.filter (fun b => A b = A ω.2), P₂.p b * Z₂ b := fun a => by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun b _ => ?_
    simp only [Dist.prod]
    ring
  have h2 : ∀ a, ∑ b ∈ univ.filter (fun b => A b = A ω.2), (P₁.prod P₂).p (a, b) =
      P₁.p a * ∑ b ∈ univ.filter (fun b => A b = A ω.2), P₂.p b := fun a => by
    simp only [Dist.prod]
    rw [Finset.mul_sum]
  simp_rw [h1, h2]
  rw [← Finset.sum_mul, ← Finset.sum_mul, P₁.sum_one, one_mul, mul_div_assoc]

/-- **Prop. xy (Prop. tasks (b)): funnel tasks can fail.**  Let `X₁` and `X₂` be independent
with mean zero (e.g. symmetric about `0`) and `Y = X₁ X₂`.  In the funnel forest whose output
tree reads the activations of a tree perceiving `X₁` and a tree perceiving `X₂`, the ideal
messages `m₁ = E[Y | X₁]` and `m₂ = E[Y | X₂]` vanish, the ideal output `F = E[Y | (m₁, m₂)]`
vanishes, and the deficit `Δ = E (f - F)²`, `f = E[Y | X]`, is `Var Y`. -/
theorem funnel_xy (X₁ : Ω₁ → ℝ) (X₂ : Ω₂ → ℝ) (h₁ : expect P₁ X₁ = 0)
    (h₂ : expect P₂ X₂ = 0) :
    (∀ ω, condExp (P₁.prod P₂) (fun ω => X₁ ω.1 * X₂ ω.2) (fun ω => X₁ ω.1) ω = 0) ∧
    (∀ ω, condExp (P₁.prod P₂) (fun ω => X₁ ω.1 * X₂ ω.2) (fun ω => X₂ ω.2) ω = 0) ∧
    (∀ ω, condExp (P₁.prod P₂) (fun ω => X₁ ω.1 * X₂ ω.2)
      (fun ω => (condExp (P₁.prod P₂) (fun ω => X₁ ω.1 * X₂ ω.2) (fun ω => X₁ ω.1) ω,
                 condExp (P₁.prod P₂) (fun ω => X₁ ω.1 * X₂ ω.2) (fun ω => X₂ ω.2) ω)) ω = 0) ∧
    expect (P₁.prod P₂) (fun ω =>
      (condExp (P₁.prod P₂) (fun ω => X₁ ω.1 * X₂ ω.2) (fun ω => (X₁ ω.1, X₂ ω.2)) ω -
       condExp (P₁.prod P₂) (fun ω => X₁ ω.1 * X₂ ω.2)
        (fun ω => (condExp (P₁.prod P₂) (fun ω => X₁ ω.1 * X₂ ω.2) (fun ω => X₁ ω.1) ω,
                   condExp (P₁.prod P₂) (fun ω => X₁ ω.1 * X₂ ω.2) (fun ω => X₂ ω.2) ω)) ω) ^ 2) =
      var (P₁.prod P₂) (fun ω => X₁ ω.1 * X₂ ω.2) := by
  set P := P₁.prod P₂ with hP
  set Y : Ω₁ × Ω₂ → ℝ := fun ω => X₁ ω.1 * X₂ ω.2 with hY
  have hm₁ : ∀ ω, condExp P Y (fun ω => X₁ ω.1) ω = 0 := fun ω => by
    rw [hP, hY, condExp_prod_left P₁ P₂ X₁ X₁ X₂ ω, h₂, mul_zero]
  have hm₂ : ∀ ω, condExp P Y (fun ω => X₂ ω.2) ω = 0 := fun ω => by
    rw [hP, hY, condExp_prod_right P₁ P₂ X₂ X₁ X₂ ω, h₁, zero_mul]
  have hEY : expect P Y = 0 := by rw [hP, hY, expect_prod_mul, h₁, zero_mul]
  -- the output tree's ideal input is the constant `(0, 0)`
  have hcue : (fun ω => (condExp P Y (fun ω => X₁ ω.1) ω, condExp P Y (fun ω => X₂ ω.2) ω)) =
      fun _ => ((0 : ℝ), (0 : ℝ)) := by
    funext ω
    rw [hm₁, hm₂]
  have hF : ∀ ω, condExp P Y
      (fun ω => (condExp P Y (fun ω => X₁ ω.1) ω, condExp P Y (fun ω => X₂ ω.2) ω)) ω = 0 := by
    intro ω
    rw [hcue, condExp_const_cue, hEY]
  refine ⟨hm₁, hm₂, hF, ?_⟩
  -- `f = E[Y | X] = Y` on the support, so `Δ = E Y² = Var Y`
  simp_rw [hF, sub_zero]
  unfold var expect
  rw [show ∑ ω, P.p ω * Y ω = expect P Y from rfl, hEY]
  refine Finset.sum_congr rfl fun ω _ => ?_
  rcases (P.nonneg ω).lt_or_eq with hω | hω
  · beta_reduce
    rw [condExp_of_fn P Y (fun ω => (X₁ ω.1, X₂ ω.2)) (fun x => x.1 * x.2) (fun _ => rfl) hω,
      sub_zero]
  · rw [← hω, zero_mul, zero_mul]

end Product

end CobwebForest
