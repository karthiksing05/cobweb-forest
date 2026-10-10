/-
Copyright (c) 2026 Cobweb Forest authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Cobweb Forest authors
-/
import Mathlib

/-!
# A cue calculus on finite probability spaces

An observation space `Ω` (a finite type) carries a distribution `p`.  An *outcome* is a real
random variable `V : Ω → ℝ`; a *cue* is any function `A : Ω → α` into a type with decidable
equality (a raw coordinate, a label, an activation, a tuple of these).

* `condExp p V A ω` is `E[V | A = A ω]`, the ideal activation of a tree whose cue is `A`
  and whose outcome is `V` (theory: Def. ideal forest).
* `validity p V A` is `Var(E[V | A])`, the part of `Var V` the cue explains.
  By the law of total variance it equals `Var V - E Var(V | A)`.

Main results:

* `validity_nonneg` : a cue never has negative validity.
* `condExp_const_of_validity_zero` : a cue of zero validity has a constant ideal activation
  on the support -- a purely *contextual* cue passes nothing through a funnel activation.
  This is the formal core of the rule "merge contextual cues or pass them as labels".
-/

open Finset BigOperators

namespace CobwebForest

variable {Ω : Type*} [Fintype Ω] {α : Type*} [DecidableEq α]

/-- A probability distribution on a finite space. -/
structure Dist (Ω : Type*) [Fintype Ω] where
  p : Ω → ℝ
  nonneg : ∀ ω, 0 ≤ p ω
  sum_one : ∑ ω, p ω = 1

variable (P : Dist Ω)

/-- Expectation. -/
noncomputable def expect (V : Ω → ℝ) : ℝ := ∑ ω, P.p ω * V ω

/-- Probability mass of the fibre of `A` through `ω`. -/
noncomputable def fibreMass (A : Ω → α) (ω : Ω) : ℝ :=
  ∑ ω' ∈ univ.filter (fun ω' => A ω' = A ω), P.p ω'

/-- `E[V | A]` evaluated at `ω`: the average of `V` over the fibre of `A` through `ω`.
(On a fibre of mass zero the value is `0`; such fibres carry no probability.) -/
noncomputable def condExp (V : Ω → ℝ) (A : Ω → α) (ω : Ω) : ℝ :=
  (∑ ω' ∈ univ.filter (fun ω' => A ω' = A ω), P.p ω' * V ω') / fibreMass P A ω

/-- Variance. -/
noncomputable def var (V : Ω → ℝ) : ℝ := expect P (fun ω => (V ω - expect P V) ^ 2)

/-- Validity of a cue: the variance of the ideal activation `E[V | A]`. -/
noncomputable def validity (V : Ω → ℝ) (A : Ω → α) : ℝ := var P (condExp P V A)

theorem var_nonneg (V : Ω → ℝ) : 0 ≤ var P V := by
  unfold var expect
  exact Finset.sum_nonneg (fun ω _ => mul_nonneg (P.nonneg ω) (sq_nonneg _))

/-- A cue never has negative validity. -/
theorem validity_nonneg (V : Ω → ℝ) (A : Ω → α) : 0 ≤ validity P V A :=
  var_nonneg P _

/-- If a random variable has zero variance it is constant (equal to its mean) on the support. -/
theorem eq_mean_of_var_zero (W : Ω → ℝ) (h : var P W = 0) :
    ∀ ω, 0 < P.p ω → W ω = expect P W := by
  intro ω hω
  unfold var expect at h
  have hterm : ∀ ω' ∈ (univ : Finset Ω), P.p ω' * (W ω' - ∑ ω, P.p ω * W ω) ^ 2 = 0 :=
    (Finset.sum_eq_zero_iff_of_nonneg
      (fun ω' _ => mul_nonneg (P.nonneg ω') (sq_nonneg _))).mp h
  have h1 := hterm ω (mem_univ ω)
  rcases mul_eq_zero.mp h1 with h2 | h2
  · exact absurd h2 (ne_of_gt hω)
  · have := pow_eq_zero_iff (n := 2) (by norm_num) |>.mp h2
    unfold expect
    linarith

/-- **Contextual cues pass nothing through activations.**  If a cue has zero validity, its
ideal activation `E[V | A]` is the same constant on every point of positive probability:
a funnel tree whose cue is a purely contextual cue emits a constant message. -/
theorem condExp_const_of_validity_zero (V : Ω → ℝ) (A : Ω → α)
    (h : validity P V A = 0) :
    ∀ ω₁ ω₂, 0 < P.p ω₁ → 0 < P.p ω₂ → condExp P V A ω₁ = condExp P V A ω₂ := by
  intro ω₁ ω₂ h₁ h₂
  have e₁ := eq_mean_of_var_zero P (condExp P V A) h ω₁ h₁
  have e₂ := eq_mean_of_var_zero P (condExp P V A) h ω₂ h₂
  rw [e₁, e₂]

/-- **Blocking.**  If cue `A` is a function of cue `C`, the joint cue `(A, C)` has the same
fibres as `C`, so the ideal activation -- and hence the validity -- is unchanged:
`v(A | C) = v(A, C) - v(C) = 0`.  A cue redundant given an already-learned cue is blocked
(Kamin), and the forest may drop it. -/
theorem condExp_blocked {γ : Type*} [DecidableEq γ] (V : Ω → ℝ) (C : Ω → γ) (f : γ → α) :
    condExp P V (fun ω => (f (C ω), C ω)) = condExp P V C := by
  funext ω
  have hfib : (univ.filter (fun ω' => (f (C ω'), C ω') = (f (C ω), C ω)))
      = univ.filter (fun ω' => C ω' = C ω) := by
    apply Finset.filter_congr
    intro ω' _
    constructor
    · intro h; exact (Prod.mk.inj h).2
    · intro h; rw [h]
  unfold condExp fibreMass
  simp only [hfib]

theorem conditional_validity_blocked {γ : Type*} [DecidableEq γ] (V : Ω → ℝ) (C : Ω → γ)
    (f : γ → α) :
    validity P V (fun ω => (f (C ω), C ω)) - validity P V C = 0 := by
  unfold validity
  rw [condExp_blocked]
  ring

end CobwebForest
