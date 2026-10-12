/-
Copyright (c) 2026 Cobweb Forest authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Cobweb Forest authors
-/
import CobwebForest.UnivTree
import Mathlib.Analysis.InnerProductSpace.Basic

/-!
# The perturbed design does not matter (theory: Thm. ideal, Step 2)

A tree trained on the learned inputs `Û_i = U_i + Δ_i`, `‖Δ_i‖ ≤ ε`, of its frozen parents is
the Nadaraya–Watson estimate `ψ̂` on the perturbed design; `ψ̌` is the same estimate on the ideal
inputs `U_i`.  Step 2 of the proof of Thm. ideal compares them; its deterministic part is here.
The probabilistic part -- the near points carry kernel mass at least `c N` and the far points
at most `C N e^{-R²/16}`, uniformly, with probability tending to one (uniform kernel density
laws, Einmahl & Mason 2005) -- enters as hypotheses.

* `gauss_ratio` : for `‖u - U‖ ≤ R h`, the Gaussian weights at `U + Δ` and at `U` differ by a
  factor in `[e^{-η}, e^{η}]`, `η = R ε / h + ε² / 2h²`.
* `gauss_far_perturbed` : for `‖u - U‖ > R h ≥ 2ε`, the weight at `U + Δ` is at most
  `e^{-‖u - U‖² / 8h²}`.
* `wavg_perturb` : two weighted averages of values bounded by `B`, whose weights agree up to
  `e^{±η}` on a near set, differ by at most `B (e^{2η} - 1) + 2B (far/near mass ratios)`.
* `step2_bound` : combining them,
  `|ψ̂(u) - ψ̌(u)| ≤ B (e^{2η} - 1) + 2B (C/c)(1 + e^{η}) e^{-R²/16}`,
  the paper's `2B (e^{2η} - 1 + C' e^{-R²/16})`.
-/

open Finset

namespace CobwebForest

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **Kernel ratio.**  For `‖u - U‖ ≤ R h` and `‖Δ‖ ≤ ε`, the Gaussian weights at `U + Δ` and at
`U` differ by a factor in `[e^{-η}, e^{η}]`, `η = R ε / h + ε² / 2h²`. -/
theorem gauss_ratio {h R ε : ℝ} (hh : 0 < h) (hR : 0 ≤ R) {u U Δ : E}
    (hu : ‖u - U‖ ≤ R * h) (hΔ : ‖Δ‖ ≤ ε) :
    Real.exp (-(R * ε / h + ε ^ 2 / (2 * h ^ 2))) * gauss h ‖u - U‖ ≤ gauss h ‖u - (U + Δ)‖ ∧
      gauss h ‖u - (U + Δ)‖ ≤ Real.exp (R * ε / h + ε ^ 2 / (2 * h ^ 2)) * gauss h ‖u - U‖ := by
  have hε : 0 ≤ ε := (norm_nonneg _).trans hΔ
  have hsq : ‖u - (U + Δ)‖ ^ 2 = ‖u - U‖ ^ 2 - 2 * inner ℝ (u - U) Δ + ‖Δ‖ ^ 2 := by
    rw [show u - (U + Δ) = (u - U) - Δ by abel]
    exact norm_sub_sq_real _ _
  have hinner : |inner ℝ (u - U) Δ| ≤ R * h * ε :=
    (abs_real_inner_le_norm _ _).trans (mul_le_mul hu hΔ (norm_nonneg _) (by positivity))
  have hΔsq : ‖Δ‖ ^ 2 ≤ ε ^ 2 := pow_le_pow_left₀ (norm_nonneg _) hΔ 2
  have hh2 : 0 < 2 * h ^ 2 := by positivity
  -- the exponent moves by at most `η`
  set D := (2 * inner ℝ (u - U) Δ - ‖Δ‖ ^ 2) / (2 * h ^ 2) with hD
  have hsplit : -(‖u - (U + Δ)‖ ^ 2) / (2 * h ^ 2) = -(‖u - U‖ ^ 2) / (2 * h ^ 2) + D := by
    rw [hsq, hD]
    field_simp
    ring
  have hη : R * ε / h + ε ^ 2 / (2 * h ^ 2) = (2 * (R * h * ε) + ε ^ 2) / (2 * h ^ 2) := by
    field_simp
  have hDle : |D| ≤ R * ε / h + ε ^ 2 / (2 * h ^ 2) := by
    rw [hη, hD, abs_div, abs_of_pos hh2]
    apply div_le_div_of_nonneg_right _ hh2.le
    calc |2 * inner ℝ (u - U) Δ - ‖Δ‖ ^ 2| ≤ |2 * inner ℝ (u - U) Δ| + |‖Δ‖ ^ 2| :=
          abs_sub _ _
      _ = 2 * |inner ℝ (u - U) Δ| + ‖Δ‖ ^ 2 := by
          rw [abs_mul, abs_two, abs_of_nonneg (sq_nonneg ‖Δ‖)]
      _ ≤ 2 * (R * h * ε) + ε ^ 2 := by linarith
  rw [abs_le] at hDle
  unfold gauss
  rw [hsplit, Real.exp_add]
  constructor
  · rw [mul_comm]
    exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr hDle.1) (Real.exp_pos _).le
  · rw [mul_comm (Real.exp (R * ε / h + ε ^ 2 / (2 * h ^ 2)))]
    exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr hDle.2) (Real.exp_pos _).le

omit [InnerProductSpace ℝ E] in
/-- **Far points.**  For `‖u - U‖ > R h ≥ 2ε` and `‖Δ‖ ≤ ε`, the Gaussian weight at `U + Δ` is at
most `e^{-‖u - U‖² / 8h²}`. -/
theorem gauss_far_perturbed {h R ε : ℝ} (hh : 0 < h) {u U Δ : E} (hu : R * h < ‖u - U‖)
    (hRε : 2 * ε ≤ R * h) (hΔ : ‖Δ‖ ≤ ε) :
    gauss h ‖u - (U + Δ)‖ ≤ Real.exp (-(‖u - U‖ ^ 2) / (8 * h ^ 2)) := by
  have hlow : ‖u - U‖ / 2 ≤ ‖u - (U + Δ)‖ := by
    have := norm_sub_norm_le (u - U) Δ
    rw [show u - U - Δ = u - (U + Δ) by abel] at this
    linarith
  have hsq : (‖u - U‖ / 2) ^ 2 ≤ ‖u - (U + Δ)‖ ^ 2 :=
    pow_le_pow_left₀ (by positivity) hlow 2
  unfold gauss
  apply Real.exp_le_exp.mpr
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith [sq_nonneg h]

/-- Splitting a weighted average into its near part: the far part moves it by at most
`2B` times the far-to-near mass ratio. -/
theorem wavg_sub_near_le {I : Type*} (S : Finset I) (near : I → Prop) [DecidablePred near]
    (w V : I → ℝ) {B : ℝ} (hw : ∀ i ∈ S, 0 ≤ w i) (hV : ∀ i ∈ S, |V i| ≤ B)
    (hN : 0 < ∑ i ∈ S with near i, w i) :
    |(∑ i ∈ S, w i * V i) / (∑ i ∈ S, w i) -
        (∑ i ∈ S with near i, w i * V i) / (∑ i ∈ S with near i, w i)| ≤
      2 * B * ((∑ i ∈ S with ¬near i, w i) / (∑ i ∈ S with near i, w i)) := by
  set WN := ∑ i ∈ S with near i, w i with hWN
  set WF := ∑ i ∈ S with ¬near i, w i with hWF
  set SN := ∑ i ∈ S with near i, w i * V i with hSN
  set SF := ∑ i ∈ S with ¬near i, w i * V i with hSF
  have hW : ∑ i ∈ S, w i = WN + WF := (Finset.sum_filter_add_sum_filter_not S near w).symm
  have hS : ∑ i ∈ S, w i * V i = SN + SF :=
    (Finset.sum_filter_add_sum_filter_not S near (fun i => w i * V i)).symm
  have hWF0 : 0 ≤ WF := Finset.sum_nonneg fun i hi => hw i (Finset.mem_filter.mp hi).1
  have hB : 0 ≤ B := by
    obtain ⟨i, hi⟩ : (S.filter near).Nonempty := by
      by_contra h
      rw [Finset.not_nonempty_iff_eq_empty] at h
      rw [hWN, h, Finset.sum_empty] at hN
      exact lt_irrefl _ hN
    exact (abs_nonneg _).trans (hV i (Finset.mem_filter.mp hi).1)
  have habs : ∀ T : Finset I, T ⊆ S → |∑ i ∈ T, w i * V i| ≤ B * ∑ i ∈ T, w i := by
    intro T hT
    calc |∑ i ∈ T, w i * V i| ≤ ∑ i ∈ T, |w i * V i| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ i ∈ T, B * w i := Finset.sum_le_sum fun i hi => by
          rw [abs_mul, abs_of_nonneg (hw i (hT hi)), mul_comm]
          exact mul_le_mul_of_nonneg_right (hV i (hT hi)) (hw i (hT hi))
      _ = B * ∑ i ∈ T, w i := by rw [Finset.mul_sum]
  have hSN := habs _ (Finset.filter_subset near S)
  have hSF := habs _ (Finset.filter_subset (fun i => ¬near i) S)
  rw [hW, hS]
  have hWpos : 0 < WN + WF := by linarith
  have key : (SN + SF) / (WN + WF) - SN / WN = (WN * SF - WF * SN) / ((WN + WF) * WN) := by
    field_simp
    ring
  rw [key, abs_div, abs_of_pos (mul_pos hWpos hN), div_le_iff₀ (mul_pos hWpos hN)]
  have h1 : |WN * SF - WF * SN| ≤ WN * (B * WF) + WF * (B * WN) := by
    calc |WN * SF - WF * SN| ≤ |WN * SF| + |WF * SN| := abs_sub _ _
      _ = WN * |SF| + WF * |SN| := by
          rw [abs_mul, abs_mul, abs_of_pos hN, abs_of_nonneg hWF0]
      _ ≤ WN * (B * WF) + WF * (B * WN) :=
          add_le_add (mul_le_mul_of_nonneg_left hSF hN.le) (mul_le_mul_of_nonneg_left hSN hWF0)
  calc |WN * SF - WF * SN| ≤ WN * (B * WF) + WF * (B * WN) := h1
    _ = 2 * B * WF * WN := by ring
    _ ≤ 2 * B * WF * WN + 2 * B * WF * WF := by
        have : 0 ≤ 2 * B * WF * WF := by positivity
        linarith
    _ = 2 * B * (WF / WN) * ((WN + WF) * WN) := by
        field_simp

/-- Two weighted averages over the same points whose weights agree up to `e^{±η}` differ by
at most `B (e^{2η} - 1)`. -/
theorem wavg_ratio_le {I : Type*} (T : Finset I) (a b V : I → ℝ) {B η : ℝ} (hη : 0 ≤ η)
    (ha : ∀ i ∈ T, 0 ≤ a i) (hV : ∀ i ∈ T, |V i| ≤ B)
    (hratio : ∀ i ∈ T, Real.exp (-η) * a i ≤ b i ∧ b i ≤ Real.exp η * a i)
    (hA : 0 < ∑ i ∈ T, a i) :
    |(∑ i ∈ T, b i * V i) / (∑ i ∈ T, b i) - (∑ i ∈ T, a i * V i) / (∑ i ∈ T, a i)| ≤
      B * (Real.exp (2 * η) - 1) := by
  set A := ∑ i ∈ T, a i with hAdef
  set Bm := ∑ i ∈ T, b i with hBdef
  have he : 0 < Real.exp η := Real.exp_pos _
  have he' : 0 < Real.exp (-η) := Real.exp_pos _
  have hBlo : Real.exp (-η) * A ≤ Bm := by
    rw [hAdef, Finset.mul_sum]
    exact Finset.sum_le_sum fun i hi => (hratio i hi).1
  have hBhi : Bm ≤ Real.exp η * A := by
    rw [hAdef, Finset.mul_sum]
    exact Finset.sum_le_sum fun i hi => (hratio i hi).2
  have hBpos : 0 < Bm := lt_of_lt_of_le (mul_pos he' hA) hBlo
  have hexp2 : Real.exp η * Real.exp η = Real.exp (2 * η) := by rw [← Real.exp_add]; ring_nf
  have hexpm : Real.exp (-η) * Real.exp η = 1 := by rw [← Real.exp_add]; simp
  -- normalised weights
  have hdiff : ∀ i ∈ T, |b i / Bm - a i / A| ≤ (Real.exp (2 * η) - 1) * (a i / A) := by
    intro i hi
    have hai := ha i hi
    have hup : b i / Bm ≤ Real.exp (2 * η) * (a i / A) := by
      rw [div_le_iff₀ hBpos]
      have hk : Real.exp (2 * η) * (a i / A) * (Real.exp (-η) * A) = Real.exp η * a i := by
        have h1 : Real.exp (2 * η) * Real.exp (-η) = Real.exp η := by
          rw [← Real.exp_add]; ring_nf
        rw [show Real.exp (2 * η) * (a i / A) * (Real.exp (-η) * A) =
          (Real.exp (2 * η) * Real.exp (-η)) * a i * (A / A) by ring, h1, div_self hA.ne',
          mul_one]
      calc b i ≤ Real.exp η * a i := (hratio i hi).2
        _ = Real.exp (2 * η) * (a i / A) * (Real.exp (-η) * A) := hk.symm
        _ ≤ Real.exp (2 * η) * (a i / A) * Bm := by
            apply mul_le_mul_of_nonneg_left hBlo
            positivity
    have hlo : Real.exp (-(2 * η)) * (a i / A) ≤ b i / Bm := by
      rw [le_div_iff₀ hBpos]
      have hk : Real.exp (-(2 * η)) * (a i / A) * (Real.exp η * A) = Real.exp (-η) * a i := by
        have h1 : Real.exp (-(2 * η)) * Real.exp η = Real.exp (-η) := by
          rw [← Real.exp_add]; ring_nf
        rw [show Real.exp (-(2 * η)) * (a i / A) * (Real.exp η * A) =
          (Real.exp (-(2 * η)) * Real.exp η) * a i * (A / A) by ring, h1, div_self hA.ne',
          mul_one]
      calc Real.exp (-(2 * η)) * (a i / A) * Bm ≤ Real.exp (-(2 * η)) * (a i / A) *
            (Real.exp η * A) := by
            apply mul_le_mul_of_nonneg_left hBhi
            positivity
        _ = Real.exp (-η) * a i := hk
        _ ≤ b i := (hratio i hi).1
    have hgap : 1 - Real.exp (-(2 * η)) ≤ Real.exp (2 * η) - 1 := by
      have h1 : Real.exp (2 * η) * Real.exp (-(2 * η)) = 1 := by rw [← Real.exp_add]; simp
      have h2 : 0 < Real.exp (-(2 * η)) := Real.exp_pos _
      nlinarith [sq_nonneg (Real.exp (2 * η) - 1), Real.exp_pos (2 * η)]
    have haA : 0 ≤ a i / A := div_nonneg hai hA.le
    rw [abs_le]
    constructor <;> nlinarith
  have hsum1 : ∑ i ∈ T, a i / A = 1 := by rw [← Finset.sum_div, div_self hA.ne']
  calc |(∑ i ∈ T, b i * V i) / Bm - (∑ i ∈ T, a i * V i) / A|
      = |∑ i ∈ T, (b i / Bm - a i / A) * V i| := by
        congr 1
        rw [Finset.sum_div, Finset.sum_div, ← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun i _ => by ring
    _ ≤ ∑ i ∈ T, |(b i / Bm - a i / A) * V i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i ∈ T, (Real.exp (2 * η) - 1) * (a i / A) * B := by
        refine Finset.sum_le_sum fun i hi => ?_
        rw [abs_mul]
        exact mul_le_mul (hdiff i hi) (hV i hi) (abs_nonneg _)
          (mul_nonneg (by linarith [Real.add_one_le_exp (2 * η)]) (div_nonneg (ha i hi) hA.le))
    _ = B * (Real.exp (2 * η) - 1) := by
        rw [← Finset.sum_mul, ← Finset.mul_sum, hsum1]
        ring

/-- **Perturbed weighted averages.**  Weights `a` (ideal design) and `b` (perturbed design) that
agree up to `e^{±η}` on the near points give weighted averages of values bounded by `B` that
differ by at most `B (e^{2η} - 1) + 2B (A_far/A_near + B_far/B_near)`. -/
theorem wavg_perturb {I : Type*} (S : Finset I) (near : I → Prop) [DecidablePred near]
    (a b V : I → ℝ) {B η : ℝ} (hη : 0 ≤ η) (ha : ∀ i ∈ S, 0 ≤ a i) (hb : ∀ i ∈ S, 0 ≤ b i)
    (hV : ∀ i ∈ S, |V i| ≤ B)
    (hratio : ∀ i ∈ S, near i → Real.exp (-η) * a i ≤ b i ∧ b i ≤ Real.exp η * a i)
    (hAN : 0 < ∑ i ∈ S with near i, a i) :
    |(∑ i ∈ S, b i * V i) / (∑ i ∈ S, b i) - (∑ i ∈ S, a i * V i) / (∑ i ∈ S, a i)| ≤
      B * (Real.exp (2 * η) - 1) +
        2 * B * ((∑ i ∈ S with ¬near i, a i) / (∑ i ∈ S with near i, a i) +
          (∑ i ∈ S with ¬near i, b i) / (∑ i ∈ S with near i, b i)) := by
  have hsub : ∀ i ∈ S.filter near, i ∈ S ∧ near i := fun i hi => Finset.mem_filter.mp hi
  have hBN : 0 < ∑ i ∈ S with near i, b i := by
    have hlo : Real.exp (-η) * ∑ i ∈ S with near i, a i ≤ ∑ i ∈ S with near i, b i := by
      rw [Finset.mul_sum]
      exact Finset.sum_le_sum fun i hi => (hratio i (hsub i hi).1 (hsub i hi).2).1
    exact lt_of_lt_of_le (mul_pos (Real.exp_pos _) hAN) hlo
  have h1 := wavg_sub_near_le S near b V hb hV hBN
  have h2 := wavg_sub_near_le S near a V ha hV hAN
  have h3 := wavg_ratio_le (S.filter near) a b V hη (fun i hi => ha i (hsub i hi).1)
    (fun i hi => hV i (hsub i hi).1) (fun i hi => hratio i (hsub i hi).1 (hsub i hi).2) hAN
  have t1 := abs_sub_le ((∑ i ∈ S, b i * V i) / (∑ i ∈ S, b i))
    ((∑ i ∈ S with near i, b i * V i) / (∑ i ∈ S with near i, b i))
    ((∑ i ∈ S, a i * V i) / (∑ i ∈ S, a i))
  have t2 := abs_sub_le ((∑ i ∈ S with near i, b i * V i) / (∑ i ∈ S with near i, b i))
    ((∑ i ∈ S with near i, a i * V i) / (∑ i ∈ S with near i, a i))
    ((∑ i ∈ S, a i * V i) / (∑ i ∈ S, a i))
  rw [abs_sub_comm ((∑ i ∈ S with near i, a i * V i) / (∑ i ∈ S with near i, a i))] at t2
  linarith

/-- **Thm. ideal, Step 2 (deterministic form).**  Let the ideal inputs `U i` and the learned
inputs `U i + Δ i`, `‖Δ i‖ ≤ ε`, carry outcomes `|V i| ≤ B`, and let `ψ̌(u)` and `ψ̂(u)` be the
Gaussian Nadaraya–Watson estimates at `u` on the ideal and on the learned inputs.  If
`2ε ≤ R h`, the near points (`‖u - U i‖ ≤ R h`) carry ideal kernel mass at least `c N > 0` and
the far points carry `∑ e^{-‖u - U i‖²/8h²} ≤ C N e^{-R²/16}` (the cited uniform kernel density
laws), then `|ψ̂(u) - ψ̌(u)| ≤ B (e^{2η} - 1) + 2B (C/c)(1 + e^{η}) e^{-R²/16}`,
`η = R ε / h + ε²/2h²`. -/
theorem step2_bound {I : Type*} (S : Finset I) (U Δ : I → E) (V : I → ℝ) (u : E)
    {h R ε B c C N : ℝ} (hh : 0 < h) (hR : 0 ≤ R) (hRε : 2 * ε ≤ R * h)
    (hΔ : ∀ i ∈ S, ‖Δ i‖ ≤ ε) (hV : ∀ i ∈ S, |V i| ≤ B) (hc : 0 < c * N)
    (hnear : c * N ≤ ∑ i ∈ S with ‖u - U i‖ ≤ R * h, gauss h ‖u - U i‖)
    (hfar : ∑ i ∈ S with ¬‖u - U i‖ ≤ R * h, Real.exp (-(‖u - U i‖ ^ 2) / (8 * h ^ 2)) ≤
      C * N * Real.exp (-(R ^ 2) / 16)) :
    |(∑ i ∈ S, gauss h ‖u - (U i + Δ i)‖ * V i) / (∑ i ∈ S, gauss h ‖u - (U i + Δ i)‖) -
        (∑ i ∈ S, gauss h ‖u - U i‖ * V i) / (∑ i ∈ S, gauss h ‖u - U i‖)| ≤
      B * (Real.exp (2 * (R * ε / h + ε ^ 2 / (2 * h ^ 2))) - 1) +
        2 * B * (C / c) * (1 + Real.exp (R * ε / h + ε ^ 2 / (2 * h ^ 2))) *
          Real.exp (-(R ^ 2) / 16) := by
  set η := R * ε / h + ε ^ 2 / (2 * h ^ 2) with hηdef
  have hε : 0 ≤ ε := by
    by_cases hS : S.Nonempty
    · obtain ⟨i, hi⟩ := hS
      exact (norm_nonneg _).trans (hΔ i hi)
    · rw [Finset.not_nonempty_iff_eq_empty] at hS
      rw [hS, Finset.filter_empty, Finset.sum_empty] at hnear
      linarith
  have hη : 0 ≤ η := by positivity
  set a : I → ℝ := fun i => gauss h ‖u - U i‖ with ha
  set b : I → ℝ := fun i => gauss h ‖u - (U i + Δ i)‖ with hb
  set near : I → Prop := fun i => ‖u - U i‖ ≤ R * h with hnear_def
  have hAN : 0 < ∑ i ∈ S with near i, a i := lt_of_lt_of_le hc hnear
  have hmain := wavg_perturb S near a b V hη (fun i _ => (gauss_pos _ _).le)
    (fun i _ => (gauss_pos _ _).le) hV
    (fun i hi hni => gauss_ratio hh hR hni (hΔ i hi)) hAN
  refine hmain.trans ?_
  -- the far masses
  have hFa : ∑ i ∈ S with ¬near i, a i ≤ C * N * Real.exp (-(R ^ 2) / 16) := by
    refine le_trans (Finset.sum_le_sum fun i _ => ?_) hfar
    show gauss h ‖u - U i‖ ≤ _
    unfold gauss
    apply Real.exp_le_exp.mpr
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    nlinarith [mul_nonneg (sq_nonneg ‖u - U i‖) (sq_nonneg h)]
  have hFb : ∑ i ∈ S with ¬near i, b i ≤ C * N * Real.exp (-(R ^ 2) / 16) := by
    refine le_trans (Finset.sum_le_sum fun i hi => ?_) hfar
    have hi' := Finset.mem_filter.mp hi
    exact gauss_far_perturbed hh (not_le.mp hi'.2) hRε (hΔ i hi'.1)
  have hBN : Real.exp (-η) * (c * N) ≤ ∑ i ∈ S with near i, b i := by
    calc Real.exp (-η) * (c * N) ≤ Real.exp (-η) * ∑ i ∈ S with near i, a i :=
          mul_le_mul_of_nonneg_left hnear (Real.exp_pos _).le
      _ = ∑ i ∈ S with near i, Real.exp (-η) * a i := by rw [Finset.mul_sum]
      _ ≤ ∑ i ∈ S with near i, b i := Finset.sum_le_sum fun i hi => by
          have hi' := Finset.mem_filter.mp hi
          exact (gauss_ratio hh hR hi'.2 (hΔ i hi'.1)).1
  have hB : 0 ≤ B := by
    obtain ⟨i, hi⟩ : (S.filter near).Nonempty := by
      by_contra hne
      rw [Finset.not_nonempty_iff_eq_empty] at hne
      rw [hne, Finset.sum_empty] at hAN
      exact lt_irrefl _ hAN
    exact (abs_nonneg _).trans (hV i (Finset.mem_filter.mp hi).1)
  have hEe : 0 ≤ C * N * Real.exp (-(R ^ 2) / 16) :=
    le_trans (Finset.sum_nonneg fun i _ => (Real.exp_pos _).le) hfar
  have hratA : (∑ i ∈ S with ¬near i, a i) / (∑ i ∈ S with near i, a i) ≤
      C * N * Real.exp (-(R ^ 2) / 16) / (c * N) :=
    div_le_div₀ hEe hFa hc hnear
  have hratB : (∑ i ∈ S with ¬near i, b i) / (∑ i ∈ S with near i, b i) ≤
      C * N * Real.exp (-(R ^ 2) / 16) / (Real.exp (-η) * (c * N)) :=
    div_le_div₀ hEe hFb (mul_pos (Real.exp_pos _) hc) hBN
  have hsimpA : C * N * Real.exp (-(R ^ 2) / 16) / (c * N) = (C / c) * Real.exp (-(R ^ 2) / 16) := by
    have hcN : c * N ≠ 0 := hc.ne'
    have hc0 : c ≠ 0 := left_ne_zero_of_mul hcN
    have hN0 : N ≠ 0 := right_ne_zero_of_mul hcN
    field_simp
  have hsimpB : C * N * Real.exp (-(R ^ 2) / 16) / (Real.exp (-η) * (c * N)) =
      (C / c) * Real.exp η * Real.exp (-(R ^ 2) / 16) := by
    have hcN : c * N ≠ 0 := hc.ne'
    have hc0 : c ≠ 0 := left_ne_zero_of_mul hcN
    have hN0 : N ≠ 0 := right_ne_zero_of_mul hcN
    have hexp : Real.exp (-η) * Real.exp η = 1 := by rw [← Real.exp_add]; simp
    field_simp
    rw [mul_assoc, hexp, mul_one]
  rw [hsimpA] at hratA
  rw [hsimpB] at hratB
  have h2B : 0 ≤ 2 * B := by positivity
  calc B * (Real.exp (2 * η) - 1) +
        2 * B * ((∑ i ∈ S with ¬near i, a i) / (∑ i ∈ S with near i, a i) +
          (∑ i ∈ S with ¬near i, b i) / (∑ i ∈ S with near i, b i))
      ≤ B * (Real.exp (2 * η) - 1) + 2 * B * ((C / c) * Real.exp (-(R ^ 2) / 16) +
          (C / c) * Real.exp η * Real.exp (-(R ^ 2) / 16)) := by
        gcongr
    _ = B * (Real.exp (2 * η) - 1) +
          2 * B * (C / c) * (1 + Real.exp η) * Real.exp (-(R ^ 2) / 16) := by ring

end CobwebForest
