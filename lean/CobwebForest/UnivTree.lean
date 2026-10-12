/-
Copyright (c) 2026 Cobweb Forest authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Cobweb Forest authors
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Normed.Group.Bounded
import Mathlib.Topology.UniformSpace.HeineCantor
import Mathlib.Topology.MetricSpace.Pseudo.Basic

/-!
# One Cobweb tree is a universal approximator (theory: Thm. univ-tree)

A depth-one Cobweb tree whose leaves hold exact copies of the examples `(γ s, y s)`, with
counts `n s`, predicts at `x` with the **count-weighted leaf mixture** (paper, eq. (1))

  `ŷ(x) = ∑ n_s e^{-d(x, γ_s)² / 2h²} y_s / ∑ n_s e^{-d(x, γ_s)² / 2h²}`.

(The Gaussian normalising constants are equal for all leaves and cancel.)

* `leafMixture` is this prediction, over any pseudo-metric space; on `EuclideanSpace ℝ (Fin d)`
  the distance is the Euclidean one of the paper.
* `univ_tree` (Thm. univ-tree): for `K` compact, `g` continuous on `K` and `ε > 0` there are a
  finite `S ⊆ K` and `h > 0` such that the tree whose leaves hold `(s, g s)`, `s ∈ S`, each
  with count one, is within `ε` of `g` on `K`.
* `univ_tree_counts`: the same with any counts in `[nmin, nmax]`, `0 < nmin`: one `S` and one
  `h` work for all such counts (the paper's remark after the proof).
-/

open Finset Metric

namespace CobwebForest

/-- Gaussian weight of a leaf at distance `d` under the variance floor `h`. -/
noncomputable def gauss (h d : ℝ) : ℝ := Real.exp (-(d ^ 2) / (2 * h ^ 2))

theorem gauss_pos (h d : ℝ) : 0 < gauss h d := Real.exp_pos _

/-- **The count-weighted leaf mixture** (paper, eq. (1)): the prediction at `x` of a depth-one
Cobweb tree whose leaves `s ∈ S` hold `n s` copies of the example `(γ s, y s)`, with variance
floor `h`. -/
noncomputable def leafMixture {E ι : Type*} [PseudoMetricSpace E] (S : Finset ι) (γ : ι → E)
    (y n : ι → ℝ) (h : ℝ) (x : E) : ℝ :=
  (∑ s ∈ S, n s * gauss h (dist x (γ s)) * y s) / ∑ s ∈ S, n s * gauss h (dist x (γ s))

/-- The error of a weighted average is at most the weighted average of the errors. -/
theorem abs_wavg_sub_le {ι : Type*} (S : Finset ι) (w y : ι → ℝ) (c : ℝ)
    (hw : ∀ s ∈ S, 0 ≤ w s) (hW : 0 < ∑ s ∈ S, w s) :
    |(∑ s ∈ S, w s * y s) / (∑ s ∈ S, w s) - c| ≤
      (∑ s ∈ S, w s * |y s - c|) / ∑ s ∈ S, w s := by
  have hW' : (∑ s ∈ S, w s) ≠ 0 := hW.ne'
  have h1 : ∑ s ∈ S, w s * (y s - c) = ∑ s ∈ S, w s * y s - (∑ s ∈ S, w s) * c := by
    rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun s _ => by ring
  have h2 : (∑ s ∈ S, w s * y s) / (∑ s ∈ S, w s) - c =
      (∑ s ∈ S, w s * (y s - c)) / ∑ s ∈ S, w s := by
    rw [h1, sub_div, mul_div_cancel_left₀ c hW']
  rw [h2, abs_div, abs_of_pos hW]
  gcongr
  calc |∑ s ∈ S, w s * (y s - c)| ≤ ∑ s ∈ S, |w s * (y s - c)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ = ∑ s ∈ S, w s * |y s - c| :=
        Finset.sum_congr rfl fun s hs => by rw [abs_mul, abs_of_nonneg (hw s hs)]

/-- A near leaf (distance `< ρ/2`) has Gaussian weight at least `e^{-ρ²/8h²}`. -/
theorem gauss_near {h ρ d : ℝ} (hh : 0 < h) (hd0 : 0 ≤ d) (hd : d < ρ / 2) :
    Real.exp (-(ρ ^ 2) / (8 * h ^ 2)) ≤ gauss h d := by
  unfold gauss
  apply Real.exp_le_exp.mpr
  have hh2 : 0 < h ^ 2 := by positivity
  have hdd : 4 * d ^ 2 ≤ ρ ^ 2 := by nlinarith
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith

/-- A far leaf (distance `≥ ρ ≥ 0`) has Gaussian weight at most `e^{-ρ²/2h²}`. -/
theorem gauss_far {h ρ d : ℝ} (hh : 0 < h) (hρ : 0 ≤ ρ) (hd : ρ ≤ d) :
    gauss h d ≤ Real.exp (-(ρ ^ 2) / (2 * h ^ 2)) := by
  unfold gauss
  apply Real.exp_le_exp.mpr
  have hh2 : 0 < 2 * h ^ 2 := by positivity
  apply div_le_div_of_nonneg_right _ hh2.le
  have : ρ ^ 2 ≤ d ^ 2 := pow_le_pow_left₀ hρ hd 2
  linarith

/-- The ratio of a far weight to a near weight. -/
theorem exp_far_div_near (h ρ : ℝ) (hh : 0 < h) :
    Real.exp (-(ρ ^ 2) / (2 * h ^ 2)) / Real.exp (-(ρ ^ 2) / (8 * h ^ 2)) =
      Real.exp (-(3 * ρ ^ 2) / (8 * h ^ 2)) := by
  rw [← Real.exp_sub]
  congr 1
  have : h ^ 2 ≠ 0 := by positivity
  field_simp
  ring

/-- **The kernel-smoother bound at one query.**  If some leaf `s₀` is within `ρ/2` of `x`, the
leaves within `ρ` have values within `η` of `g x`, and all values are within `2M`, then the
count-weighted leaf mixture is within `η + 2M |S| (nmax/nmin) e^{-3ρ²/8h²}` of `g x`. -/
theorem leafMixture_err_le {E : Type*} [PseudoMetricSpace E] (S : Finset E) (g n : E → ℝ)
    {h ρ η M nmin nmax : ℝ} (x s₀ : E) (hs₀ : s₀ ∈ S) (hx₀ : dist x s₀ < ρ / 2)
    (hρ : 0 < ρ) (hh : 0 < h) (hnmin : 0 < nmin) (hM : 0 ≤ M)
    (hn : ∀ s ∈ S, nmin ≤ n s ∧ n s ≤ nmax)
    (hnear : ∀ s ∈ S, dist x s < ρ → |g s - g x| ≤ η)
    (hfar : ∀ s ∈ S, |g s - g x| ≤ 2 * M) :
    |leafMixture S id g n h x - g x| ≤
      η + 2 * M * S.card * (nmax / nmin) * Real.exp (-(3 * ρ ^ 2) / (8 * h ^ 2)) := by
  classical
  set w : E → ℝ := fun s => n s * gauss h (dist x s) with hw_def
  have hn0 : ∀ s ∈ S, 0 < n s := fun s hs => hnmin.trans_le (hn s hs).1
  have hw0 : ∀ s ∈ S, 0 ≤ w s := fun s hs => (mul_pos (hn0 s hs) (gauss_pos _ _)).le
  have hnmax : 0 < nmax := (hn0 s₀ hs₀).trans_le (hn s₀ hs₀).2
  -- the normaliser is at least the weight of the near leaf `s₀`
  have hEn : 0 < Real.exp (-(ρ ^ 2) / (8 * h ^ 2)) := Real.exp_pos _
  have hws₀ : nmin * Real.exp (-(ρ ^ 2) / (8 * h ^ 2)) ≤ w s₀ :=
    mul_le_mul (hn s₀ hs₀).1 (gauss_near hh dist_nonneg hx₀) hEn.le (hn0 s₀ hs₀).le
  have hWs₀ : w s₀ ≤ ∑ s ∈ S, w s := Finset.single_le_sum hw0 hs₀
  have hW : 0 < ∑ s ∈ S, w s := lt_of_lt_of_le (mul_pos hnmin hEn) (hws₀.trans hWs₀)
  -- the mixture is a weighted average with weights `w`
  have hmix : leafMixture S id g n h x = (∑ s ∈ S, w s * g s) / ∑ s ∈ S, w s := rfl
  rw [hmix]
  refine (abs_wavg_sub_le S w g (g x) hw0 hW).trans ?_
  -- split into near and far leaves
  set Ef := Real.exp (-(ρ ^ 2) / (2 * h ^ 2)) with hEf
  have hsplit := Finset.sum_filter_add_sum_filter_not S (fun s => dist x s < ρ)
    (fun s => w s * |g s - g x|)
  have hnear_sum : ∑ s ∈ S with dist x s < ρ, w s * |g s - g x| ≤ η * ∑ s ∈ S, w s := by
    calc ∑ s ∈ S with dist x s < ρ, w s * |g s - g x|
        ≤ ∑ s ∈ S with dist x s < ρ, w s * η := by
          apply Finset.sum_le_sum
          intro s hs
          rw [Finset.mem_filter] at hs
          exact mul_le_mul_of_nonneg_left (hnear s hs.1 hs.2) (hw0 s hs.1)
      _ = η * ∑ s ∈ S with dist x s < ρ, w s := by rw [Finset.mul_sum]; simp [mul_comm]
      _ ≤ η * ∑ s ∈ S, w s := by
          have hη : 0 ≤ η := (abs_nonneg _).trans (hnear s₀ hs₀ (by linarith))
          apply mul_le_mul_of_nonneg_left _ hη
          exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
            fun s hs _ => hw0 s hs
  have hfar_sum : ∑ s ∈ S with ¬dist x s < ρ, w s * |g s - g x| ≤
      S.card * (nmax * Ef * (2 * M)) := by
    calc ∑ s ∈ S with ¬dist x s < ρ, w s * |g s - g x|
        ≤ ∑ s ∈ S with ¬dist x s < ρ, nmax * Ef * (2 * M) := by
          apply Finset.sum_le_sum
          intro s hs
          rw [Finset.mem_filter] at hs
          have hg := gauss_far hh hρ.le (not_lt.mp hs.2)
          apply mul_le_mul _ (hfar s hs.1) (abs_nonneg _) (by positivity)
          exact mul_le_mul (hn s hs.1).2 hg (gauss_pos _ _).le hnmax.le
      _ = (S.filter fun s => ¬dist x s < ρ).card * (nmax * Ef * (2 * M)) := by
          rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ S.card * (nmax * Ef * (2 * M)) := by
          apply mul_le_mul_of_nonneg_right _ (by positivity)
          exact_mod_cast Finset.card_filter_le _ _
  have hfrac : S.card * (nmax * Ef * (2 * M)) / ∑ s ∈ S, w s ≤
      2 * M * S.card * (nmax / nmin) * Real.exp (-(3 * ρ ^ 2) / (8 * h ^ 2)) := by
    rw [← exp_far_div_near h ρ hh, div_le_iff₀ hW]
    have key : S.card * (nmax * Ef * (2 * M)) ≤
        2 * M * S.card * (nmax / nmin) * (Ef / Real.exp (-(ρ ^ 2) / (8 * h ^ 2))) *
          (nmin * Real.exp (-(ρ ^ 2) / (8 * h ^ 2))) := by
      apply le_of_eq
      field_simp
    refine key.trans ?_
    apply mul_le_mul_of_nonneg_left (hws₀.trans hWs₀)
    have : 0 ≤ Ef := (Real.exp_pos _).le
    positivity
  calc (∑ s ∈ S, w s * |g s - g x|) / ∑ s ∈ S, w s
      = (∑ s ∈ S with dist x s < ρ, w s * |g s - g x| +
          ∑ s ∈ S with ¬dist x s < ρ, w s * |g s - g x|) / ∑ s ∈ S, w s := by rw [hsplit]
    _ ≤ (η * ∑ s ∈ S, w s + S.card * (nmax * Ef * (2 * M))) / ∑ s ∈ S, w s := by
        gcongr
    _ = η + S.card * (nmax * Ef * (2 * M)) / ∑ s ∈ S, w s := by
        rw [add_div, mul_div_cancel_right₀ _ hW.ne']
    _ ≤ η + 2 * M * S.card * (nmax / nmin) * Real.exp (-(3 * ρ ^ 2) / (8 * h ^ 2)) := by
        gcongr

/-- For every `ρ > 0` and every target `q > 0`, a small enough variance floor makes the
far-to-near weight ratio `e^{-3ρ²/8h²}` smaller than `q`. -/
theorem exists_h_exp_lt {ρ q : ℝ} (hρ : 0 < ρ) (hq : 0 < q) :
    ∃ h > 0, Real.exp (-(3 * ρ ^ 2) / (8 * h ^ 2)) < q := by
  set T := |Real.log q| + 1 with hT
  have hT0 : 0 < T := by positivity
  refine ⟨ρ * Real.sqrt (3 / (8 * T)), by positivity, ?_⟩
  have hs : Real.sqrt (3 / (8 * T)) ^ 2 = 3 / (8 * T) := Real.sq_sqrt (by positivity)
  have hexp : -(3 * ρ ^ 2) / (8 * (ρ * Real.sqrt (3 / (8 * T))) ^ 2) = -T := by
    rw [mul_pow, hs]
    field_simp
  rw [hexp]
  calc Real.exp (-T) < Real.exp (Real.log q) := by
        apply Real.exp_lt_exp.mpr
        have := neg_abs_le (Real.log q)
        linarith
    _ = q := Real.exp_log hq

/-- **One Cobweb tree is a universal approximator, for any counts** (Thm. univ-tree and the
remark after its proof).  For `K` compact, `g` continuous on `K`, `ε > 0` and count bounds
`0 < nmin ≤ nmax`, there are a finite set `S ⊆ K` and a variance floor `h > 0` such that for
*every* choice of counts `n s ∈ [nmin, nmax]`, the depth-one tree whose leaves hold `n s`
copies of `(s, g s)` predicts within `ε` of `g` on all of `K`. -/
theorem univ_tree_counts {E : Type*} [PseudoMetricSpace E] {K : Set E} (hK : IsCompact K)
    {g : E → ℝ} (hg : ContinuousOn g K) {ε : ℝ} (hε : 0 < ε) {nmin nmax : ℝ}
    (hnmin : 0 < nmin) (hnmax : nmin ≤ nmax) :
    ∃ S : Finset E, (↑S : Set E) ⊆ K ∧ (K.Nonempty → S.Nonempty) ∧ ∃ h > 0, ∀ n : E → ℝ,
      (∀ s ∈ S, nmin ≤ n s ∧ n s ≤ nmax) → ∀ x ∈ K, |leafMixture S id g n h x - g x| < ε := by
  -- Step 1: `M ≥ sup_K |g|`
  obtain ⟨M₀, hM₀⟩ := hK.exists_bound_of_continuousOn hg
  set M := max M₀ 0 with hM_def
  have hM0 : 0 ≤ M := le_max_right _ _
  have hM : ∀ x ∈ K, |g x| ≤ M := fun x hx =>
    (Real.norm_eq_abs (g x) ▸ hM₀ x hx).trans (le_max_left _ _)
  -- Step 2: uniform continuity on `K`
  obtain ⟨ρ, hρ, hρg⟩ := Metric.uniformContinuousOn_iff.mp
    (hK.uniformContinuousOn_of_continuous hg) (ε / 2) (by positivity)
  -- Step 3: a finite `ρ/2`-net `S ⊆ K`
  obtain ⟨T, hTK, hTfin, hcover⟩ :=
    Metric.finite_approx_of_totallyBounded hK.totallyBounded (ρ / 2) (by positivity)
  refine ⟨hTfin.toFinset, by simpa using hTK, fun ⟨x, hx⟩ => ?_, ?_⟩
  · obtain ⟨s₀, hs₀T, -⟩ := Set.mem_iUnion₂.mp (hcover hx)
    exact ⟨s₀, (Set.Finite.mem_toFinset hTfin).mpr hs₀T⟩
  set S := hTfin.toFinset with hS_def
  have hSK : ∀ s ∈ S, s ∈ K := fun s hs => hTK ((Set.Finite.mem_toFinset hTfin).mp hs)
  -- Step 6: choose `h` with `2 M |S| (nmax/nmin) e^{-3ρ²/8h²} < ε/2`
  set C := 2 * M * S.card * (nmax / nmin) + 1 with hC_def
  have hC : 0 < C := by
    have : 0 ≤ 2 * M * S.card * (nmax / nmin) := by
      have : 0 < nmax := hnmin.trans_le hnmax
      positivity
    linarith
  obtain ⟨h, hh, hhq⟩ := exists_h_exp_lt hρ (show 0 < ε / 2 / C by positivity)
  refine ⟨h, hh, fun n hn x hx => ?_⟩
  -- Step 4: a leaf `s₀` within `ρ/2` of `x`
  obtain ⟨s₀, hs₀T, hxs₀⟩ := Set.mem_iUnion₂.mp (hcover hx)
  have hs₀ : s₀ ∈ S := (Set.Finite.mem_toFinset hTfin).mpr hs₀T
  rw [Metric.mem_ball] at hxs₀
  -- Step 5: the kernel-smoother bound
  have hbound := leafMixture_err_le S g n x s₀ hs₀ hxs₀ hρ hh hnmin hM0 hn
    (η := ε / 2)
    (fun s hs hd => by
      have := hρg s (hSK s hs) x hx (by rwa [dist_comm])
      rw [Real.dist_eq] at this
      exact this.le)
    (fun s hs => by
      calc |g s - g x| ≤ |g s| + |g x| := abs_sub _ _
        _ ≤ M + M := add_le_add (hM s (hSK s hs)) (hM x hx)
        _ = 2 * M := by ring)
  refine hbound.trans_lt ?_
  have hE := Real.exp_pos (-(3 * ρ ^ 2) / (8 * h ^ 2))
  have hlt : 2 * M * S.card * (nmax / nmin) * Real.exp (-(3 * ρ ^ 2) / (8 * h ^ 2)) < ε / 2 :=
    calc 2 * M * S.card * (nmax / nmin) * Real.exp (-(3 * ρ ^ 2) / (8 * h ^ 2))
        ≤ C * Real.exp (-(3 * ρ ^ 2) / (8 * h ^ 2)) := by
          apply mul_le_mul_of_nonneg_right _ hE.le
          linarith
      _ < C * (ε / 2 / C) := mul_lt_mul_of_pos_left hhq hC
      _ = ε / 2 := by field_simp
  linarith

/-- **Thm. univ-tree: one Cobweb tree is a universal approximator.**  For `K` compact, `g`
continuous on `K` and `ε > 0` there are a finite `S ⊆ K` and `h > 0` such that the depth-one
Cobweb tree whose leaves hold the examples `(s, g s)`, `s ∈ S`, each with count one, satisfies
`|ŷ(x) - g(x)| < ε` for every `x ∈ K`. -/
theorem univ_tree {E : Type*} [PseudoMetricSpace E] {K : Set E} (hK : IsCompact K)
    {g : E → ℝ} (hg : ContinuousOn g K) {ε : ℝ} (hε : 0 < ε) :
    ∃ S : Finset E, (↑S : Set E) ⊆ K ∧ ∃ h > 0,
      ∀ x ∈ K, |leafMixture S id g (fun _ => 1) h x - g x| < ε := by
  obtain ⟨S, hSK, -, h, hh, hS⟩ := univ_tree_counts hK hg hε one_pos le_rfl
  exact ⟨S, hSK, h, hh, hS _ fun _ _ => ⟨le_rfl, le_rfl⟩⟩

end CobwebForest
