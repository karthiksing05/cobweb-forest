/-
Copyright (c) 2026 Cobweb Forest authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Cobweb Forest authors
-/
import CobwebForest.UnivTree
import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# The grid rate of one tree (theory: Prop. rate, appendix Thm. single)

Let `Q = ∏_i [a_i, a_i + w]` be a cube in `ℝ^d`, split into `m^d` cells of side `δ = w/m`, and
let `g` be `L`-Lipschitz on `Q` (for the `ℓ¹` distance, which covers the Euclidean and the max
norm).  The **grid tree** (`gridTree`) is the depth-one tree whose `G = m^d` leaves hold
`(γ, g γ)` at the cell centres `γ`, count one each, with variance floor `h = β δ`; it predicts
with the count-weighted leaf mixture (paper, eq. (1)).

* `grid_rate` (Prop. rate): `|ŷ(x) - g(x)| ≤ C(d, β) L w / m` for every `x ∈ Q`, with the explicit
  constant `C(d, β) = 2 d e^{1/2β²} / (1 - e^{-1/2β²})²` (`rateConst`).  Since `G = m^d`,
  `w / m = w G^{-1/d}`.

The proof: the Gaussian weight of a leaf is a product of one-dimensional weights
(`gauss_dist_toLp`), so the normaliser and the weighted distances factorise over the coordinates
(`sum_gauss_grid`, `sum_gauss_grid_mul`), and the error is at most `L ∑_i T_i / S_i` with
one-dimensional sums `S_i ≥ e^{-1/8β²}` (the nearest centre, `sum_w1_ge`) and
`T_i ≤ 2 δ e^{3/8β²} / (1 - q)²`, `q = e^{-1/2β²}` (`sum_w1_mul_le`), by comparison of the
`j`-th centre away from the nearest one with `q^j` and `∑_j (j + 1) q^j ≤ (1 - q)^{-2}`.
-/

open Finset

namespace CobwebForest

/-! ## Two elementary sums -/

theorem one_sub_mul_sum_succ_mul_pow (q : ℝ) (m : ℕ) :
    (1 - q) * ∑ j ∈ range m, ((j : ℝ) + 1) * q ^ j = ∑ j ∈ range m, q ^ j - m * q ^ m := by
  induction m with
  | zero => simp
  | succ m ih =>
    rw [Finset.sum_range_succ, Finset.sum_range_succ, mul_add, ih]
    push_cast
    ring

/-- `∑_{j<m} (j+1) q^j ≤ (1 - q)^{-2}` for `0 ≤ q < 1`. -/
theorem sum_succ_mul_pow_le {q : ℝ} (hq0 : 0 ≤ q) (hq1 : q < 1) (m : ℕ) :
    ∑ j ∈ range m, ((j : ℝ) + 1) * q ^ j ≤ 1 / (1 - q) ^ 2 := by
  have h1q : 0 < 1 - q := by linarith
  rw [le_div_iff₀ (by positivity)]
  have hgeo := mul_neg_geom_sum q m
  have htel := one_sub_mul_sum_succ_mul_pow q m
  have hextra : 0 ≤ (1 - q) * (m * q ^ m) := by positivity
  have hqm : 0 ≤ q ^ m := pow_nonneg hq0 m
  calc (∑ j ∈ range m, ((j : ℝ) + 1) * q ^ j) * (1 - q) ^ 2
      = (1 - q) * ((1 - q) * ∑ j ∈ range m, ((j : ℝ) + 1) * q ^ j) := by ring
    _ = (1 - q) * ∑ j ∈ range m, q ^ j - (1 - q) * (m * q ^ m) := by rw [htel]; ring
    _ = 1 - q ^ m - (1 - q) * (m * q ^ m) := by rw [hgeo]
    _ ≤ 1 := by linarith

/-- Summing `F(|a - a₀|)` over `a < m` meets every distance at most twice. -/
theorem sum_natDist_le (F : ℕ → ℝ) (hF : ∀ j, 0 ≤ F j) {m a₀ : ℕ} (ha₀ : a₀ < m) :
    ∑ a ∈ range m, F (a - a₀ + (a₀ - a)) ≤ 2 * ∑ j ∈ range m, F j := by
  rw [← Finset.sum_range_add_sum_Ico _ ha₀.le, two_mul]
  apply add_le_add
  · calc ∑ a ∈ range a₀, F (a - a₀ + (a₀ - a))
        = ∑ j ∈ range a₀, F ((a₀ - 1 - j) - a₀ + (a₀ - (a₀ - 1 - j))) :=
          (Finset.sum_range_reflect (fun a => F (a - a₀ + (a₀ - a))) a₀).symm
      _ = ∑ j ∈ range a₀, F (j + 1) := Finset.sum_congr rfl fun j hj => by
          rw [Finset.mem_range] at hj
          congr 1
          omega
      _ ≤ ∑ j ∈ range (a₀ + 1), F j := by
          rw [Finset.sum_range_succ']
          linarith [hF 0]
      _ ≤ ∑ j ∈ range m, F j :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono ha₀) fun j _ _ => hF j
  · calc ∑ a ∈ Ico a₀ m, F (a - a₀ + (a₀ - a))
        = ∑ j ∈ range (m - a₀), F ((a₀ + j) - a₀ + (a₀ - (a₀ + j))) :=
          Finset.sum_Ico_eq_sum_range _ _ _
      _ = ∑ j ∈ range (m - a₀), F j := Finset.sum_congr rfl fun j _ => by
          congr 1
          omega
      _ ≤ ∑ j ∈ range m, F j :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono (Nat.sub_le m a₀))
            fun j _ _ => hF j

/-! ## One coordinate -/

/-- The one-dimensional Gaussian factor of a leaf at `y` for the query coordinate `x`. -/
noncomputable def w1 (h y x : ℝ) : ℝ := Real.exp (-(x - y) ^ 2 / (2 * h ^ 2))

theorem w1_pos (h y x : ℝ) : 0 < w1 h y x := Real.exp_pos _

/-- The centre of cell `j` of a one-dimensional grid of spacing `δ` starting at `a`. -/
noncomputable def ctr (a δ : ℝ) (j : ℕ) : ℝ := a + ((j : ℝ) + 1 / 2) * δ

/-- Every point of `[a, a + m δ]` is within `δ/2` of a cell centre. -/
theorem exists_near_ctr {m : ℕ} (hm : 0 < m) {a δ x : ℝ} (hδ : 0 < δ) (hx1 : a ≤ x)
    (hx2 : x ≤ a + m * δ) : ∃ j₀ < m, |x - ctr a δ j₀| ≤ δ / 2 := by
  set u := (x - a) / δ with hu
  have hu0 : 0 ≤ u := div_nonneg (by linarith) hδ.le
  have hum : u ≤ m := by rw [hu, div_le_iff₀ hδ]; linarith
  set j₀ := min ⌊u⌋₊ (m - 1) with hj₀
  refine ⟨j₀, by omega, ?_⟩
  have h1 : (j₀ : ℝ) ≤ u :=
    (Nat.cast_le.mpr (min_le_left _ _)).trans (Nat.floor_le hu0)
  have h2 : u ≤ (j₀ : ℝ) + 1 := by
    rcases le_total ⌊u⌋₊ (m - 1) with h | h
    · rw [hj₀, min_eq_left h]
      exact (Nat.lt_floor_add_one u).le
    · rw [hj₀, min_eq_right h, Nat.cast_sub (by omega : 1 ≤ m)]
      push_cast
      linarith
  have hx' : x - ctr a δ j₀ = δ * (u - j₀ - 1 / 2) := by
    rw [hu, ctr]
    field_simp
    ring
  rw [hx', abs_mul, abs_of_pos hδ]
  have h3 : |u - j₀ - 1 / 2| ≤ 1 / 2 := by
    rw [abs_le]
    constructor <;> linarith
  calc δ * |u - j₀ - 1 / 2| ≤ δ * (1 / 2) := mul_le_mul_of_nonneg_left h3 hδ.le
    _ = δ / 2 := by ring

/-- The leaf `j` cells away from the nearest one contributes at most
`e^{3/8β²} δ (j + 1) q^j`, `q = e^{-1/2β²}`. -/
theorem w1_mul_le {β δ a x : ℝ} (hβ : 0 < β) (hδ : 0 < δ) {j₀ : ℕ}
    (hj₀ : |x - ctr a δ j₀| ≤ δ / 2) (j : ℕ) :
    w1 (β * δ) (ctr a δ j) x * |x - ctr a δ j| ≤
      Real.exp (3 / (8 * β ^ 2)) * δ * ((((j - j₀ + (j₀ - j) : ℕ) : ℝ) + 1) *
        Real.exp (-1 / (2 * β ^ 2)) ^ (j - j₀ + (j₀ - j))) := by
  set J := j - j₀ + (j₀ - j) with hJ
  set D := |x - ctr a δ j| with hD
  have hD0 : 0 ≤ D := abs_nonneg _
  have hJr : ((J : ℕ) : ℝ) = |(j : ℝ) - j₀| := by
    rcases le_total j j₀ with h | h
    · have hJ' : J = j₀ - j := by omega
      have hle : (j : ℝ) ≤ j₀ := Nat.cast_le.mpr h
      rw [hJ', Nat.cast_sub h, abs_of_nonpos (by linarith)]
      ring
    · have hJ' : J = j - j₀ := by omega
      have hle : (j₀ : ℝ) ≤ j := Nat.cast_le.mpr h
      rw [hJ', Nat.cast_sub h, abs_of_nonneg (by linarith)]
  have hcc : |ctr a δ j - ctr a δ j₀| = J * δ := by
    rw [hJr, show ctr a δ j - ctr a δ j₀ = ((j : ℝ) - j₀) * δ by simp only [ctr]; ring,
      abs_mul, abs_of_pos hδ]
  -- the distance to the leaf is between `Jδ - δ/2` and `Jδ + δ/2`
  have hDup : D ≤ (J + 1 / 2) * δ := by
    calc D = |(x - ctr a δ j₀) + (ctr a δ j₀ - ctr a δ j)| := by rw [hD]; ring_nf
      _ ≤ |x - ctr a δ j₀| + |ctr a δ j₀ - ctr a δ j| := abs_add_le _ _
      _ ≤ δ / 2 + J * δ := by rw [abs_sub_comm (ctr a δ j₀), hcc]; linarith
      _ = (J + 1 / 2) * δ := by ring
  have hDlo : (J : ℝ) * δ - δ / 2 ≤ D := by
    have h := abs_sub_abs_le_abs_sub (ctr a δ j - ctr a δ j₀) (x - ctr a δ j₀)
    rw [hcc, show ctr a δ j - ctr a δ j₀ - (x - ctr a δ j₀) = -(x - ctr a δ j) by ring,
      abs_neg] at h
    linarith
  have hsq : ((J : ℝ) - 3 / 4) * δ ^ 2 ≤ D ^ 2 := by
    rcases Nat.eq_zero_or_pos J with h0 | hpos
    · rw [h0]
      push_cast
      nlinarith [sq_nonneg D, sq_nonneg δ]
    · have hJ1 : (1 : ℝ) ≤ J := by exact_mod_cast hpos
      have hlo0 : 0 ≤ (J : ℝ) * δ - δ / 2 := by nlinarith
      have h2 := pow_le_pow_left₀ hlo0 hDlo 2
      nlinarith [sq_nonneg ((J : ℝ) - 1), sq_nonneg δ]
  -- the Gaussian factor is at most `e^{3/8β²} q^J`
  have hw : w1 (β * δ) (ctr a δ j) x ≤
      Real.exp (3 / (8 * β ^ 2)) * Real.exp (-1 / (2 * β ^ 2)) ^ J := by
    rw [← Real.exp_nat_mul, ← Real.exp_add]
    unfold w1
    apply Real.exp_le_exp.mpr
    have hlhs : -(x - ctr a δ j) ^ 2 / (2 * (β * δ) ^ 2) =
        -(1 / (2 * β ^ 2)) * (D ^ 2 / δ ^ 2) := by
      rw [hD, sq_abs]
      field_simp
    have hrhs : 3 / (8 * β ^ 2) + J * (-1 / (2 * β ^ 2)) = -(1 / (2 * β ^ 2)) * (J - 3 / 4) := by
      field_simp
      ring
    have hdiv : (J : ℝ) - 3 / 4 ≤ D ^ 2 / δ ^ 2 := by
      rw [le_div_iff₀ (by positivity)]
      exact hsq
    rw [hlhs, hrhs]
    exact mul_le_mul_of_nonpos_left hdiv (by rw [neg_nonpos]; positivity)
  calc w1 (β * δ) (ctr a δ j) x * D
      ≤ (Real.exp (3 / (8 * β ^ 2)) * Real.exp (-1 / (2 * β ^ 2)) ^ J) * ((J + 1 / 2) * δ) :=
        mul_le_mul hw hDup hD0 (by positivity)
    _ ≤ (Real.exp (3 / (8 * β ^ 2)) * Real.exp (-1 / (2 * β ^ 2)) ^ J) * ((J + 1) * δ) := by
        gcongr
        linarith
    _ = Real.exp (3 / (8 * β ^ 2)) * δ * ((J + 1) * Real.exp (-1 / (2 * β ^ 2)) ^ J) := by ring

/-- The one-dimensional normaliser is at least the weight of the nearest centre. -/
theorem sum_w1_ge {m : ℕ} {β δ a x : ℝ} (hβ : 0 < β) (hδ : 0 < δ) {j₀ : ℕ} (hj₀m : j₀ < m)
    (hj₀ : |x - ctr a δ j₀| ≤ δ / 2) :
    Real.exp (-1 / (8 * β ^ 2)) ≤ ∑ j ∈ range m, w1 (β * δ) (ctr a δ j) x := by
  refine le_trans ?_ (Finset.single_le_sum (fun j _ => (w1_pos _ _ _).le)
    (Finset.mem_range.mpr hj₀m))
  unfold w1
  apply Real.exp_le_exp.mpr
  have hsq : (x - ctr a δ j₀) ^ 2 ≤ (δ / 2) ^ 2 := by
    rw [← sq_abs]
    exact pow_le_pow_left₀ (abs_nonneg _) hj₀ 2
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith [mul_le_mul_of_nonneg_left hsq (by positivity : (0 : ℝ) ≤ 8 * β ^ 2)]

/-- The one-dimensional weighted distance is `O(δ)`. -/
theorem sum_w1_mul_le {m : ℕ} {β δ a x : ℝ} (hβ : 0 < β) (hδ : 0 < δ) {j₀ : ℕ} (hj₀m : j₀ < m)
    (hj₀ : |x - ctr a δ j₀| ≤ δ / 2) :
    ∑ j ∈ range m, w1 (β * δ) (ctr a δ j) x * |x - ctr a δ j| ≤
      Real.exp (3 / (8 * β ^ 2)) * δ * (2 / (1 - Real.exp (-1 / (2 * β ^ 2))) ^ 2) := by
  set q := Real.exp (-1 / (2 * β ^ 2)) with hq
  have hq0 : 0 ≤ q := (Real.exp_pos _).le
  have hq1 : q < 1 := Real.exp_lt_one_iff.mpr (by
    have : 0 < 1 / (2 * β ^ 2) := by positivity
    rw [neg_div]
    linarith)
  calc ∑ j ∈ range m, w1 (β * δ) (ctr a δ j) x * |x - ctr a δ j|
      ≤ ∑ j ∈ range m, Real.exp (3 / (8 * β ^ 2)) * δ *
          ((((j - j₀ + (j₀ - j) : ℕ) : ℝ) + 1) * q ^ (j - j₀ + (j₀ - j))) :=
        Finset.sum_le_sum fun j _ => w1_mul_le hβ hδ hj₀ j
    _ = Real.exp (3 / (8 * β ^ 2)) * δ *
          ∑ j ∈ range m, ((((j - j₀ + (j₀ - j) : ℕ) : ℝ) + 1) * q ^ (j - j₀ + (j₀ - j))) := by
        rw [Finset.mul_sum]
    _ ≤ Real.exp (3 / (8 * β ^ 2)) * δ * (2 * ∑ j ∈ range m, ((j : ℝ) + 1) * q ^ j) := by
        gcongr
        exact sum_natDist_le (fun J => ((J : ℝ) + 1) * q ^ J) (fun J => by positivity) hj₀m
    _ ≤ Real.exp (3 / (8 * β ^ 2)) * δ * (2 * (1 / (1 - q) ^ 2)) := by
        gcongr
        exact sum_succ_mul_pow_le hq0 hq1 m
    _ = Real.exp (3 / (8 * β ^ 2)) * δ * (2 / (1 - q) ^ 2) := by ring

/-! ## The grid tree -/

variable {d : ℕ}

/-- The cube `∏_i [a_i, a_i + w]`. -/
def cube (a : Fin d → ℝ) (w : ℝ) : Set (Fin d → ℝ) := {u | ∀ i, a i ≤ u i ∧ u i ≤ a i + w}

/-- The cell centres of the `m^d` grid of spacing `δ` on `∏_i [a_i, a_i + m δ]`. -/
noncomputable def gridPt (m : ℕ) (a : Fin d → ℝ) (δ : ℝ) (k : Fin d → Fin m) : Fin d → ℝ :=
  fun i => ctr (a i) δ (k i)

/-- **The grid tree** of Prop. rate: the depth-one tree whose leaves hold `(γ, g γ)` at the
`m^d` cell centres `γ`, count one each, with variance floor `h`, queried at `x`. -/
noncomputable def gridTree (m : ℕ) (a : Fin d → ℝ) (δ h : ℝ) (g : (Fin d → ℝ) → ℝ)
    (x : Fin d → ℝ) : ℝ :=
  leafMixture (univ : Finset (Fin d → Fin m)) (fun k => WithLp.toLp 2 (gridPt m a δ k))
    (fun k => g (gridPt m a δ k)) (fun _ => 1) h (WithLp.toLp 2 x)

/-- The constant `C(d, β) = 2 d e^{1/2β²} / (1 - e^{-1/2β²})²` of Prop. rate. -/
noncomputable def rateConst (d : ℕ) (β : ℝ) : ℝ :=
  2 * d * Real.exp (1 / (2 * β ^ 2)) / (1 - Real.exp (-1 / (2 * β ^ 2))) ^ 2

theorem rateConst_mono {d d' : ℕ} (hdd : d ≤ d') (β : ℝ) : rateConst d β ≤ rateConst d' β := by
  unfold rateConst
  gcongr

theorem rateConst_nonneg (d : ℕ) (β : ℝ) : 0 ≤ rateConst d β := by
  unfold rateConst
  positivity

/-- The Gaussian weight of a leaf is a product of one-dimensional weights. -/
theorem gauss_dist_toLp (h : ℝ) (x y : Fin d → ℝ) :
    gauss h (dist (WithLp.toLp 2 x : EuclideanSpace ℝ (Fin d)) (WithLp.toLp 2 y)) =
      ∏ i, w1 h (y i) (x i) := by
  unfold gauss w1
  rw [EuclideanSpace.dist_sq_eq, ← Real.exp_sum]
  congr 1
  simp only [Real.dist_eq, sq_abs]
  rw [← Finset.sum_neg_distrib, Finset.sum_div]

/-- The normaliser of the grid tree factorises over the coordinates. -/
theorem sum_gauss_grid (m : ℕ) (a : Fin d → ℝ) (δ h : ℝ) (x : Fin d → ℝ) :
    ∑ k : Fin d → Fin m, gauss h (dist (WithLp.toLp 2 x : EuclideanSpace ℝ (Fin d))
      (WithLp.toLp 2 (gridPt m a δ k))) = ∏ i, ∑ j : Fin m, w1 h (ctr (a i) δ j) (x i) := by
  rw [Finset.prod_univ_sum, Fintype.piFinset_univ]
  exact Finset.sum_congr rfl fun k _ => gauss_dist_toLp h x _

/-- The weighted distances along one coordinate factorise too. -/
theorem sum_gauss_grid_mul (m : ℕ) (a : Fin d → ℝ) (δ h : ℝ) (x : Fin d → ℝ) (i₀ : Fin d) :
    ∑ k : Fin d → Fin m, gauss h (dist (WithLp.toLp 2 x : EuclideanSpace ℝ (Fin d))
      (WithLp.toLp 2 (gridPt m a δ k))) * |x i₀ - gridPt m a δ k i₀| =
      (∑ j : Fin m, w1 h (ctr (a i₀) δ j) (x i₀) * |x i₀ - ctr (a i₀) δ j|) *
        ∏ i ∈ univ.erase i₀, ∑ j : Fin m, w1 h (ctr (a i) δ j) (x i) := by
  classical
  set f : Fin d → Fin m → ℝ := fun i j =>
    w1 h (ctr (a i) δ j) (x i) * (if i = i₀ then |x i₀ - ctr (a i₀) δ j| else 1) with hf
  have hL : ∀ k : Fin d → Fin m, gauss h (dist (WithLp.toLp 2 x : EuclideanSpace ℝ (Fin d))
      (WithLp.toLp 2 (gridPt m a δ k))) * |x i₀ - gridPt m a δ k i₀| = ∏ i, f i (k i) := by
    intro k
    rw [gauss_dist_toLp, hf]
    simp only
    rw [Finset.prod_mul_distrib]
    congr 1
    rw [Finset.prod_ite_eq']
    simp [gridPt]
  have hR : (∑ j : Fin m, w1 h (ctr (a i₀) δ j) (x i₀) * |x i₀ - ctr (a i₀) δ j|) *
      ∏ i ∈ univ.erase i₀, ∑ j : Fin m, w1 h (ctr (a i) δ j) (x i) = ∏ i, ∑ j, f i j := by
    rw [← Finset.mul_prod_erase univ (fun i => ∑ j, f i j) (mem_univ i₀)]
    congr 1
    · simp [hf]
    · refine Finset.prod_congr rfl fun i hi => ?_
      simp [hf, Finset.ne_of_mem_erase hi]
  rw [Finset.sum_congr rfl fun k _ => hL k, hR, Finset.prod_univ_sum, Fintype.piFinset_univ]

/-- **Prop. rate: the grid rate of one tree.**  If `g` is `L`-Lipschitz on the cube
`Q = ∏_i [a_i, a_i + w]` for the `ℓ¹` distance (as is every `L`-Lipschitz map for the Euclidean
or the max norm), the grid tree with `m^d` leaves and variance floor `h = β w / m` predicts
within `C(d, β) L w / m` of `g` on all of `Q`. -/
theorem grid_rate {m : ℕ} (hm : 0 < m) (a : Fin d → ℝ) {w β L : ℝ} (hw : 0 < w) (hβ : 0 < β)
    (hL : 0 ≤ L) (g : (Fin d → ℝ) → ℝ)
    (hg : ∀ u ∈ cube a w, ∀ v ∈ cube a w, |g u - g v| ≤ L * ∑ i, |u i - v i|)
    {x : Fin d → ℝ} (hx : x ∈ cube a w) :
    |gridTree m a (w / m) (β * (w / m)) g x - g x| ≤ rateConst d β * L * w / m := by
  have : Nonempty (Fin m) := ⟨⟨0, hm⟩⟩
  set δ := w / m with hδ_def
  have hm' : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  have hδ : 0 < δ := by positivity
  have hmδ : (m : ℝ) * δ = w := by rw [hδ_def]; field_simp
  set q := Real.exp (-1 / (2 * β ^ 2)) with hq
  have hq1 : q < 1 := Real.exp_lt_one_iff.mpr (by
    have : 0 < 1 / (2 * β ^ 2) := by positivity
    rw [neg_div]
    linarith)
  have h1q : 0 < 1 - q := by linarith
  -- the one-dimensional sums
  set S : Fin d → ℝ := fun i => ∑ j : Fin m, w1 (β * δ) (ctr (a i) δ j) (x i) with hS
  set T : Fin d → ℝ := fun i =>
    ∑ j : Fin m, w1 (β * δ) (ctr (a i) δ j) (x i) * |x i - ctr (a i) δ j| with hT
  set B : ℝ := 2 * δ * Real.exp (1 / (2 * β ^ 2)) / (1 - q) ^ 2 with hB
  have hS0 : ∀ i, 0 < S i := fun i =>
    Finset.sum_pos (fun j _ => w1_pos _ _ _) Finset.univ_nonempty
  have hTS : ∀ i, T i ≤ B * S i := by
    intro i
    obtain ⟨j₀, hj₀m, hj₀⟩ := exists_near_ctr hm hδ (hx i).1 (by rw [hmδ]; exact (hx i).2)
    have hT' : T i ≤ Real.exp (3 / (8 * β ^ 2)) * δ * (2 / (1 - q) ^ 2) := by
      rw [hT]
      simp only
      rw [Fin.sum_univ_eq_sum_range (fun j => w1 (β * δ) (ctr (a i) δ j) (x i) *
        |x i - ctr (a i) δ j|) m]
      exact sum_w1_mul_le hβ hδ hj₀m hj₀
    have hS' : Real.exp (-1 / (8 * β ^ 2)) ≤ S i := by
      rw [hS]
      simp only
      rw [Fin.sum_univ_eq_sum_range (fun j => w1 (β * δ) (ctr (a i) δ j) (x i)) m]
      exact sum_w1_ge hβ hδ hj₀m hj₀
    have hexp : Real.exp (3 / (8 * β ^ 2)) =
        Real.exp (1 / (2 * β ^ 2)) * Real.exp (-1 / (8 * β ^ 2)) := by
      rw [← Real.exp_add]
      congr 1
      field_simp
      ring
    calc T i ≤ Real.exp (3 / (8 * β ^ 2)) * δ * (2 / (1 - q) ^ 2) := hT'
      _ = B * Real.exp (-1 / (8 * β ^ 2)) := by rw [hexp, hB]; ring
      _ ≤ B * S i := mul_le_mul_of_nonneg_left hS' (by positivity)
  -- the leaf weights
  set G : (Fin d → Fin m) → ℝ := fun k =>
    1 * gauss (β * δ) (dist (WithLp.toLp 2 x : EuclideanSpace ℝ (Fin d))
      (WithLp.toLp 2 (gridPt m a δ k))) with hG
  have hG0 : ∀ k ∈ (univ : Finset (Fin d → Fin m)), 0 ≤ G k := fun k _ =>
    mul_nonneg zero_le_one (gauss_pos _ _).le
  have hGsum : ∑ k, G k = ∏ i, S i := by
    rw [hG]
    simp only [one_mul]
    exact sum_gauss_grid m a δ (β * δ) x
  have hGpos : 0 < ∑ k, G k := by
    rw [hGsum]
    exact Finset.prod_pos fun i _ => hS0 i
  have hmix : gridTree m a δ (β * δ) g x = (∑ k, G k * g (gridPt m a δ k)) / ∑ k, G k := rfl
  -- the centres lie in the cube
  have hcube : ∀ k : Fin d → Fin m, gridPt m a δ k ∈ cube a w := by
    intro k i
    have hk : ((k i : ℕ) : ℝ) + 1 / 2 ≤ m := by
      have : ((k i : ℕ) : ℝ) + 1 ≤ m := by exact_mod_cast (k i).isLt
      linarith
    simp only [gridPt, ctr]
    constructor
    · have : 0 ≤ (((k i : ℕ) : ℝ) + 1 / 2) * δ := by positivity
      linarith
    · have : (((k i : ℕ) : ℝ) + 1 / 2) * δ ≤ m * δ := mul_le_mul_of_nonneg_right hk hδ.le
      linarith
  rw [hmix]
  refine (abs_wavg_sub_le _ G _ _ hG0 hGpos).trans ?_
  rw [div_le_iff₀ hGpos]
  have hsplit : ∀ i₀, ∑ k, G k * |x i₀ - gridPt m a δ k i₀| ≤ B * ∑ k, G k := by
    intro i₀
    have h1 : ∑ k, G k * |x i₀ - gridPt m a δ k i₀| =
        T i₀ * ∏ i ∈ univ.erase i₀, S i := by
      rw [hG]
      simp only [one_mul]
      exact sum_gauss_grid_mul m a δ (β * δ) x i₀
    rw [h1, hGsum, ← Finset.mul_prod_erase univ S (mem_univ i₀), ← mul_assoc]
    exact mul_le_mul_of_nonneg_right (hTS i₀) (Finset.prod_nonneg fun i _ => (hS0 i).le)
  calc ∑ k, G k * |g (gridPt m a δ k) - g x|
      ≤ ∑ k, G k * (L * ∑ i, |x i - gridPt m a δ k i|) := by
        refine Finset.sum_le_sum fun k _ => mul_le_mul_of_nonneg_left ?_ (hG0 k (mem_univ k))
        have := hg _ (hcube k) x hx
        simpa only [abs_sub_comm (gridPt m a δ k _)] using this
    _ = L * ∑ k, ∑ i, G k * |x i - gridPt m a δ k i| := by
        simp only [Finset.mul_sum]
        exact Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun i _ => by ring
    _ = L * ∑ i, ∑ k, G k * |x i - gridPt m a δ k i| := by rw [Finset.sum_comm]
    _ ≤ L * ∑ _i : Fin d, B * ∑ k, G k := by
        gcongr with i
        exact hsplit i
    _ = rateConst d β * L * w / m * ∑ k, G k := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, hB,
          rateConst, ← hmδ, ← hq]
        field_simp

end CobwebForest
