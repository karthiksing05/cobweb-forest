/-
Copyright (c) 2026 Cobweb Forest authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Cobweb Forest authors
-/
import CobwebForest.Compose
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Separation: one tree cannot beat `G^{-1/n}` on compositions (theory: Thm. separation)

`𝓑_n` (`InB n`) is the class of functions computed by fan-in-two compositions of depth at most
`⌈log₂ n⌉ + 2` whose maps are `1`-Lipschitz in the max-norm.

* `separation_a` (Thm. separation (a)): for every set `Γ` of `G` stored points there is
  `f ∈ 𝓑_n` such that every predictor whose output is a weighted combination of the stored
  values `f(γ)`, `γ ∈ Γ` -- as every prediction rule of a Cobweb tree is (Def. multi-node
  prediction: each concept mean is an average of stored values and every rule mixes concept
  means, or outputs `0` or the root mean when a leaves-only mixture is empty) -- errs by at least
  `G^{-1/n}/4` at some point of `[0,1]^n`.
* `numTrees_le` (used in Thm. separation (b)): a fan-in-two forest of depth `D` has at most
  `2^D - 1` trees.

The proof of (a): with `m = ⌊G^{1/n}⌋ + 1` we have `m^n > G`, so one of the `m^n` cubes of side
`1/m` holds no stored point (`exists_empty_cube`); `f` is the bump `(ρ - ‖x - c‖_∞)_+` at its
centre `c`, `ρ = 1/(2m) ≥ G^{-1/n}/4`.  It vanishes on `Γ`, so every such predictor is `0`, while
`f(c) = ρ`.  The bump is `ramp ∘ (binary max tree) ∘ (|x_i - c_i|)_i` (`bumpNet`).
-/

open Finset

namespace CobwebForest

variable {n : ℕ}

/-- The bump `f(x) = (ρ - ‖x - c‖_∞)_+` of Thm. separation. -/
noncomputable def bump (c : Fin n → ℝ) (ρ : ℝ) (x : Fin n → ℝ) : ℝ := max (ρ - dist x c) 0

/-! ## Counting -/

/-- The index of the cube of side `1/m` that contains `γ` (clamped to the grid). -/
noncomputable def cubeIdx (m : ℕ) (hm : 0 < m) (γ : Fin n → ℝ) : Fin n → Fin m :=
  fun j => ⟨min ⌊γ j * m⌋₊ (m - 1), by omega⟩

/-- The centre of the cube with index `i`. -/
noncomputable def centre (m : ℕ) (i : Fin n → Fin m) : Fin n → ℝ :=
  fun j => (((i j : ℕ) : ℝ) + 1 / 2) / m

/-- A point closer than `1/(2m)` (max-norm) to the centre of a cube lies in that cube. -/
theorem cubeIdx_eq_of_dist_lt {m : ℕ} (hm : 0 < m) (i : Fin n → Fin m) {γ : Fin n → ℝ}
    (h : dist γ (centre m i) < 1 / (2 * m)) : cubeIdx m hm γ = i := by
  have hm' : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  funext j
  have hj : |γ j - centre m i j| < 1 / (2 * m) := by
    have := (dist_pi_lt_iff (by positivity)).mp h j
    rwa [Real.dist_eq] at this
  have key : γ j * m - (((i j : ℕ) : ℝ) + 1 / 2) = (γ j - centre m i j) * m := by
    simp only [centre]
    field_simp
  have hb : |γ j * m - (((i j : ℕ) : ℝ) + 1 / 2)| < 1 / 2 := by
    rw [key, abs_mul, abs_of_pos hm']
    calc |γ j - centre m i j| * m < 1 / (2 * m) * m := mul_lt_mul_of_pos_right hj hm'
      _ = 1 / 2 := by field_simp
  rw [abs_lt] at hb
  have h1 : ((i j : ℕ) : ℝ) ≤ γ j * m := by linarith
  have h2 : γ j * m < ((i j : ℕ) : ℝ) + 1 := by linarith
  have hfl : ⌊γ j * m⌋₊ = (i j : ℕ) :=
    (Nat.floor_eq_iff (le_trans (Nat.cast_nonneg _) h1)).mpr ⟨h1, h2⟩
  apply Fin.ext
  simp only [cubeIdx, hfl]
  have := (i j).isLt
  omega

/-- **Counting.**  If `|Γ| < m^n`, some cube of the `m^n` grid holds no point of `Γ`. -/
theorem exists_empty_cube (Γ : Finset (Fin n → ℝ)) {m : ℕ} (hm : 0 < m) (hG : Γ.card < m ^ n) :
    ∃ i : Fin n → Fin m, ∀ γ ∈ Γ, cubeIdx m hm γ ≠ i := by
  classical
  have hcard : (Γ.image (cubeIdx m hm)).card < (univ : Finset (Fin n → Fin m)).card := by
    calc (Γ.image (cubeIdx m hm)).card ≤ Γ.card := Finset.card_image_le
      _ < m ^ n := hG
      _ = (univ : Finset (Fin n → Fin m)).card := by simp
  obtain ⟨i, -, hi⟩ := Finset.exists_mem_notMem_of_card_lt_card hcard
  exact ⟨i, fun γ hγ h => hi (h ▸ Finset.mem_image_of_mem _ hγ)⟩

/-- With `m = ⌊G^{1/n}⌋ + 1`, `G < m^n`. -/
theorem card_lt_pow (G : ℕ) (hn : n ≠ 0) : G < (⌊(G : ℝ) ^ ((n : ℝ)⁻¹)⌋₊ + 1) ^ n := by
  have hx : (0 : ℝ) ≤ (G : ℝ) ^ ((n : ℝ)⁻¹) := by positivity
  have h1 : (G : ℝ) ^ ((n : ℝ)⁻¹) < ((⌊(G : ℝ) ^ ((n : ℝ)⁻¹)⌋₊ + 1 : ℕ) : ℝ) := by
    push_cast
    exact Nat.lt_floor_add_one _
  have h2 := pow_lt_pow_left₀ h1 hx hn
  rw [Real.rpow_inv_natCast_pow (Nat.cast_nonneg G) hn] at h2
  exact_mod_cast h2

/-- With `m = ⌊G^{1/n}⌋ + 1`, `1/(2m) ≥ G^{-1/n}/4`. -/
theorem rho_bound (G : ℕ) (hn : n ≠ 0) :
    (G : ℝ) ^ (-(n : ℝ)⁻¹) / 4 ≤ 1 / (2 * ((⌊(G : ℝ) ^ ((n : ℝ)⁻¹)⌋₊ + 1 : ℕ) : ℝ)) := by
  rcases Nat.eq_zero_or_pos G with rfl | hG
  · rw [Nat.cast_zero, Real.zero_rpow (by simp [hn]), zero_div]
    positivity
  · set a := (G : ℝ) ^ ((n : ℝ)⁻¹) with ha_def
    have ha1 : 1 ≤ a := Real.one_le_rpow (by exact_mod_cast hG) (by positivity)
    have hm : ((⌊a⌋₊ + 1 : ℕ) : ℝ) ≤ 2 * a := by
      push_cast
      linarith [Nat.floor_le (by linarith : (0 : ℝ) ≤ a)]
    rw [Real.rpow_neg (Nat.cast_nonneg G), ← ha_def]
    have hm0 : (0 : ℝ) < ((⌊a⌋₊ + 1 : ℕ) : ℝ) := by positivity
    calc a⁻¹ / 4 = 1 / (2 * (2 * a)) := by field_simp; ring
      _ ≤ 1 / (2 * ((⌊a⌋₊ + 1 : ℕ) : ℝ)) := by
        apply one_div_le_one_div_of_le (by positivity)
        linarith

/-! ## The bump is a fan-in-two composition of `1`-Lipschitz maps -/

namespace Net

/-- Every vertex map is `L`-Lipschitz (max-norm) on all of `ℝ^k`. -/
inductive AllLip (L : NNReal) : Net n → Prop
  | raw (i : Fin n) : AllLip L (raw i)
  | node {k : ℕ} {args : Fin k → Net n} {g : (Fin k → ℝ) → ℝ} (hg : LipschitzWith L g)
      (hargs : ∀ j, AllLip L (args j)) : AllLip L (node k args g)

/-- `u ↦ |u₀ - a|`. -/
noncomputable def absMap (a : ℝ) : (Fin 1 → ℝ) → ℝ := fun u => |u 0 - a|
/-- `(u₀, u₁) ↦ max u₀ u₁`. -/
noncomputable def maxMap : (Fin 2 → ℝ) → ℝ := fun u => max (u 0) (u 1)
/-- The ramp `t ↦ (ρ - t)_+`. -/
noncomputable def rampMap (ρ : ℝ) : (Fin 1 → ℝ) → ℝ := fun u => max (ρ - u 0) 0
/-- The constant `0` (a tree with no input). -/
def zeroMap : (Fin 0 → ℝ) → ℝ := fun _ => 0

theorem lipschitz_absMap (a : ℝ) : LipschitzWith 1 (absMap a) := by
  refine LipschitzWith.of_dist_le_mul fun u v => ?_
  simp only [absMap, Real.dist_eq, NNReal.coe_one, one_mul]
  calc |(|u 0 - a| - |v 0 - a|)| ≤ |(u 0 - a) - (v 0 - a)| := abs_abs_sub_abs_le_abs_sub _ _
    _ = |u 0 - v 0| := by congr 1; ring
    _ = dist (u 0) (v 0) := (Real.dist_eq _ _).symm
    _ ≤ dist u v := dist_le_pi_dist u v 0

theorem lipschitz_maxMap : LipschitzWith 1 maxMap := by
  refine LipschitzWith.of_dist_le_mul fun u v => ?_
  simp only [maxMap, Real.dist_eq, NNReal.coe_one, one_mul]
  calc |max (u 0) (u 1) - max (v 0) (v 1)| ≤ max |u 0 - v 0| |u 1 - v 1| :=
        abs_max_sub_max_le_max _ _ _ _
    _ ≤ dist u v := by
        rw [← Real.dist_eq, ← Real.dist_eq]
        exact max_le (dist_le_pi_dist u v 0) (dist_le_pi_dist u v 1)

theorem lipschitz_rampMap (ρ : ℝ) : LipschitzWith 1 (rampMap ρ) := by
  refine LipschitzWith.of_dist_le_mul fun u v => ?_
  simp only [rampMap, Real.dist_eq, NNReal.coe_one, one_mul]
  calc |max (ρ - u 0) 0 - max (ρ - v 0) 0| ≤ max |(ρ - u 0) - (ρ - v 0)| |(0 : ℝ) - 0| :=
        abs_max_sub_max_le_max _ _ _ _
    _ = |v 0 - u 0| := by
        rw [sub_self, abs_zero, max_eq_left (abs_nonneg _)]
        congr 1
        ring
    _ = dist (u 0) (v 0) := by rw [Real.dist_eq, abs_sub_comm]
    _ ≤ dist u v := dist_le_pi_dist u v 0

theorem lipschitz_zeroMap : LipschitzWith 1 zeroMap :=
  (LipschitzWith.const (0 : ℝ)).weaken zero_le_one

/-- A balanced binary tree of `max` over the leaves `|x_i - c_i|`, `o ≤ i < o + 2^m`
(a missing leaf is the constant `0`). -/
noncomputable def maxTree (c : Fin n → ℝ) : ℕ → ℕ → Net n
  | 0, o => if h : o < n then node 1 ![raw ⟨o, h⟩] (absMap (c ⟨o, h⟩)) else node 0 ![] zeroMap
  | m + 1, o => node 2 ![maxTree c m o, maxTree c m (o + 2 ^ m)] maxMap

/-- The bump `(ρ - ‖x - c‖_∞)_+` as a fan-in-two composition. -/
noncomputable def bumpNet (c : Fin n → ℝ) (ρ : ℝ) : Net n :=
  node 1 ![maxTree c (Nat.clog 2 n) 0] (rampMap ρ)

/-- The leaves `o ≤ i < o + 2^m` of a max tree. -/
def win (n o m : ℕ) : Finset (Fin n) :=
  Finset.filter (fun i : Fin n => o ≤ (i : ℕ) ∧ (i : ℕ) < o + 2 ^ m) univ

theorem mem_win {o m : ℕ} {i : Fin n} : i ∈ win n o m ↔ o ≤ (i : ℕ) ∧ (i : ℕ) < o + 2 ^ m := by
  simp [win]

theorem eval_maxTree (c x : Fin n → ℝ) : ∀ m o, eval (maxTree c m o) x =
    (((win n o m).sup fun i => nndist (x i) (c i) : NNReal) : ℝ) := by
  intro m
  induction m with
  | zero =>
    intro o
    by_cases h : o < n
    · have hw : win n o 0 = {⟨o, h⟩} := by
        ext i
        rw [mem_win, mem_singleton, Fin.ext_iff, pow_zero]
        simp only
        omega
      rw [hw, Finset.sup_singleton, coe_nndist, Real.dist_eq]
      simp [maxTree, h, absMap]
    · have hw : win n o 0 = ∅ := by
        ext i
        rw [mem_win, pow_zero]
        simp only [Finset.notMem_empty, iff_false]
        have := i.isLt
        omega
      rw [hw, Finset.sup_empty]
      simp [maxTree, h, zeroMap]
  | succ m ih =>
    intro o
    have hw : win n o (m + 1) = win n o m ∪ win n (o + 2 ^ m) m := by
      ext i
      rw [Finset.mem_union, mem_win, mem_win, mem_win, pow_succ]
      omega
    rw [hw, Finset.sup_union, NNReal.coe_max, ← ih o, ← ih (o + 2 ^ m)]
    simp [maxTree, maxMap]

theorem eval_bumpNet (c : Fin n → ℝ) (ρ : ℝ) (x : Fin n → ℝ) :
    eval (bumpNet c ρ) x = bump c ρ x := by
  have hwin : win n 0 (Nat.clog 2 n) = univ := by
    ext i
    rw [mem_win]
    have := Nat.le_pow_clog one_lt_two n
    have := i.isLt
    simp only [mem_univ, iff_true]
    omega
  have hd : eval (maxTree c (Nat.clog 2 n) 0) x = dist x c := by
    rw [eval_maxTree, hwin, dist_pi_def]
  simp [bumpNet, rampMap, bump, hd]

theorem depth_maxTree (c : Fin n → ℝ) : ∀ m o, depth (maxTree c m o) ≤ m + 1 := by
  intro m
  induction m with
  | zero =>
    intro o
    by_cases h : o < n
    · simp only [maxTree, h, dite_true]
      exact depth_node_le (d := 0) (Fin.forall_fin_one.mpr le_rfl)
    · simp only [maxTree, h, dite_false]
      exact depth_node_le (d := 0) fun j => j.elim0
  | succ m ih =>
    intro o
    exact depth_node_le (Fin.forall_fin_two.mpr ⟨ih o, ih (o + 2 ^ m)⟩)

theorem fanIn_maxTree (c : Fin n → ℝ) : ∀ m o, FanInLE 2 (maxTree c m o) := by
  intro m
  induction m with
  | zero =>
    intro o
    by_cases h : o < n
    · simp only [maxTree, h, dite_true]
      exact .node (by norm_num) (Fin.forall_fin_one.mpr (.raw _))
    · simp only [maxTree, h, dite_false]
      exact .node (by norm_num) fun j => j.elim0
  | succ m ih =>
    intro o
    exact .node le_rfl (Fin.forall_fin_two.mpr ⟨ih o, ih (o + 2 ^ m)⟩)

theorem lip_maxTree (c : Fin n → ℝ) : ∀ m o, AllLip 1 (maxTree c m o) := by
  intro m
  induction m with
  | zero =>
    intro o
    by_cases h : o < n
    · simp only [maxTree, h, dite_true]
      exact .node (lipschitz_absMap _) (Fin.forall_fin_one.mpr (.raw _))
    · simp only [maxTree, h, dite_false]
      exact .node lipschitz_zeroMap fun j => j.elim0
  | succ m ih =>
    intro o
    exact .node lipschitz_maxMap (Fin.forall_fin_two.mpr ⟨ih o, ih (o + 2 ^ m)⟩)

end Net

open Net

/-- `𝓑_n`: the functions computed by fan-in-two compositions of depth at most `⌈log₂ n⌉ + 2`
whose maps are `1`-Lipschitz in the max-norm. -/
def InB (n : ℕ) (f : (Fin n → ℝ) → ℝ) : Prop :=
  ∃ t : Net n, FanInLE 2 t ∧ depth t ≤ Nat.clog 2 n + 2 ∧ AllLip 1 t ∧ ∀ x, eval t x = f x

/-- The bump is in `𝓑_n`. -/
theorem bump_mem_B (c : Fin n → ℝ) (ρ : ℝ) : InB n (bump c ρ) :=
  ⟨bumpNet c ρ, .node (by norm_num) (Fin.forall_fin_one.mpr (fanIn_maxTree c _ 0)),
    depth_node_le (Fin.forall_fin_one.mpr (depth_maxTree c _ 0)),
    .node (lipschitz_rampMap ρ) (Fin.forall_fin_one.mpr (lip_maxTree c _ 0)),
    eval_bumpNet c ρ⟩

/-- For every set `Γ` of `G` points there is a bump `f ∈ 𝓑_n` vanishing on `Γ` whose value at a
point of `[0,1]^n` is at least `G^{-1/n}/4`. -/
theorem exists_bump (hn : 0 < n) (Γ : Finset (Fin n → ℝ)) :
    ∃ f : (Fin n → ℝ) → ℝ, InB n f ∧ (∀ γ ∈ Γ, f γ = 0) ∧
      ∃ c ∈ Set.Icc (0 : Fin n → ℝ) 1, (Γ.card : ℝ) ^ (-(n : ℝ)⁻¹) / 4 ≤ f c := by
  set m := ⌊(Γ.card : ℝ) ^ ((n : ℝ)⁻¹)⌋₊ + 1 with hm_def
  have hm : 0 < m := Nat.succ_pos _
  have hm' : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  obtain ⟨i, hi⟩ := exists_empty_cube Γ hm (card_lt_pow Γ.card hn.ne')
  set c := centre m i with hc_def
  set ρ : ℝ := 1 / (2 * m) with hρ_def
  have hρ : 0 < ρ := by positivity
  refine ⟨bump c ρ, bump_mem_B c ρ, fun γ hγ => ?_, c, ?_, ?_⟩
  · -- every stored point is at distance `≥ ρ` from `c`
    have hfar : ρ ≤ dist γ c := by
      by_contra hlt
      exact hi γ hγ (cubeIdx_eq_of_dist_lt hm i (not_le.mp hlt))
    simp only [bump]
    exact max_eq_right (by linarith)
  · -- `c ∈ [0,1]^n`
    constructor <;> intro j <;> simp only [Pi.zero_apply, Pi.one_apply, hc_def, centre]
    · positivity
    · rw [div_le_one hm']
      have : ((i j : ℕ) : ℝ) + 1 ≤ m := by exact_mod_cast (i j).isLt
      linarith
  · have hc : bump c ρ c = ρ := by simp [bump, hρ.le]
    rw [hc]
    exact rho_bound Γ.card hn.ne'

/-- **Thm. separation (a).**  Let `Γ` be the `G` points at which a tree stores examples of its
target.  There is `f ∈ 𝓑_n` such that every predictor whose output at every query is a
weighted combination `∑_{γ ∈ Γ} w_γ f(γ)` of the stored values (with weights depending in any
way on the query and on the stored values) errs by at least `G^{-1/n}/4` somewhere on
`[0,1]^n`. -/
theorem separation_a (hn : 0 < n) (Γ : Finset (Fin n → ℝ)) :
    ∃ f : (Fin n → ℝ) → ℝ, InB n f ∧
      ∀ pred : ((Fin n → ℝ) → ℝ) → (Fin n → ℝ) → ℝ,
        (∀ F x, ∃ w : (Fin n → ℝ) → ℝ, pred F x = ∑ γ ∈ Γ, w γ * F γ) →
        ∃ x ∈ Set.Icc (0 : Fin n → ℝ) 1, (Γ.card : ℝ) ^ (-(n : ℝ)⁻¹) / 4 ≤ |pred f x - f x| := by
  obtain ⟨f, hfB, hf0, c, hc, hfc⟩ := exists_bump hn Γ
  refine ⟨f, hfB, fun pred hpred => ⟨c, hc, ?_⟩⟩
  obtain ⟨w, hw⟩ := hpred f c
  have hzero : pred f c = 0 := by
    rw [hw]
    exact Finset.sum_eq_zero fun γ hγ => by rw [hf0 γ hγ, mul_zero]
  rw [hzero, zero_sub, abs_neg]
  exact hfc.trans (le_abs_self _)

/-! ## Tree count -/

namespace Net

/-- The number of trees (non-source vertices). -/
def numTrees : Net n → ℕ
  | raw _ => 0
  | node _ args _ => (∑ j, numTrees (args j)) + 1

theorem numTrees_node {k : ℕ} (args : Fin k → Net n) (g : (Fin k → ℝ) → ℝ) :
    numTrees (node k args g) = (∑ j, numTrees (args j)) + 1 := rfl

/-- A fan-in-two forest of depth `D` has at most `2^D - 1` trees. -/
theorem numTrees_le : ∀ t : Net n, FanInLE 2 t → numTrees t + 1 ≤ 2 ^ depth t := by
  intro t
  induction t with
  | raw i => intro _; simp [numTrees, depth]
  | node k args g ih =>
    intro ht
    rcases ht with _ | ⟨hk, hargs⟩
    rw [numTrees_node, depth_node, pow_succ]
    set d := Finset.univ.sup fun j => depth (args j)
    have hsum : ∑ j, (numTrees (args j) + 1) ≤ ∑ _j : Fin k, 2 ^ d :=
      Finset.sum_le_sum fun j _ =>
        (ih j (hargs j)).trans (Nat.pow_le_pow_right (by norm_num) (depth_arg_le args j))
    rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      smul_eq_mul, mul_one, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      smul_eq_mul] at hsum
    obtain ⟨Q, hQ⟩ : ∃ Q, 2 ^ d = Q + 1 :=
      ⟨2 ^ d - 1, by have := Nat.one_le_two_pow (n := d); omega⟩
    rw [hQ] at hsum ⊢
    have h2 : k * Q ≤ 2 * Q := Nat.mul_le_mul_right Q hk
    nlinarith

end Net

end CobwebForest
