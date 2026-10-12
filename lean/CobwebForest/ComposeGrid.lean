/-
Copyright (c) 2026 Cobweb Forest authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Cobweb Forest authors
-/
import CobwebForest.Rate
import CobwebForest.Separation

/-!
# Composition with grid trees, and the forest side of the separation
(theory: Thm. compose, Thm. separation (b))

* `compose_grid` (Thm. compose, full form): let `f` be a `k`-composition on `K` whose every vertex
  map `g_v` is `L`-Lipschitz (max-norm) on `B_v^r`, which lies in a cube `Q_v` of side `w`
  (`InCubes`).  Give every vertex the grid tree of Prop. rate for the McShane extension of
  `g_v` to `Q_v` (`m^{k_v} ≤ m^k` leaves).  If `η ∑_{l<D} L^l ≤ r` with `η = C(k, β) L w / m`,
  the resulting Cobweb forest has the shape of the composition and is within
  `η ∑_{l<D} L^l = C L w (∑_{l<D} L^l) G^{-1/k}` of `f` on `K` (`G = m^k`).
* `separation_b` (Thm. separation (b)): every `f ∈ 𝓑_n` is approximated on `[0,1]^n` by a Cobweb
  forest of at most `2^D - 1 ≤ 8n - 1` trees (`D = ⌈log₂ n⌉ + 2`), with at most two inputs and
  `G = m²` leaves each, within `C D (1 + 2r) G^{-1/2}` once this is below the margin `r`.
-/

open Metric Finset

namespace CobwebForest

namespace Net

variable {n : ℕ}

/-- Every vertex's input set, thickened by `r` in the max-norm, lies in a cube of side `w`. -/
inductive InCubes (K : Set (Fin n → ℝ)) (r w : ℝ) : Net n → Prop
  | raw (i : Fin n) : InCubes K r w (raw i)
  | node {k : ℕ} {args : Fin k → Net n} {g : (Fin k → ℝ) → ℝ} (a : Fin k → ℝ)
      (ha : cthickening r (inSet K args) ⊆ cube a w)
      (hargs : ∀ j, InCubes K r w (args j)) : InCubes K r w (node k args g)

theorem Close.depth_eq {K : Set (Fin n → ℝ)} {r η : ℝ} {t' t : Net n} (h : Close K r η t' t) :
    depth t' = depth t := by
  induction h with
  | raw i => rfl
  | node _ _ ih => simp only [depth_node, ih]

theorem Close.numTrees_eq {K : Set (Fin n → ℝ)} {r η : ℝ} {t' t : Net n}
    (h : Close K r η t' t) : numTrees t' = numTrees t := by
  induction h with
  | raw i => rfl
  | node _ _ ih => simp only [numTrees_node, ih]

/-- **The tree of Prop. rate at one vertex.**  If `g` is `L`-Lipschitz (max-norm) on a set `C`
inside the cube `Q = cube a w`, the grid tree with `m^k` leaves for the McShane extension of
`g` to `ℝ^k` is a Cobweb tree within `C(k, β) L w / m` of `g` on `C`. -/
theorem exists_gridTree {k m : ℕ} (hm : 0 < m) {w β : ℝ} (hw : 0 < w) (hβ : 0 < β)
    {L : NNReal} {C : Set (Fin k → ℝ)} {a : Fin k → ℝ} (hC : C ⊆ cube a w)
    {g : (Fin k → ℝ) → ℝ} (hg : LipschitzOnWith L g C) :
    ∃ φ, IsCobwebTree k (m ^ k) φ ∧ ∀ u ∈ C, |φ u - g u| ≤ rateConst k β * L * w / m := by
  obtain ⟨g', hg'L, hgg'⟩ := hg.extend_real
  have hl1 : ∀ u ∈ cube a w, ∀ v ∈ cube a w, |g' u - g' v| ≤ (L : ℝ) * ∑ i, |u i - v i| := by
    intro u _ v _
    have h1 := hg'L.dist_le_mul u v
    rw [Real.dist_eq] at h1
    refine h1.trans (mul_le_mul_of_nonneg_left ?_ L.coe_nonneg)
    refine (dist_pi_le_iff (Finset.sum_nonneg fun i _ => abs_nonneg _)).mpr fun i => ?_
    rw [Real.dist_eq]
    exact Finset.single_le_sum (f := fun i => |u i - v i|) (fun i _ => abs_nonneg _)
      (Finset.mem_univ i)
  have : Nonempty (Fin m) := ⟨⟨0, hm⟩⟩
  refine ⟨gridTree m a (w / m) (β * (w / m)) g',
    ⟨Fin k → Fin m, univ, fun κ => WithLp.toLp 2 (gridPt m a (w / m) κ),
      fun κ => g' (gridPt m a (w / m) κ), fun _ => 1, β * (w / m), univ_nonempty,
      by simp, by positivity, fun _ _ => one_pos, fun _ => rfl⟩, fun u hu => ?_⟩
  rw [hgg' hu]
  exact grid_rate hm a hw hβ L.coe_nonneg g' hl1 (hC hu)

/-- Replace every vertex map by the grid tree of its McShane extension. -/
theorem exists_grid_close {K : Set (Fin n → ℝ)} {r w β : ℝ} {L : NNReal} {k m : ℕ}
    (hm : 0 < m) (hw : 0 < w) (hβ : 0 < β) :
    ∀ t : Net n, FanInLE k t → LipNear K r L t → InCubes K r w t →
      ∃ t', IsCobweb (m ^ k) t' ∧ Close K r (rateConst k β * L * w / m) t' t := by
  intro t
  induction t with
  | raw i => exact fun _ _ _ => ⟨raw i, .raw i, .raw i⟩
  | node kv args g ih =>
    intro hk hL hQ
    rcases hk with _ | ⟨hkv, hk⟩
    rcases hL with _ | ⟨hgL, hL⟩
    rcases hQ with _ | ⟨a, ha, hQ⟩
    choose args' hc hclose using fun j => ih j (hk j) (hL j) (hQ j)
    obtain ⟨φ, hφ, hφg⟩ := exists_gridTree (k := kv) hm hw hβ ha hgL
    refine ⟨node kv args' φ, .node (hφ.mono (Nat.pow_le_pow_right hm hkv)) hc,
      .node (fun u hu => (hφg u hu).trans ?_) hclose⟩
    have hLw : 0 ≤ (L : ℝ) * w / m := by positivity
    calc rateConst kv β * L * w / m = rateConst kv β * ((L : ℝ) * w / m) := by ring
      _ ≤ rateConst k β * ((L : ℝ) * w / m) :=
          mul_le_mul_of_nonneg_right (rateConst_mono hkv β) hLw
      _ = rateConst k β * L * w / m := by ring

/-- **Thm. compose (full form): composition beats dimension.**  Let `t` compute a
`k`-composition on `K` whose vertex maps are `L`-Lipschitz in the max-norm on the closed
`r`-neighbourhoods `B_v^r` of their input sets, each `B_v^r` inside a cube of side `w`.  Give
every vertex the grid tree of Prop. rate (`m^{k_v} ≤ m^k` leaves, variance floor `β w / m`) for
the McShane extension of its map, and let `η = C(k, β) L w / m`.  If `η ∑_{l<D} L^l ≤ r`, the
resulting Cobweb forest has the shape of `t` (same fan-in, depth and number of trees) and is
within `η ∑_{l<D} L^l` of `t` on `K`. -/
theorem compose_grid {K : Set (Fin n → ℝ)} {r w β : ℝ} {L : NNReal} {k m : ℕ} (hm : 0 < m)
    (hw : 0 < w) (hβ : 0 < β) {t : Net n} (hk : FanInLE k t) (hL : LipNear K r L t)
    (hQ : InCubes K r w t)
    (hmargin : rateConst k β * L * w / m * ∑ l ∈ range (depth t), (L : ℝ) ^ l ≤ r) :
    ∃ t', IsCobweb (m ^ k) t' ∧ FanInLE k t' ∧ depth t' = depth t ∧
      numTrees t' = numTrees t ∧ ∀ x ∈ K, |eval t' x - eval t x| ≤
        rateConst k β * L * w / m * ∑ l ∈ range (depth t), (L : ℝ) ^ l := by
  obtain ⟨t', ht'c, hclose⟩ := exists_grid_close hm hw hβ t hk hL hQ
  have hη : 0 ≤ rateConst k β * L * w / m := by
    have := rateConst_nonneg k β
    positivity
  exact ⟨t', ht'c, hclose.fanInLE hk, hclose.depth_eq, hclose.numTrees_eq,
    compose hη hL hclose hmargin⟩

/-! ## The class `𝓑_n` on the unit cube -/

theorem AllLip.lipNear {L : NNReal} {t : Net n} (ht : AllLip L t) (K : Set (Fin n → ℝ))
    (r : ℝ) : LipNear K r L t := by
  induction ht with
  | raw i => exact .raw i
  | node hg _ ih => exact .node hg.lipschitzOnWith ih

/-- A composition of `1`-Lipschitz maps (max-norm) is `1`-Lipschitz. -/
theorem AllLip.lipschitz_eval {t : Net n} (ht : AllLip 1 t) : LipschitzWith 1 (eval t) := by
  induction ht with
  | raw i =>
    refine LipschitzWith.of_dist_le_mul fun x y => ?_
    rw [NNReal.coe_one, one_mul]
    exact dist_le_pi_dist x y i
  | @node k args g hg _ ih =>
    have hin : LipschitzWith 1 (inVec args) := by
      refine LipschitzWith.of_dist_le_mul fun x y => ?_
      rw [NNReal.coe_one, one_mul]
      refine (dist_pi_le_iff dist_nonneg).mpr fun j => ?_
      have := (ih j).dist_le_mul x y
      rwa [NNReal.coe_one, one_mul] at this
    have := hg.comp hin
    rwa [mul_one] at this

theorem dist_le_one_of_mem_Icc {x y : Fin n → ℝ} (hx : x ∈ Set.Icc (0 : Fin n → ℝ) 1)
    (hy : y ∈ Set.Icc (0 : Fin n → ℝ) 1) : dist x y ≤ 1 := by
  refine (dist_pi_le_iff zero_le_one).mpr fun i => ?_
  rw [Real.dist_eq, abs_sub_le_iff]
  have hx0 := hx.1 i
  have hx1 := hx.2 i
  have hy0 := hy.1 i
  have hy1 := hy.2 i
  simp only [Pi.zero_apply, Pi.one_apply] at hx0 hx1 hy0 hy1
  constructor <;> linarith

/-- On `[0,1]^n`, every input set of a composition of `1`-Lipschitz maps, thickened by `r`, lies
in a cube of side `1 + 2r`. -/
theorem AllLip.inCubes {t : Net n} (ht : AllLip 1 t) {r : ℝ} (hr : 0 ≤ r) :
    InCubes (Set.Icc (0 : Fin n → ℝ) 1) r (1 + 2 * r) t := by
  induction ht with
  | raw i => exact .raw i
  | @node k args g _ hargs ih =>
    set K := Set.Icc (0 : Fin n → ℝ) 1 with hK
    have hKc : IsCompact K := isCompact_Icc
    have hKne : K.Nonempty := Set.nonempty_Icc.mpr zero_le_one
    have hcont : ∀ j, Continuous (eval (args j)) := fun j => (hargs j).lipschitz_eval.continuous
    choose xm hxmK hxmin using fun j =>
      hKc.exists_isMinOn hKne (hcont j).continuousOn
    refine .node (fun j => eval (args j) (xm j) - r) ?_ ih
    have hBc : IsCompact (inSet K args) :=
      hKc.image (continuous_pi fun j => hcont j)
    intro u hu
    rw [cthickening_eq_biUnion_closedBall _ hr, hBc.isClosed.closure_eq] at hu
    obtain ⟨b, ⟨x, hxK, rfl⟩, hub⟩ := Set.mem_iUnion₂.mp hu
    rw [mem_closedBall] at hub
    intro j
    have hj := (dist_le_pi_dist u (inVec args x) j).trans hub
    rw [Real.dist_eq, abs_le] at hj
    have hmin : eval (args j) (xm j) ≤ eval (args j) x := hxmin j hxK
    have hmax : eval (args j) x ≤ eval (args j) (xm j) + 1 := by
      have h1 := ((hargs j).lipschitz_eval).dist_le_mul x (xm j)
      rw [NNReal.coe_one, one_mul, Real.dist_eq] at h1
      have h2 := dist_le_one_of_mem_Icc hxK (hxmK j)
      have h3 := le_abs_self (eval (args j) x - eval (args j) (xm j))
      linarith
    rw [inVec_apply] at hj
    constructor <;> linarith

/-- `2^{⌈log₂ n⌉ + 2} ≤ 8n`. -/
theorem two_pow_clog_add_two_le {n : ℕ} (hn : 1 ≤ n) : 2 ^ (Nat.clog 2 n + 2) ≤ 8 * n := by
  rcases Nat.eq_or_lt_of_le hn with rfl | h2
  · simp
  · have hlt := Nat.pow_pred_clog_lt_self one_lt_two h2
    have hpos : 0 < Nat.clog 2 n := Nat.clog_pos one_lt_two h2
    have : 2 ^ Nat.clog 2 n = 2 * 2 ^ (Nat.clog 2 n).pred := by
      rw [← pow_succ']
      exact congrArg (2 ^ ·) (Nat.succ_pred_eq_of_pos hpos).symm
    rw [pow_add, this]
    omega

end Net

open Net

/-- **Thm. separation (b).**  Every `f ∈ 𝓑_n` is approximated on `[0,1]^n` by a Cobweb forest
with at most two inputs and `G = m²` leaves per tree and at most `2^D - 1 ≤ 8n - 1` trees
(`D = ⌈log₂ n⌉ + 2`), within `C D (1 + 2r) / m = C D (1 + 2r) G^{-1/2}` (`C = C(2, β)`), once
this is below the margin `r` of Thm. compose. -/
theorem separation_b {n : ℕ} (hn : 1 ≤ n) {f : (Fin n → ℝ) → ℝ} (hf : InB n f) {r β : ℝ}
    (hr : 0 ≤ r) (hβ : 0 < β) {m : ℕ} (hm : 0 < m)
    (hmargin : rateConst 2 β * (1 + 2 * r) / m * ((Nat.clog 2 n + 2 : ℕ) : ℝ) ≤ r) :
    ∃ t' : Net n, IsCobweb (m ^ 2) t' ∧ FanInLE 2 t' ∧
      numTrees t' + 1 ≤ 2 ^ (Nat.clog 2 n + 2) ∧ 2 ^ (Nat.clog 2 n + 2) ≤ 8 * n ∧
      ∀ x ∈ Set.Icc (0 : Fin n → ℝ) 1,
        |eval t' x - f x| ≤ rateConst 2 β * ((Nat.clog 2 n + 2 : ℕ) : ℝ) * (1 + 2 * r) / m := by
  obtain ⟨t, hfan, hdep, hlip, heval⟩ := hf
  have hw : (0 : ℝ) < 1 + 2 * r := by linarith
  have hC := rateConst_nonneg 2 β
  have hsum : ∑ l ∈ range (depth t), ((1 : NNReal) : ℝ) ^ l = depth t := by simp
  have hdepR : (depth t : ℝ) ≤ ((Nat.clog 2 n + 2 : ℕ) : ℝ) := by exact_mod_cast hdep
  have hη : 0 ≤ rateConst 2 β * ((1 : NNReal) : ℝ) * (1 + 2 * r) / m := by positivity
  obtain ⟨t', hc, hfan', -, hnum, herr⟩ := compose_grid (L := 1) hm hw hβ hfan
    (hlip.lipNear _ r) (hlip.inCubes hr) (by
      rw [hsum]
      calc rateConst 2 β * ((1 : NNReal) : ℝ) * (1 + 2 * r) / m * (depth t : ℝ)
          ≤ rateConst 2 β * ((1 : NNReal) : ℝ) * (1 + 2 * r) / m *
              ((Nat.clog 2 n + 2 : ℕ) : ℝ) := mul_le_mul_of_nonneg_left hdepR hη
        _ = rateConst 2 β * (1 + 2 * r) / m * ((Nat.clog 2 n + 2 : ℕ) : ℝ) := by
            rw [NNReal.coe_one, mul_one]
        _ ≤ r := hmargin)
  refine ⟨t', hc, hfan', ?_, two_pow_clog_add_two_le hn, fun x hx => ?_⟩
  · rw [hnum]
    exact (numTrees_le t hfan).trans (Nat.pow_le_pow_right (by norm_num) hdep)
  · rw [← heval x]
    refine (herr x hx).trans ?_
    rw [hsum]
    calc rateConst 2 β * ((1 : NNReal) : ℝ) * (1 + 2 * r) / m * (depth t : ℝ)
        ≤ rateConst 2 β * ((1 : NNReal) : ℝ) * (1 + 2 * r) / m *
            ((Nat.clog 2 n + 2 : ℕ) : ℝ) := mul_le_mul_of_nonneg_left hdepR hη
      _ = rateConst 2 β * ((Nat.clog 2 n + 2 : ℕ) : ℝ) * (1 + 2 * r) / m := by
          rw [NNReal.coe_one]
          ring

end CobwebForest
