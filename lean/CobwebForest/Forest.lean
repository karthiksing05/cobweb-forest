/-
Copyright (c) 2026 Cobweb Forest authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Cobweb Forest authors
-/
import CobwebForest.UnivTree
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Topology.ContinuousMap.StoneWeierstrass
import Mathlib.Topology.MetricSpace.Thickening

/-!
# Forests of two-input trees are universal approximators
(theory: Lem. stable, Thm. univ-forest)

A forest (or a composition, paper Sec. "What composition buys") over raw inputs
`x : Fin n → ℝ` is written as a syntax tree `Net n`: a source reads a raw coordinate, a vertex
with `k` inputs applies a map `g : (Fin k → ℝ) → ℝ` to the values of its `k` input sub-forests.
A DAG is the syntax tree obtained by unfolding it from its output: a shared vertex is repeated,
which changes neither the value computed nor any depth.  `Fin k → ℝ` carries the max-norm.

* `Close K r η t' t`: `t'` has the shape of `t`, and every vertex map of `t'` is within `η` of the
  map of `t` on the closed `r`-neighbourhood of that vertex's input set `B_v` (the input vectors
  of `v` as `x` ranges over `K`).
* `stable` (Lem. stable): if every vertex map of `t` is continuous on the closed
  `r`-neighbourhood of its input set, then for every `ε > 0` there is `η > 0` such that every
  `t'` with `Close K r η t' t` is within `ε` of `t` on `K` (the paper's `ρ_v = r`, `η_v = η`).
* `univ_forest` (Thm. univ-forest): for `K` compact and `f` continuous on `K`, there is a forest
  whose every vertex is a Cobweb tree (a count-weighted leaf mixture with finitely many leaves,
  `IsCobweb G`), with at most two inputs per tree (`FanInLE 2`) and a tree at the output
  (`IsNode`), within `ε` of `f` on `K`.
-/

open Metric Finset

namespace CobwebForest

/-- A forest over raw inputs `x : Fin n → ℝ`, as a syntax tree.  `raw i` is the source reading
coordinate `i`; `node k args g` is a vertex with `k` inputs, the sub-forests `args`, applying
the map `g`.  (A vertex with `k = 0` inputs is a constant.) -/
inductive Net (n : ℕ) : Type
  | raw (i : Fin n) : Net n
  | node (k : ℕ) (args : Fin k → Net n) (g : (Fin k → ℝ) → ℝ) : Net n

namespace Net

variable {n : ℕ}

/-- The value the forest computes at its output on input `x`. -/
def eval : Net n → (Fin n → ℝ) → ℝ
  | raw i, x => x i
  | node _ args g, x => g fun j => eval (args j) x

/-- The input vector of a vertex whose input sub-forests are `args`, at `x`. -/
def inVec {k : ℕ} (args : Fin k → Net n) (x : Fin n → ℝ) : Fin k → ℝ :=
  fun j => eval (args j) x

/-- The input set `B_v` of a vertex: its input vectors as `x` ranges over `K`. -/
def inSet {k : ℕ} (K : Set (Fin n → ℝ)) (args : Fin k → Net n) : Set (Fin k → ℝ) :=
  inVec args '' K

@[simp] theorem eval_raw (i : Fin n) (x : Fin n → ℝ) : eval (raw i) x = x i := rfl

@[simp] theorem eval_node {k : ℕ} (args : Fin k → Net n) (g : (Fin k → ℝ) → ℝ)
    (x : Fin n → ℝ) : eval (node k args g) x = g (inVec args x) := rfl

@[simp] theorem inVec_apply {k : ℕ} (args : Fin k → Net n) (x : Fin n → ℝ) (j : Fin k) :
    inVec args x j = eval (args j) x := rfl

theorem inVec_mem_inSet {k : ℕ} {K : Set (Fin n → ℝ)} (args : Fin k → Net n)
    {x : Fin n → ℝ} (hx : x ∈ K) : inVec args x ∈ inSet K args :=
  ⟨x, hx, rfl⟩

/-! ## Structural predicates -/

/-- Every vertex has at most `k₀` inputs. -/
inductive FanInLE (k₀ : ℕ) : Net n → Prop
  | raw (i : Fin n) : FanInLE k₀ (raw i)
  | node {k : ℕ} {args : Fin k → Net n} {g : (Fin k → ℝ) → ℝ} (hk : k ≤ k₀)
      (hargs : ∀ j, FanInLE k₀ (args j)) : FanInLE k₀ (node k args g)

/-- The output is a tree (a vertex), not a raw coordinate. -/
def IsNode : Net n → Prop
  | raw _ => False
  | node _ _ _ => True

/-- Every vertex map is continuous. -/
inductive AllCont : Net n → Prop
  | raw (i : Fin n) : AllCont (raw i)
  | node {k : ℕ} {args : Fin k → Net n} {g : (Fin k → ℝ) → ℝ} (hg : Continuous g)
      (hargs : ∀ j, AllCont (args j)) : AllCont (node k args g)

/-- Every vertex map is continuous on the closed `r`-neighbourhood of its input set. -/
inductive ContNear (K : Set (Fin n → ℝ)) (r : ℝ) : Net n → Prop
  | raw (i : Fin n) : ContNear K r (raw i)
  | node {k : ℕ} {args : Fin k → Net n} {g : (Fin k → ℝ) → ℝ}
      (hg : ContinuousOn g (cthickening r (inSet K args)))
      (hargs : ∀ j, ContNear K r (args j)) : ContNear K r (node k args g)

/-- `Close K r η t' t`: the forest `t'` has the shape of `t` (same sources, same arities), and
every vertex map of `t'` is within `η` of the corresponding map of `t` on the closed
`r`-neighbourhood (max-norm) of that vertex's input set in `t`. -/
inductive Close (K : Set (Fin n → ℝ)) (r η : ℝ) : Net n → Net n → Prop
  | raw (i : Fin n) : Close K r η (raw i) (raw i)
  | node {k : ℕ} {args' args : Fin k → Net n} {g' g : (Fin k → ℝ) → ℝ}
      (hg : ∀ u ∈ cthickening r (inSet K args), |g' u - g u| ≤ η)
      (hargs : ∀ j, Close K r η (args' j) (args j)) :
      Close K r η (node k args' g') (node k args g)

theorem AllCont.continuous_eval {t : Net n} (ht : AllCont t) : Continuous (eval t) := by
  induction ht with
  | raw i => exact continuous_apply i
  | node hg _ ih => exact hg.comp (continuous_pi ih)

theorem AllCont.contNear {t : Net n} (ht : AllCont t) (K : Set (Fin n → ℝ)) (r : ℝ) :
    ContNear K r t := by
  induction ht with
  | raw i => exact .raw i
  | node hg _ ih => exact .node hg.continuousOn ih

theorem ContNear.continuousOn_eval {K : Set (Fin n → ℝ)} {r : ℝ} {t : Net n}
    (ht : ContNear K r t) : ContinuousOn (eval t) K := by
  induction ht with
  | raw i => exact (continuous_apply i).continuousOn
  | node hg _ ih =>
    exact hg.comp (continuousOn_pi.mpr ih)
      (fun x hx => self_subset_cthickening _ (inVec_mem_inSet _ hx))

theorem isCompact_inSet {K : Set (Fin n → ℝ)} (hK : IsCompact K) {r : ℝ} {k : ℕ}
    {args : Fin k → Net n} (hargs : ∀ j, ContNear K r (args j)) :
    IsCompact (inSet K args) :=
  hK.image_of_continuousOn (continuousOn_pi.mpr fun j => (hargs j).continuousOn_eval)

theorem Close.mono {K : Set (Fin n → ℝ)} {r η η' : ℝ} {t' t : Net n} (h : Close K r η t' t)
    (hη : η ≤ η') : Close K r η' t' t := by
  induction h with
  | raw i => exact .raw i
  | node hg _ ih => exact .node (fun u hu => (hg u hu).trans hη) ih

theorem Close.fanInLE {K : Set (Fin n → ℝ)} {r η : ℝ} {t' t : Net n} (h : Close K r η t' t)
    {k₀ : ℕ} (ht : FanInLE k₀ t) : FanInLE k₀ t' := by
  induction h with
  | raw i => exact .raw i
  | node _ _ ih =>
    cases ht with
    | node hk hargs => exact .node hk fun j => ih j (hargs j)

theorem Close.isNode {K : Set (Fin n → ℝ)} {r η : ℝ} {t' t : Net n} (h : Close K r η t' t)
    (ht : IsNode t) : IsNode t' := by
  cases h with
  | raw i => exact ht
  | node _ _ => trivial

/-- A positive lower bound for finitely many positive reals. -/
theorem exists_pos_le_of_fin {k : ℕ} (f : Fin k → ℝ) (hf : ∀ j, 0 < f j) :
    ∃ c > 0, ∀ j, c ≤ f j := by
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · exact ⟨1, one_pos, fun j => j.elim0⟩
  · have : Nonempty (Fin k) := ⟨⟨0, hk⟩⟩
    obtain ⟨j₀, hj₀⟩ := Finite.exists_min f
    exact ⟨f j₀, hf j₀, hj₀⟩

/-! ## Stability of composition -/

theorem stable_aux {K : Set (Fin n → ℝ)} (hK : IsCompact K) {r : ℝ} (hr : 0 < r) :
    ∀ t : Net n, ContNear K r t → ∀ ε > 0,
      ∃ η > 0, ∀ t' : Net n, Close K r η t' t → ∀ x ∈ K, |eval t' x - eval t x| < ε := by
  intro t
  induction t with
  | raw i =>
    intro _ ε hε
    refine ⟨1, one_pos, fun t' ht' x _ => ?_⟩
    cases ht'
    simpa using hε
  | node k args g ih =>
    intro ht ε hε
    rcases ht with _ | ⟨hg, hargs⟩
    -- uniform continuity of `g` on the compact neighbourhood of its input set
    have hCc : IsCompact (cthickening r (inSet K args)) :=
      (isCompact_inSet hK hargs).cthickening
    obtain ⟨ρ₀, hρ₀, hρ₀g⟩ := Metric.uniformContinuousOn_iff.mp
      (hCc.uniformContinuousOn_of_continuous hg) (ε / 2) (by positivity)
    -- tolerances for the inputs, then a common one
    have hδ : 0 < min ρ₀ r := lt_min hρ₀ hr
    choose ηs hηs hηt using fun j => ih j (hargs j) (min ρ₀ r) hδ
    obtain ⟨c, hc, hcη⟩ := exists_pos_le_of_fin ηs hηs
    refine ⟨min (ε / 2) c, lt_min (by positivity) hc, fun t' ht' x hx => ?_⟩
    rcases ht' with _ | @⟨_, args', _, g', _, hg', hargs'⟩
    -- the inputs deviate by less than `min ρ₀ r`
    have hdev : ∀ j, dist (inVec args' x j) (inVec args x j) < min ρ₀ r := fun j => by
      rw [inVec_apply, inVec_apply, Real.dist_eq]
      exact hηt j (args' j) ((hargs' j).mono ((min_le_right _ _).trans (hcη j))) x hx
    have hdist : dist (inVec args' x) (inVec args x) < min ρ₀ r := (dist_pi_lt_iff hδ).mpr hdev
    have hu : inVec args x ∈ cthickening r (inSet K args) :=
      self_subset_cthickening _ (inVec_mem_inSet args hx)
    have hu' : inVec args' x ∈ cthickening r (inSet K args) :=
      mem_cthickening_of_dist_le _ _ _ _ (inVec_mem_inSet args hx)
        (hdist.le.trans (min_le_right _ _))
    have h1 := (hg' _ hu').trans (min_le_left _ _)
    have h2 := hρ₀g _ hu' _ hu (hdist.trans_le (min_le_left _ _))
    rw [Real.dist_eq] at h2
    rw [eval_node, eval_node]
    calc |g' (inVec args' x) - g (inVec args x)|
        ≤ |g' (inVec args' x) - g (inVec args' x)| +
            |g (inVec args' x) - g (inVec args x)| := abs_sub_le _ _ _
      _ < ε / 2 + ε / 2 := add_lt_add_of_le_of_lt h1 h2
      _ = ε := add_halves ε

/-- **Lem. stable (stability of composition).**  Let `K` be compact and let every vertex map of
the forest `t` be continuous on the closed `r`-neighbourhood (`r > 0`) of its input set.  For
every `ε > 0` there is `η > 0` such that replacing every vertex map by any map that is within
`η` of it on the closed `r`-neighbourhood of the vertex's input set changes the output by less
than `ε` on `K`. -/
theorem stable {K : Set (Fin n → ℝ)} (hK : IsCompact K) {r : ℝ} (hr : 0 < r) {t : Net n}
    (ht : ContNear K r t) {ε : ℝ} (hε : 0 < ε) :
    ∃ η > 0, ∀ t' : Net n, Close K r η t' t → ∀ x ∈ K, |eval t' x - eval t x| < ε :=
  stable_aux hK hr t ht ε hε

/-- Every vertex map is continuous on *some* closed neighbourhood of its input set (each vertex
with its own radius), as in the hypothesis of Lem. stable. -/
inductive ContNearSome (K : Set (Fin n → ℝ)) : Net n → Prop
  | raw (i : Fin n) : ContNearSome K (raw i)
  | node {k : ℕ} {args : Fin k → Net n} {g : (Fin k → ℝ) → ℝ}
      (hg : ∃ r > 0, ContinuousOn g (cthickening r (inSet K args)))
      (hargs : ∀ j, ContNearSome K (args j)) : ContNearSome K (node k args g)

theorem ContNear.mono {K : Set (Fin n → ℝ)} {r r' : ℝ} {t : Net n} (ht : ContNear K r t)
    (hr : r' ≤ r) : ContNear K r' t := by
  induction ht with
  | raw i => exact .raw i
  | node hg _ ih => exact .node (hg.mono (cthickening_mono hr _)) ih

/-- Finitely many vertices: a common radius. -/
theorem ContNearSome.exists_uniform {K : Set (Fin n → ℝ)} :
    ∀ t : Net n, ContNearSome K t → ∃ r > 0, ContNear K r t := by
  intro t
  induction t with
  | raw i => exact fun _ => ⟨1, one_pos, .raw i⟩
  | node k args g ih =>
    intro ht
    rcases ht with _ | ⟨⟨rv, hrv, hg⟩, hargs⟩
    choose rs hrs hcont using fun j => ih j (hargs j)
    obtain ⟨c, hc, hcr⟩ := exists_pos_le_of_fin rs hrs
    refine ⟨min rv c, lt_min hrv hc, .node (hg.mono (cthickening_mono (min_le_left _ _) _))
      fun j => (hcont j).mono ((min_le_right _ _).trans (hcr j))⟩

/-- **Lem. stable, with a radius for each vertex.**  If every vertex map is continuous on some
closed neighbourhood of its input set, there is a radius `ρ > 0` such that for every `ε > 0`
some tolerance `η > 0` works: maps within `η` on the closed `ρ`-neighbourhoods of the input
sets change the output by less than `ε` on `K`. -/
theorem stable_of_margins {K : Set (Fin n → ℝ)} (hK : IsCompact K) {t : Net n}
    (ht : ContNearSome K t) : ∃ ρ > 0, ∀ ε > 0,
      ∃ η > 0, ∀ t' : Net n, Close K ρ η t' t → ∀ x ∈ K, |eval t' x - eval t x| < ε := by
  obtain ⟨ρ, hρ, hcont⟩ := ht.exists_uniform t
  exact ⟨ρ, hρ, fun ε hε => stable hK hρ hcont hε⟩

/-! ## Cobweb forests -/

/-- A vertex map is a **Cobweb tree** with `k` inputs and at most `G` leaves: the count-weighted
leaf mixture (paper, eq. (1)) of a depth-one tree whose leaves `s ∈ S` hold `c s > 0` copies of
the example `(γ s, y s)`, with `γ s ∈ ℝ^k` (Euclidean distance), `S` nonempty with at most `G`
elements, and variance floor `h > 0`. -/
def IsCobwebTree (k G : ℕ) (φ : (Fin k → ℝ) → ℝ) : Prop :=
  ∃ (ι : Type) (S : Finset ι) (γ : ι → EuclideanSpace ℝ (Fin k)) (y c : ι → ℝ) (h : ℝ),
    S.Nonempty ∧ S.card ≤ G ∧ 0 < h ∧ (∀ s ∈ S, 0 < c s) ∧
      ∀ u, φ u = leafMixture S γ y c h (WithLp.toLp 2 u)

theorem IsCobwebTree.mono {k G G' : ℕ} {φ : (Fin k → ℝ) → ℝ} (h : IsCobwebTree k G φ)
    (hG : G ≤ G') : IsCobwebTree k G' φ := by
  obtain ⟨ι, S, γ, y, c, h, hS, hSG, hh, hc, hφ⟩ := h
  exact ⟨ι, S, γ, y, c, h, hS, hSG.trans hG, hh, hc, hφ⟩

/-- A **Cobweb forest** whose trees have at most `G` leaves each: every vertex is a Cobweb tree
that passes its activation. -/
inductive IsCobweb (G : ℕ) : Net n → Prop
  | raw (i : Fin n) : IsCobweb G (raw i)
  | node {k : ℕ} {args : Fin k → Net n} {g : (Fin k → ℝ) → ℝ} (hg : IsCobwebTree k G g)
      (hargs : ∀ j, IsCobweb G (args j)) : IsCobweb G (node k args g)

theorem IsCobweb.mono {G G' : ℕ} {t : Net n} (h : IsCobweb G t) (hG : G ≤ G') :
    IsCobweb G' t := by
  induction h with
  | raw i => exact .raw i
  | node hg _ ih => exact .node (hg.mono hG) ih

/-- Thm. univ-tree applied to a vertex map: on a compact set of inputs, a Cobweb tree is within
`η` of any continuous map. -/
theorem exists_cobwebTree {k : ℕ} {C : Set (Fin k → ℝ)} (hC : IsCompact C)
    {g : (Fin k → ℝ) → ℝ} (hg : ContinuousOn g C) {η : ℝ} (hη : 0 < η) :
    ∃ φ G, IsCobwebTree k G φ ∧ ∀ u ∈ C, |φ u - g u| < η := by
  rcases C.eq_empty_or_nonempty with rfl | hne
  · refine ⟨fun u => leafMixture {0} id (fun _ => 0) (fun _ => 1) 1 (WithLp.toLp 2 u), 1,
      ⟨EuclideanSpace ℝ (Fin k), {0}, id, fun _ => 0, fun _ => 1, 1,
        ⟨0, Finset.mem_singleton_self _⟩, by simp, one_pos, fun _ _ => one_pos, fun _ => rfl⟩,
      fun u hu => hu.elim⟩
  · set e : (Fin k → ℝ) → EuclideanSpace ℝ (Fin k) := WithLp.toLp 2 with he_def
    have he : Continuous e := PiLp.continuous_toLp 2 _
    have hC' : IsCompact (e '' C) := hC.image he
    set g' : EuclideanSpace ℝ (Fin k) → ℝ := fun v => g (WithLp.ofLp v) with hg'_def
    have hg' : ContinuousOn g' (e '' C) := by
      refine hg.comp (PiLp.continuous_ofLp 2 _).continuousOn ?_
      rintro _ ⟨u, hu, rfl⟩
      exact hu
    obtain ⟨S, -, hSne, h, hh, hS⟩ := univ_tree_counts hC' hg' hη one_pos le_rfl
    refine ⟨fun u => leafMixture S id g' (fun _ => 1) h (e u), S.card,
      ⟨EuclideanSpace ℝ (Fin k), S, id, g', fun _ => 1, h, hSne (hne.image e), le_rfl, hh,
        fun _ _ => one_pos, fun _ => rfl⟩, fun u hu => ?_⟩
    exact hS (fun _ => 1) (fun _ _ => ⟨le_rfl, le_rfl⟩) (e u) ⟨u, hu, rfl⟩

/-- Every forest of continuous vertex maps is `Close` to a Cobweb forest of the same shape. -/
theorem exists_cobweb_close {K : Set (Fin n → ℝ)} (hK : IsCompact K) {r η : ℝ} (hη : 0 < η) :
    ∀ t : Net n, ContNear K r t → ∃ t' G, IsCobweb G t' ∧ Close K r η t' t := by
  intro t
  induction t with
  | raw i => exact fun _ => ⟨raw i, 0, .raw i, .raw i⟩
  | node k args g ih =>
    intro ht
    rcases ht with _ | ⟨hg, hargs⟩
    choose args' Gs hargs'c hargs'close using fun j => ih j (hargs j)
    have hCc : IsCompact (cthickening r (inSet K args)) :=
      (isCompact_inSet hK hargs).cthickening
    obtain ⟨φ, Gφ, hφ, hφg⟩ := exists_cobwebTree hCc hg hη
    refine ⟨node k args' φ, Gφ + ∑ j, Gs j,
      .node (hφ.mono (Nat.le_add_right _ _)) fun j => (hargs'c j).mono ?_,
      .node (fun u hu => (hφg u hu).le) hargs'close⟩
    exact (Finset.single_le_sum (fun j _ => Nat.zero_le (Gs j)) (Finset.mem_univ j)).trans
      (Nat.le_add_left _ _)

/-! ## Polynomial forests and Stone–Weierstrass -/

/-- `(a, b) ↦ a + b`. -/
def addMap : (Fin 2 → ℝ) → ℝ := fun u => u 0 + u 1
/-- `(a, b) ↦ a b`. -/
def mulMap : (Fin 2 → ℝ) → ℝ := fun u => u 0 * u 1
/-- `a ↦ a`. -/
def idMap : (Fin 1 → ℝ) → ℝ := fun u => u 0

theorem continuous_addMap : Continuous addMap := (continuous_apply 0).add (continuous_apply 1)
theorem continuous_mulMap : Continuous mulMap := (continuous_apply 0).mul (continuous_apply 1)
theorem continuous_idMap : Continuous idMap := continuous_apply 0

/-- The functions on `K` computed by forests of continuous vertex maps with at most two inputs
form a subalgebra of `C(K, ℝ)` (it contains the coordinates and is closed under `+`, `×` and
constants). -/
def polyAlg (K : Set (Fin n → ℝ)) : Subalgebra ℝ C(K, ℝ) where
  carrier := {f | ∃ t : Net n, AllCont t ∧ FanInLE 2 t ∧ ∀ x : K, f x = eval t x}
  mul_mem' := by
    rintro f g ⟨a, ha, ha2, hfa⟩ ⟨b, hb, hb2, hgb⟩
    refine ⟨node 2 ![a, b] mulMap, .node continuous_mulMap (Fin.forall_fin_two.mpr ⟨ha, hb⟩),
      .node le_rfl (Fin.forall_fin_two.mpr ⟨ha2, hb2⟩), fun x => ?_⟩
    simp [hfa x, hgb x, mulMap]
  add_mem' := by
    rintro f g ⟨a, ha, ha2, hfa⟩ ⟨b, hb, hb2, hgb⟩
    refine ⟨node 2 ![a, b] addMap, .node continuous_addMap (Fin.forall_fin_two.mpr ⟨ha, hb⟩),
      .node le_rfl (Fin.forall_fin_two.mpr ⟨ha2, hb2⟩), fun x => ?_⟩
    simp [hfa x, hgb x, addMap]
  algebraMap_mem' c := by
    refine ⟨node 0 ![] fun _ => c, .node continuous_const fun j => j.elim0,
      .node (by norm_num) fun j => j.elim0, fun x => ?_⟩
    simp

theorem polyAlg_separatesPoints (K : Set (Fin n → ℝ)) : (polyAlg K).SeparatesPoints := by
  intro x y hxy
  have hne : (x : Fin n → ℝ) ≠ y := fun h => hxy (Subtype.ext h)
  obtain ⟨i, hi⟩ := Function.ne_iff.mp hne
  let F : C(K, ℝ) := ⟨fun z => (z : Fin n → ℝ) i, (continuous_apply i).comp continuous_subtype_val⟩
  exact ⟨F, ⟨F, ⟨raw i, .raw i, .raw i, fun _ => rfl⟩, rfl⟩, hi⟩

/-- **Stone–Weierstrass for forests.**  Every continuous `f` on a compact `K` is uniformly
approximated by a forest of `+`, `×`, constants and coordinates, with at most two inputs per
vertex. -/
theorem exists_poly_net {K : Set (Fin n → ℝ)} (hK : IsCompact K) {f : (Fin n → ℝ) → ℝ}
    (hf : ContinuousOn f K) {ε : ℝ} (hε : 0 < ε) :
    ∃ t : Net n, AllCont t ∧ FanInLE 2 t ∧ ∀ x ∈ K, |eval t x - f x| < ε := by
  have : CompactSpace K := isCompact_iff_compactSpace.mp hK
  obtain ⟨⟨F, hF⟩, hFf⟩ :=
    ContinuousMap.exists_mem_subalgebra_near_continuous_of_separatesPoints
      (polyAlg K) (polyAlg_separatesPoints K) (K.domRestrict f) hf.domRestrict ε hε
  obtain ⟨t, ht, ht2, hFt⟩ := hF
  refine ⟨t, ht, ht2, fun x hx => ?_⟩
  have h := hFf ⟨x, hx⟩
  rw [Real.norm_eq_abs] at h
  rw [← hFt ⟨x, hx⟩]
  exact h

/-- **Thm. univ-forest: forests of two-input trees are universal approximators.**  For `K`
compact, `f` continuous on `K` and `ε > 0`, there is a Cobweb forest (every vertex is a Cobweb
tree passing its activation) in which every tree has at most two inputs and the output is a
tree, whose output is within `ε` of `f` on `K`. -/
theorem univ_forest {K : Set (Fin n → ℝ)} (hK : IsCompact K) {f : (Fin n → ℝ) → ℝ}
    (hf : ContinuousOn f K) {ε : ℝ} (hε : 0 < ε) :
    ∃ t : Net n, (∃ G, IsCobweb G t) ∧ FanInLE 2 t ∧ IsNode t ∧
      ∀ x ∈ K, |eval t x - f x| < ε := by
  -- an exact forest within `ε/2` (Stone–Weierstrass) ...
  obtain ⟨p, hp, hp2, hpf⟩ := exists_poly_net hK hf (half_pos hε)
  -- ... with one identity vertex on top, so that the output is a tree
  have hp₁ : AllCont (node 1 ![p] idMap) := .node continuous_idMap (Fin.forall_fin_one.mpr hp)
  have hp₁2 : FanInLE 2 (node 1 ![p] idMap) := .node (by norm_num) (Fin.forall_fin_one.mpr hp2)
  -- stability with a tolerance `η`, then a Cobweb tree within `η` at every vertex
  obtain ⟨η, hη, hstab⟩ := stable hK one_pos (hp₁.contNear K 1) (half_pos hε)
  obtain ⟨t, G, htc, htclose⟩ := exists_cobweb_close hK hη _ (hp₁.contNear K 1)
  refine ⟨t, ⟨G, htc⟩, htclose.fanInLE hp₁2, htclose.isNode trivial, fun x hx => ?_⟩
  have h1 := hstab t htclose x hx
  have h2 := hpf x hx
  have hp₁x : eval (node 1 ![p] idMap) x = eval p x := rfl
  rw [hp₁x] at h1
  calc |eval t x - f x| ≤ |eval t x - eval p x| + |eval p x - f x| := abs_sub_le _ _ _
    _ < ε / 2 + ε / 2 := add_lt_add h1 h2
    _ = ε := add_halves ε

end Net

end CobwebForest
