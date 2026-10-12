/-
Copyright (c) 2026 Cobweb Forest authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Cobweb Forest authors
-/
import CobwebForest.Tasks

/-!
# Funnel tasks are sufficient for monotone-additive targets
(theory: Prop. suffclass, Prop. tasks (a))

Finite form.  The raw input consists of `m` independent blocks `X_j` with values in a finite set
`β` (product distribution `Dist.pi q`), and `Y = Φ(∑_j g_j(X_j)) + ξ` with `Φ` strictly
increasing and `ξ` centred and independent of `X`.  A funnel forest whose graph is a rooted tree
is a `BTree`: each tree reads the raw blocks `raw` and the messages of its child trees; every
tree's outcome is `Y`, so its ideal message is `msg τ = E[Y | its ideal input]`.

* `suffclass_alg` (the algebraic induction): if every `E[Y | X_S] = H_S(s_S)`, with
  `s_S = ∑_{j ∈ S} g_j(X_j)` and `H_S` strictly increasing, then every tree of a forest in which
  each block is the raw input of exactly one tree (`BTree.Disj`) has ideal message
  `E[Y | X_{R(τ)}]`, `R(τ)` its upstream blocks.  The proof is the paper's: `s_{R(τ)}` is
  `∑_{raw} g_j + ∑_children H^{-1}(message)`, a function of the tree's ideal input.
* `condExp_model` (independence): in the product model `E[Y | X_S] = H_S(s_S)` with
  `H_S(s) = E[Φ(s + ∑_{j ∉ S} g_j(X_j))]`, strictly increasing.  The independence step is
  `sum_fib_indep`, proved by exchanging the unseen blocks of two samples.
* `suffclass` (Prop. suffclass): every such forest reading all blocks is sufficient: its ideal
  output is `E[Y | X]`.
-/

open Finset

namespace CobwebForest

section Fibres

variable {Ω : Type*} [Fintype Ω] (P : Dist Ω)

theorem condExp_eq_of_cue_eq {α : Type*} [DecidableEq α] (V : Ω → ℝ) (A : Ω → α) {ω₁ ω₂ : Ω}
    (h : A ω₁ = A ω₂) : condExp P V A ω₁ = condExp P V A ω₂ := by
  unfold condExp fibreMass
  rw [h]

theorem condExp_add {α : Type*} [DecidableEq α] (V W : Ω → ℝ) (A : Ω → α) (ω : Ω) :
    condExp P (fun ω => V ω + W ω) A ω = condExp P V A ω + condExp P W A ω := by
  unfold condExp
  rw [← add_div, ← Finset.sum_add_distrib]
  congr 1
  exact Finset.sum_congr rfl fun ω' _ => by ring

/-- Tower property when the cue `B` is constant on the fibres of `A`. -/
theorem condExp_tower_of_const {α β : Type*} [DecidableEq α] [DecidableEq β] (V : Ω → ℝ)
    (A : Ω → α) (B : Ω → β) (hB : ∀ ω₁ ω₂, A ω₁ = A ω₂ → B ω₁ = B ω₂) (ω : Ω) :
    condExp P (condExp P V A) B ω = condExp P V B ω := by
  have h := sum_mul_condExp_of_saturated P V A {ω' ∈ (univ : Finset Ω) | B ω' = B ω}
    (fun ω₁ ω₂ h12 => by simp [hB ω₁ ω₂ h12])
  show (∑ ω' ∈ {ω' ∈ (univ : Finset Ω) | B ω' = B ω}, P.p ω' * condExp P V A ω') /
      fibreMass P B ω = (∑ ω' ∈ {ω' ∈ (univ : Finset Ω) | B ω' = B ω}, P.p ω' * V ω') /
      fibreMass P B ω
  rw [h]

/-- Sufficiency criterion for a cue `B` constant on the fibres of `A` that determines
`E[Y | A]`: then `E[Y | B] = E[Y | A]` on the support. -/
theorem condExp_eq_of_const {α β : Type*} [DecidableEq α] [DecidableEq β] (Y : Ω → ℝ)
    (A : Ω → α) (B : Ω → β) (φ : β → ℝ) (hB : ∀ ω₁ ω₂, A ω₁ = A ω₂ → B ω₁ = B ω₂)
    (hf : ∀ ω, 0 < P.p ω → condExp P Y A ω = φ (B ω)) {ω : Ω} (hω : 0 < P.p ω) :
    condExp P Y B ω = condExp P Y A ω := by
  rw [← condExp_tower_of_const P Y A B hB, condExp_congr P hf,
    condExp_of_fn P _ B φ (fun _ => rfl) hω, hf ω hω]

end Fibres

/-! ## Blocks and block trees -/

variable {m : ℕ} {β : Type*}

/-- The cue `X_S`: the blocks in `S`, the other blocks hidden. -/
def maskOpt (S : Finset (Fin m)) (x : Fin m → β) : Fin m → Option β :=
  fun j => if j ∈ S then some (x j) else none

theorem maskOpt_eq_iff {S : Finset (Fin m)} {x y : Fin m → β} :
    maskOpt S x = maskOpt S y ↔ ∀ j ∈ S, x j = y j := by
  constructor
  · intro h j hj
    have := congrFun h j
    simpa [maskOpt, hj] using this
  · intro h
    funext j
    by_cases hj : j ∈ S
    · simp [maskOpt, hj, h j hj]
    · simp [maskOpt, hj]

theorem maskOpt_mono {S T : Finset (Fin m)} (hST : S ⊆ T) {x y : Fin m → β}
    (h : maskOpt T x = maskOpt T y) : maskOpt S x = maskOpt S y :=
  maskOpt_eq_iff.mpr fun j hj => maskOpt_eq_iff.mp h j (hST hj)

/-- A funnel forest whose graph is a rooted tree: a tree reads the raw blocks `raw` and the
messages of its `k` child trees. -/
inductive BTree (m : ℕ) : Type
  | node (raw : Finset (Fin m)) (k : ℕ) (kids : Fin k → BTree m) : BTree m

namespace BTree

/-- The blocks upstream of a tree, `R(τ)`. -/
def up : BTree m → Finset (Fin m)
  | node raw _ kids => raw ∪ Finset.univ.biUnion fun i => up (kids i)

theorem up_node (raw : Finset (Fin m)) (k : ℕ) (kids : Fin k → BTree m) :
    up (node raw k kids) = raw ∪ Finset.univ.biUnion fun i => up (kids i) := rfl

/-- Each block is the raw input of exactly one tree. -/
inductive Disj : BTree m → Prop
  | node {raw : Finset (Fin m)} {k : ℕ} {kids : Fin k → BTree m} (hkids : ∀ i, Disj (kids i))
      (hraw : ∀ i, Disjoint raw (up (kids i)))
      (hpair : ∀ i i', i ≠ i' → Disjoint (up (kids i)) (up (kids i'))) :
      Disj (node raw k kids)

end BTree

section Messages

variable {Ω : Type*} [Fintype Ω] (P : Dist Ω) [DecidableEq β] (Y : Ω → ℝ) (X : Ω → Fin m → β)

/-- The ideal message of a funnel tree: `E[Y | U]`, `U` its raw blocks and the ideal messages of
its children. -/
noncomputable def msg : BTree m → Ω → ℝ
  | .node raw _ kids => condExp P Y fun ω => (maskOpt raw (X ω), fun i => msg (kids i) ω)

theorem msg_node (raw : Finset (Fin m)) (k : ℕ) (kids : Fin k → BTree m) :
    msg P Y X (.node raw k kids) =
      condExp P Y fun ω => (maskOpt raw (X ω), fun i => msg P Y X (kids i) ω) := rfl

/-- A tree's ideal message is a function of its upstream blocks. -/
theorem msg_const : ∀ τ : BTree m, ∀ ω₁ ω₂,
    maskOpt τ.up (X ω₁) = maskOpt τ.up (X ω₂) → msg P Y X τ ω₁ = msg P Y X τ ω₂ := by
  intro τ
  induction τ with
  | node raw k kids ih =>
    intro ω₁ ω₂ h
    rw [msg_node]
    apply condExp_eq_of_cue_eq
    refine Prod.ext (maskOpt_mono Finset.subset_union_left h) (funext fun i => ?_)
    refine ih i ω₁ ω₂ (maskOpt_mono ?_ h)
    exact (Finset.subset_biUnion_of_mem (fun i => BTree.up (kids i)) (Finset.mem_univ i)).trans
      Finset.subset_union_right

/-- **The algebraic induction of Prop. suffclass.**  If `E[Y | X_S] = H_S(s_S)` with `H_S`
strictly increasing for every set `S` of blocks, then in a funnel forest in which each block is
the raw input of exactly one tree, every tree's ideal message is `E[Y | X_{R(τ)}]`. -/
theorem suffclass_alg (g : Fin m → β → ℝ)
    (hH : ∀ S : Finset (Fin m), ∃ H : ℝ → ℝ, StrictMono H ∧ ∀ ω, 0 < P.p ω →
      condExp P Y (fun ω => maskOpt S (X ω)) ω = H (∑ j ∈ S, g j (X ω j))) :
    ∀ τ : BTree m, τ.Disj → ∀ ω, 0 < P.p ω →
      msg P Y X τ ω = condExp P Y (fun ω => maskOpt τ.up (X ω)) ω := by
  intro τ
  induction τ with
  | node raw k kids ih =>
    intro hD ω hω
    rcases hD with ⟨hkidsD, hraw, hpair⟩
    choose Hk hHk_mono hHk using fun i => hH (BTree.up (kids i))
    obtain ⟨HS, -, hHS⟩ := hH (BTree.up (.node raw k kids))
    -- the decomposition `s_{R(τ)} = s_raw + ∑_children s_{R(child)}`
    have hdisj : Disjoint raw (Finset.univ.biUnion fun i => BTree.up (kids i)) :=
      (Finset.disjoint_biUnion_right _ _ _).mpr fun i _ => hraw i
    have hpw : (↑(Finset.univ : Finset (Fin k)) : Set (Fin k)).PairwiseDisjoint
        fun i => BTree.up (kids i) := fun i _ i' _ hii' => hpair i i' hii'
    have hsplit : ∀ ω', ∑ j ∈ BTree.up (.node raw k kids), g j (X ω' j) =
        ∑ j ∈ raw, g j (X ω' j) + ∑ i, ∑ j ∈ BTree.up (kids i), g j (X ω' j) := fun ω' => by
      rw [BTree.up_node, Finset.sum_union hdisj, Finset.sum_biUnion hpw]
    -- the children's messages determine their upstream sums
    have hinv : ∀ i ω', 0 < P.p ω' → Function.invFun (Hk i) (msg P Y X (kids i) ω') =
        ∑ j ∈ BTree.up (kids i), g j (X ω' j) := fun i ω' hω' => by
      rw [ih i (hkidsD i) ω' hω', hHk i ω' hω']
      exact Function.leftInverse_invFun (hHk_mono i).injective _
    -- `E[Y | X_{R(τ)}]` is a function of the ideal input
    set φ : (Fin m → Option β) × (Fin k → ℝ) → ℝ := fun u =>
      HS (∑ j ∈ raw, (u.1 j).elim 0 (g j) + ∑ i, Function.invFun (Hk i) (u.2 i)) with hφ
    have hf : ∀ ω', 0 < P.p ω' →
        condExp P Y (fun ω => maskOpt (BTree.up (.node raw k kids)) (X ω)) ω' =
          φ (maskOpt raw (X ω'), fun i => msg P Y X (kids i) ω') := by
      intro ω' hω'
      rw [hHS ω' hω', hsplit ω', hφ]
      congr 1
      congr 1
      · refine Finset.sum_congr rfl fun j hj => ?_
        simp [maskOpt, hj]
      · exact Finset.sum_congr rfl fun i _ => (hinv i ω' hω').symm
    -- the ideal input is a function of the upstream blocks
    have hB : ∀ ω₁ ω₂, maskOpt (BTree.up (.node raw k kids)) (X ω₁) =
        maskOpt (BTree.up (.node raw k kids)) (X ω₂) →
        (maskOpt raw (X ω₁), fun i => msg P Y X (kids i) ω₁) =
          (maskOpt raw (X ω₂), fun i => msg P Y X (kids i) ω₂) := by
      intro ω₁ ω₂ h
      refine Prod.ext (maskOpt_mono Finset.subset_union_left h) (funext fun i => ?_)
      refine msg_const P Y X (kids i) ω₁ ω₂ (maskOpt_mono ?_ h)
      exact (Finset.subset_biUnion_of_mem (fun i => BTree.up (kids i)) (Finset.mem_univ i)).trans
        Finset.subset_union_right
    rw [msg_node]
    exact condExp_eq_of_const P Y _ _ φ hB hf hω

/-- A forest reading every block conditions on all of `X`. -/
theorem condExp_maskOpt_univ (ω : Ω) :
    condExp P Y (fun ω => maskOpt (Finset.univ : Finset (Fin m)) (X ω)) ω = condExp P Y X ω := by
  have hfib : ∀ ω', maskOpt (Finset.univ : Finset (Fin m)) (X ω') =
      maskOpt (Finset.univ : Finset (Fin m)) (X ω) ↔ X ω' = X ω := fun ω' => by
    rw [maskOpt_eq_iff]
    simp only [Finset.mem_univ, true_implies]
    exact funext_iff.symm
  unfold condExp fibreMass
  simp_rw [hfib]

end Messages

/-! ## Independent blocks -/

section Model

variable [Fintype β] [DecidableEq β]

/-- The product of distributions on the blocks: independent blocks. -/
noncomputable def Dist.pi (q : Fin m → Dist β) : Dist (Fin m → β) where
  p x := ∏ j, (q j).p (x j)
  nonneg x := Finset.prod_nonneg fun j _ => (q j).nonneg _
  sum_one := by
    rw [← Fintype.piFinset_univ, ← Finset.prod_univ_sum]
    simp [(q _).sum_one]

/-- Exchange the blocks outside `S`. -/
def mrg (S : Finset (Fin m)) (a b : Fin m → β) : Fin m → β := fun j => if j ∈ S then a j else b j

/-- **Independence of blocks.**  On the fibre of `X_S` through `x`, a function `F` of the
blocks outside `S` has the same average as on the whole space. -/
theorem sum_fib_indep (q : Fin m → Dist β) (S : Finset (Fin m)) (x : Fin m → β)
    (F : (Fin m → β) → ℝ) (hF : ∀ y y', (∀ j ∉ S, y j = y' j) → F y = F y') :
    ∑ x' ∈ univ.filter (fun x' => maskOpt S x' = maskOpt S x), (Dist.pi q).p x' * F x' =
      (∑ x' ∈ univ.filter (fun x' => maskOpt S x' = maskOpt S x), (Dist.pi q).p x') *
        ∑ x'', (Dist.pi q).p x'' * F x'' := by
  set fib := univ.filter (fun x' : Fin m → β => maskOpt S x' = maskOpt S x) with hfib
  set p := (Dist.pi q).p with hp
  have hmem : ∀ y, y ∈ fib ↔ ∀ j ∈ S, y j = x j := fun y => by
    rw [hfib, Finset.mem_filter, maskOpt_eq_iff]
    simp
  -- the swap of the unseen blocks preserves the product weights
  have hw : ∀ a ∈ fib, ∀ b : Fin m → β, p a * p b = p (mrg S x b) * p (mrg S b a) := by
    intro a ha b
    rw [hp]
    simp only [Dist.pi]
    rw [← Finset.prod_mul_distrib, ← Finset.prod_mul_distrib]
    refine Finset.prod_congr rfl fun j _ => ?_
    by_cases hj : j ∈ S
    · simp [mrg, hj, (hmem a).mp ha j hj]
    · simp [mrg, hj, mul_comm]
  have hFm : ∀ b, F (mrg S x b) = F b := fun b =>
    hF _ _ fun j hj => by simp [mrg, hj]
  -- both sides are sums over `fib × univ`, exchanged by the swap
  have hL : ∑ x' ∈ fib, p x' * F x' = ∑ ab ∈ fib ×ˢ (univ : Finset (Fin m → β)),
      p ab.1 * p ab.2 * F ab.1 := by
    rw [Finset.sum_product]
    refine Finset.sum_congr rfl fun a _ => ?_
    dsimp only
    have h1 : ∑ y, p a * p y * F a = p a * F a * ∑ y, p y := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun y _ => by ring
    rw [h1, hp, (Dist.pi q).sum_one, mul_one]
  have hR : (∑ x' ∈ fib, p x') * ∑ x'', p x'' * F x'' =
      ∑ ab ∈ fib ×ˢ (univ : Finset (Fin m → β)), p ab.1 * p ab.2 * F ab.2 := by
    rw [Finset.sum_product, Finset.sum_mul]
    refine Finset.sum_congr rfl fun a _ => ?_
    rw [Finset.mul_sum]
    dsimp only
    exact Finset.sum_congr rfl fun b _ => by ring
  rw [hL, hR]
  symm
  refine Finset.sum_nbij' (fun ab => (mrg S x ab.2, mrg S ab.2 ab.1))
    (fun ab => (mrg S x ab.2, mrg S ab.2 ab.1)) ?_ ?_ ?_ ?_ ?_
  · intro ab _
    simp only [Finset.mem_product, Finset.mem_univ, and_true]
    exact (hmem _).mpr fun j hj => by simp [mrg, hj]
  · intro ab _
    simp only [Finset.mem_product, Finset.mem_univ, and_true]
    exact (hmem _).mpr fun j hj => by simp [mrg, hj]
  · intro ab hab
    have ha := (hmem ab.1).mp (Finset.mem_product.mp hab).1
    ext j
    · by_cases hj : j ∈ S
      · simp [mrg, hj, ha j hj]
      · simp [mrg, hj]
    · by_cases hj : j ∈ S
      · simp [mrg, hj]
      · simp [mrg, hj]
  · intro ab hab
    have ha := (hmem ab.1).mp (Finset.mem_product.mp hab).1
    ext j
    · by_cases hj : j ∈ S
      · simp [mrg, hj, ha j hj]
      · simp [mrg, hj]
    · by_cases hj : j ∈ S
      · simp [mrg, hj]
      · simp [mrg, hj]
  · intro ab hab
    rw [hw ab.1 (Finset.mem_product.mp hab).1 ab.2, hFm]

/-- `H_S(s) = E[Φ(s + ∑_{j ∉ S} g_j(X_j))]`. -/
noncomputable def Hfun (q : Fin m → Dist β) (Φ : ℝ → ℝ) (g : Fin m → β → ℝ)
    (S : Finset (Fin m)) (s : ℝ) : ℝ :=
  ∑ x, (Dist.pi q).p x * Φ (s + ∑ j ∈ Sᶜ, g j (x j))

omit [DecidableEq β] in
theorem Hfun_strictMono (q : Fin m → Dist β) {Φ : ℝ → ℝ} (hΦ : StrictMono Φ)
    (g : Fin m → β → ℝ) (S : Finset (Fin m)) : StrictMono (Hfun q Φ g S) := by
  intro s s' hss'
  obtain ⟨x₀, hx₀⟩ : ∃ x₀, 0 < (Dist.pi q).p x₀ := by
    by_contra hneg
    have hle : ∀ x, (Dist.pi q).p x ≤ 0 := fun x => not_lt.mp fun h => hneg ⟨x, h⟩
    have : ∑ x, (Dist.pi q).p x ≤ 0 := Finset.sum_nonpos fun x _ => hle x
    rw [(Dist.pi q).sum_one] at this
    norm_num at this
  apply Finset.sum_lt_sum
  · intro x _
    exact mul_le_mul_of_nonneg_left (hΦ.monotone (by linarith)) ((Dist.pi q).nonneg x)
  · exact ⟨x₀, Finset.mem_univ _, mul_lt_mul_of_pos_left (hΦ (by linarith)) hx₀⟩

/-- **Independence: `E[Φ(∑ g_j(X_j)) | X_S] = H_S(s_S)`.** -/
theorem condExp_pi (q : Fin m → Dist β) (Φ : ℝ → ℝ) (g : Fin m → β → ℝ) (S : Finset (Fin m))
    {x : Fin m → β} (hx : 0 < (Dist.pi q).p x) :
    condExp (Dist.pi q) (fun x => Φ (∑ j, g j (x j))) (maskOpt S) x =
      Hfun q Φ g S (∑ j ∈ S, g j (x j)) := by
  set fib := univ.filter (fun x' : Fin m → β => maskOpt S x' = maskOpt S x) with hfib
  have hmem : ∀ y, y ∈ fib ↔ ∀ j ∈ S, y j = x j := fun y => by
    rw [hfib, Finset.mem_filter, maskOpt_eq_iff]
    simp
  set F : (Fin m → β) → ℝ := fun y => Φ (∑ j ∈ S, g j (x j) + ∑ j ∈ Sᶜ, g j (y j)) with hF
  have hFdep : ∀ y y', (∀ j ∉ S, y j = y' j) → F y = F y' := by
    intro y y' h
    simp only [hF]
    congr 2
    exact Finset.sum_congr rfl fun j hj => by rw [h j (Finset.mem_compl.mp hj)]
  have hnum : ∑ x' ∈ fib, (Dist.pi q).p x' * Φ (∑ j, g j (x' j)) =
      ∑ x' ∈ fib, (Dist.pi q).p x' * F x' := by
    refine Finset.sum_congr rfl fun x' hx' => ?_
    have hS : ∑ j ∈ S, g j (x' j) = ∑ j ∈ S, g j (x j) :=
      Finset.sum_congr rfl fun j hj => by rw [(hmem x').mp hx' j hj]
    rw [← Finset.sum_add_sum_compl S, hS]
  have hmass : 0 < ∑ x' ∈ fib, (Dist.pi q).p x' :=
    hx.trans_le (Finset.single_le_sum (fun x' _ => (Dist.pi q).nonneg x')
      ((hmem x).mpr fun _ _ => rfl))
  show (∑ x' ∈ fib, (Dist.pi q).p x' * Φ (∑ j, g j (x' j))) / (∑ x' ∈ fib, (Dist.pi q).p x') = _
  rw [hnum, sum_fib_indep q S x F hFdep, mul_div_cancel_left₀ _ hmass.ne']
  rfl

variable {Ωξ : Type*} [Fintype Ωξ]

/-- **The model satisfies the hypothesis of the induction.**  With independent blocks and
independent centred noise, `E[Y | X_S] = H_S(s_S)` with `H_S` strictly increasing. -/
theorem condExp_model (q : Fin m → Dist β) (Pξ : Dist Ωξ) (ξ : Ωξ → ℝ)
    (hξ : expect Pξ ξ = 0) {Φ : ℝ → ℝ} (hΦ : StrictMono Φ) (g : Fin m → β → ℝ)
    (S : Finset (Fin m)) :
    ∃ H : ℝ → ℝ, StrictMono H ∧ ∀ ω : (Fin m → β) × Ωξ, 0 < ((Dist.pi q).prod Pξ).p ω →
      condExp ((Dist.pi q).prod Pξ) (fun ω => Φ (∑ j, g j (ω.1 j)) + ξ ω.2)
        (fun ω => maskOpt S ω.1) ω = H (∑ j ∈ S, g j (ω.1 j)) := by
  refine ⟨Hfun q Φ g S, Hfun_strictMono q hΦ g S, fun ω hω => ?_⟩
  have hx : 0 < (Dist.pi q).p ω.1 := by
    have h : 0 < (Dist.pi q).p ω.1 * Pξ.p ω.2 := hω
    rcases ((Dist.pi q).nonneg ω.1).lt_or_eq with h1 | h1
    · exact h1
    · rw [← h1, zero_mul] at h
      exact absurd h (lt_irrefl 0)
  have hsplit := condExp_add ((Dist.pi q).prod Pξ) (fun ω => Φ (∑ j, g j (ω.1 j)) * 1)
    (fun ω => 1 * ξ ω.2) (fun ω => maskOpt S ω.1) ω
  simp only [mul_one, one_mul] at hsplit
  rw [hsplit]
  have h1 := condExp_prod_left (Dist.pi q) Pξ (maskOpt S) (fun x => Φ (∑ j, g j (x j)))
    (fun _ => (1 : ℝ)) ω
  have h2 := condExp_prod_left (Dist.pi q) Pξ (maskOpt S) (fun _ => (1 : ℝ)) ξ ω
  simp only [mul_one, one_mul] at h1 h2
  have hone : expect Pξ (fun _ => (1 : ℝ)) = 1 := by
    unfold expect
    simp [Pξ.sum_one]
  rw [h1, h2, hone, hξ, mul_one, mul_zero, add_zero, condExp_pi q Φ g S hx]

/-- **Prop. suffclass (Prop. tasks (a)), finite form.**  Let the raw input consist of
independent blocks `X_j` (values in a finite set), and `Y = Φ(∑_j g_j(X_j)) + ξ` with `Φ`
strictly increasing and `ξ` centred and independent of `X`.  Every funnel forest whose graph is a
rooted tree in which each block is the raw input of exactly one tree, and which reads every
block, is sufficient: its ideal output is `E[Y | X]` on the support. -/
theorem suffclass (q : Fin m → Dist β) (Pξ : Dist Ωξ) (ξ : Ωξ → ℝ) (hξ : expect Pξ ξ = 0)
    {Φ : ℝ → ℝ} (hΦ : StrictMono Φ) (g : Fin m → β → ℝ) (τ : BTree m) (hD : τ.Disj)
    (hup : τ.up = univ) (ω : (Fin m → β) × Ωξ) (hω : 0 < ((Dist.pi q).prod Pξ).p ω) :
    msg ((Dist.pi q).prod Pξ) (fun ω => Φ (∑ j, g j (ω.1 j)) + ξ ω.2) (fun ω => ω.1) τ ω =
      condExp ((Dist.pi q).prod Pξ) (fun ω => Φ (∑ j, g j (ω.1 j)) + ξ ω.2) (fun ω => ω.1) ω := by
  rw [suffclass_alg _ _ _ g (fun S => condExp_model q Pξ ξ hξ hΦ g S) τ hD ω hω, hup,
    condExp_maskOpt_univ]

end Model

end CobwebForest
