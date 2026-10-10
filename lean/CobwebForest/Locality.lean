/-
Copyright (c) 2026 Cobweb Forest authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Cobweb Forest authors
-/
import Mathlib

/-!
# Contextual cues are invisible locally

Signs are encoded as booleans (`true` = negative), so `sign(x₁x₂) = xor`.

* `sign_worlds_local_law` (theory: Thm. locality obstruction (i)): in the worlds
  `P_a : Y = x₁ ⊕ x₂` and `P_b : Y = x₁ ⊕ x₃` (with `x₃` unobserved), the pair `(x₂, Y)` has
  the same law, so every rule that reads only the stream of a tree perceiving `x₂` decides the
  same way in both worlds -- although `x₂` determines `Y` together with `x₁` in `P_a` and is
  irrelevant in `P_b`.
* `parity_fibre_balanced` (theory: Cor. parity): for the parity of `n` bits, every proper
  subset `S` of the bits leaves the parity balanced on each of its fibres, i.e. the parity is
  independent of the bits in `S`.
-/

open Finset

namespace CobwebForest

abbrev Ω3 := Bool × Bool × Bool

def x₁ (ω : Ω3) : Bool := ω.1
def x₂ (ω : Ω3) : Bool := ω.2.1
def x₃ (ω : Ω3) : Bool := ω.2.2
def Ya (ω : Ω3) : Bool := xor (x₁ ω) (x₂ ω)
def Yb (ω : Ω3) : Bool := xor (x₁ ω) (x₃ ω)

/-- The law of `(x₂, Y)` under the uniform distribution: counts of each pair. -/
def lawX2 (Y : Ω3 → Bool) (a b : Bool) : ℕ :=
  (univ.filter (fun ω : Ω3 => x₂ ω = a ∧ Y ω = b)).card

def lawX1 (Y : Ω3 → Bool) (a b : Bool) : ℕ :=
  (univ.filter (fun ω : Ω3 => x₁ ω = a ∧ Y ω = b)).card

/-- **Locality obstruction (i).**  The local law of `(x₂, Y)` -- and of `(x₁, Y)` -- is the same
in both worlds. -/
theorem sign_worlds_local_law :
    (∀ a b, lawX2 Ya a b = lawX2 Yb a b) ∧ (∀ a b, lawX1 Ya a b = lawX1 Yb a b) := by
  constructor <;> intro a b <;> cases a <;> cases b <;> decide

/-- Hence any rule computed from the local law takes the same value in both worlds. -/
theorem local_rule_same {β : Type*} (R : (Bool → Bool → ℕ) → β) :
    R (lawX2 Ya) = R (lawX2 Yb) := by
  have h : lawX2 Ya = lawX2 Yb := by
    funext a b; exact sign_worlds_local_law.1 a b
  rw [h]

/-- **Locality obstruction (ii).**  In `P_a`, `(x₁, x₂)` determines `Y`; in `P_b` it does not. -/
theorem sign_worlds_differ :
    (∀ ω ω' : Ω3, x₁ ω = x₁ ω' → x₂ ω = x₂ ω' → Ya ω = Ya ω') ∧
    (∃ ω ω' : Ω3, x₁ ω = x₁ ω' ∧ x₂ ω = x₂ ω' ∧ Yb ω ≠ Yb ω') := by
  refine ⟨?_, ⟨(false, false, false), (false, false, true), rfl, rfl, by decide⟩⟩
  intro ω ω' h1 h2
  simp [Ya, h1, h2]

/-! ## Parity of `n` bits -/

variable {n : ℕ}

/-- Parity of a bit vector, in `ZMod 2`. -/
def parity (ω : Fin n → Bool) : ZMod 2 := ∑ i, if ω i then 1 else 0

/-- Flip bit `j`. -/
def flip (j : Fin n) (ω : Fin n → Bool) : Fin n → Bool := Function.update ω j (!ω j)

theorem flip_flip (j : Fin n) (ω : Fin n → Bool) : flip j (flip j ω) = ω := by
  funext i
  by_cases h : i = j
  · subst h; simp [flip]
  · simp [flip, Function.update_of_ne h]

theorem parity_flip (j : Fin n) (ω : Fin n → Bool) : parity (flip j ω) = parity ω + 1 := by
  unfold parity
  rw [← Finset.add_sum_erase _ _ (mem_univ j), ← Finset.add_sum_erase _ _ (mem_univ j)]
  have hrest : ∑ i ∈ univ.erase j, (if flip j ω i then (1 : ZMod 2) else 0)
      = ∑ i ∈ univ.erase j, (if ω i then (1 : ZMod 2) else 0) := by
    apply Finset.sum_congr rfl
    intro i hi
    have hij : i ≠ j := Finset.ne_of_mem_erase hi
    simp [flip, Function.update_of_ne hij]
  rw [hrest, show flip j ω j = !ω j by simp [flip]]
  generalize (∑ i ∈ univ.erase j, (if ω i then (1 : ZMod 2) else 0)) = r
  cases ω j <;> revert r <;> decide

/-- **Parity requires composition.**  If bit `j` is outside `S`, then on every fibre of the
bits in `S` the parity takes the values `0` and `1` equally often. -/
theorem parity_fibre_balanced (S : Finset (Fin n)) (j : Fin n) (hj : j ∉ S)
    (s : Fin n → Bool) :
    (univ.filter (fun ω : Fin n → Bool => (∀ i ∈ S, ω i = s i) ∧ parity ω = 0)).card =
    (univ.filter (fun ω : Fin n → Bool => (∀ i ∈ S, ω i = s i) ∧ parity ω = 1)).card := by
  apply Finset.card_bij' (fun ω _ => flip j ω) (fun ω _ => flip j ω)
  · intro ω hω
    simp only [mem_filter, mem_univ, true_and] at hω ⊢
    refine ⟨fun i hi => ?_, ?_⟩
    · have hij : i ≠ j := fun h => hj (h ▸ hi)
      simp [flip, Function.update_of_ne hij, hω.1 i hi]
    · rw [parity_flip, hω.2]; rfl
  · intro ω hω
    simp only [mem_filter, mem_univ, true_and] at hω ⊢
    refine ⟨fun i hi => ?_, ?_⟩
    · have hij : i ≠ j := fun h => hj (h ▸ hi)
      simp [flip, Function.update_of_ne hij, hω.1 i hi]
    · rw [parity_flip, hω.2]; decide
  · intro ω _; exact flip_flip j ω
  · intro ω _; exact flip_flip j ω

end CobwebForest
