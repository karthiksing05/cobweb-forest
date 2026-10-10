# CobwebForest (Lean 4 + Mathlib)

Formal proofs of the discrete cores of `theory/main.tex`. Build:

    export ELAN_HOME=/coc/flash10/ksingara3/.elan PATH=$ELAN_HOME/bin:$PATH
    cd lean && lake exe cache get && lake build

`verification/checks.py::check_lean` rebuilds, prints `#print axioms` for every theorem
below and fails on `sorryAx`, `sorry`, `admit` or a new `axiom`.

| paper | Lean | file |
|---|---|---|
| Prop. cues (a) | `validity_nonneg`, `condExp_const_of_validity_zero` | Cues.lean |
| Prop. cues (b), blocking | `condExp_blocked`, `conditional_validity_blocked` | Cues.lean |
| Thm. locality (i)-(ii), sign skeleton | `sign_worlds_local_law`, `local_rule_same`, `sign_worlds_differ` | Locality.lean |
| Cor. parity (J = all bits) | `parity_fibre_balanced` | Locality.lean |
| Prop. interweave | `merged_prediction`, `merged_weights` | Mixture.lean |
| Prop. fan-in | `Chain.fanin_one` | Structure.lean |
| Thm. CKA tree count | `ka_tree_count` | Structure.lean |

The theory side (these proofs) and the practice side (`cobweb_forest/cues.py`, which
estimates validities with Cobweb trees and builds the forest they call for) are checked
together: `check_lean` and `check_cues`.
