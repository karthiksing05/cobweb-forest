# Review round 5 (draft v7, KAN-style main text) — 2026-10-10

Reviewer: fresh general-purpose agent with the calibrated prompt (verification/REVIEW_PROMPT.md),
extended with one instruction: check that every informal main-text statement is implied by its
precise appendix counterpart and by tables/. Reviewed copy: review_copies/review_v7.
Scripts: scripts/round5/.

## Outcome
- No INVALID statement. Every appendix theorem/lemma/proposition VALID under its hypotheses
  (thm:ideal GAP minor: Step 2 uniform law assumed, acknowledged).
- Implementation claims checked against cobweb_hybrid_node.cpp / cobweb_hybrid_tree.cpp: all match
  (SPLIT deletes only an internal best1; MERGE/fringe split keep leaves; exact match var<=1e-5 and
  |x-mean|<=1e-5; root scored by likelihood; branch-and-bound stop; reviewer built a nominal-only
  counterexample confirming rem:structure).
- All table-backed numbers in the main text and Appendix F recomputed and match (slopes
  -0.244/-0.223 single, -0.163/-0.154 forest).
- rem:hierarchy re-simulated: prior-all error x D roughly constant (~1/D); posterior leaves-only floor.

## Findings and resolutions
| id | where | severity | finding | resolution |
|---|---|---|---|---|
| A | thm:main-forest, abstract | major | omitted activation-only messages and split protocol assumed by thm:ideal | added to the theorem and the abstract |
| B | prop:main-tasks(a) | minor | omitted the graph hypothesis (Y=X1X2 as one block is a counterexample) and continuity/strictness of Phi | added |
| C | prop:main-tasks(b), prop:main-local | minor | independence omitted (X2=X1, X3=X2 counterexamples) | added "independent" |
| D | prop:main-tasks(c) | minor | omitted alpha_N M_N -> 0 and the split protocol; "exponential in n" is rem:tradeoff | added; cites rem:tradeoff |
| E | thm:main-learn | minor | NW identity stated before "exact matching"; depth limit omitted | reordered; depth limit added |
| F | main remark, rem:rates | minor | "single tree pays G^{-1/n}" is only an upper bound; Lipschitz omitted | "available bound"; Lipschitz added; rem:rates says no lower bound |
| G | abstract, tab:analogy | minor | "forests do not forget" rests on a single-tree experiment | "a Cobweb tree does not forget"; table: "none within a tree" |
| H | thm:ideal Step 2 | minor | uniform NW law up to the boundary assumed | stated as explicit assumption (A5); limitations updated |
| I | prequential remark | minor | Cesaro gives only convergence in expectation | added martingale strong-law step |

Unchecked by the reviewer: figure-only numbers, Lean/verifier tables (no repo access by design),
the parenthetical failure claim in conj:hybrid.
