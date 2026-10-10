# Verifying the theory

`theory/main.tex` is checked by two independent verifiers.

## 1. Executable verifier

`verification/checks.py` builds the objects each statement talks about, using the real C++ tree. Some trees are learned by `ifit`; others are built exactly for the constructions in the proofs (`cobweb_forest/construct.py` loads any prescribed hierarchy through the JSON reader). Each check then tests the statement's conclusion. `python -m verification.run` writes `theory/verification.json` and `theory/tables/verification.tex`, which the appendix includes. `tests/test_theory.py` asserts every check.

The first check is the most important one. It implements Def. *multi-node prediction* and Def. *concept* from the definitions and compares them with the code's scores and mixtures. It caught a discrepancy: the implementation seeds best-first search with the root's **likelihood** under every weighting. Under `posterior` this differs from π_root = 0, so the definition and the posterior-inconsistency argument were corrected.

## 2. Adversarial reviewer (an independent agent)

The reviewer prompt is `verification/REVIEW_PROMPT.md`. The reviewer reads only the document and the two C++ files that the model formalises. It returns a verdict for every statement (VALID / GAP / INVALID / UNCHECKED), with evidence, and it is told to settle questions by computation.

**The verifier is verified before it is trusted.** It is first run on a copy of the document with eight planted errors. The answer key is kept outside the repository, and the reviewer cannot see the original. Three of the planted errors are real mistakes from the first draft:

| planted in | error |
|---|---|
| `prop:prior` | the leaves' mass "tends to 1" (it is a Riemann sum of fixed mesh β that oscillates with x/h) |
| `lem:leaves` | "leaves are in bijection with the distinct instances" (a duplicate can be sorted to another leaf) |
| `thm:single` | the rate drops the kernel-tail term (ρ = h√(2 log 1/h) leaves an O(h)‖g‖ tail) |
| `prop:suffclass` | Φ only continuous (continuity does not give injectivity of H_S) |
| `thm:label` | the bound drops the factor K in the count of mismatching leaves |
| `thm:locality` | "errs with probability ≥ 1/4 in each world" (the argument gives ≥ 1/2 in one world) |
| `thm:selection` | selection on the training stream (memorising trees make in-sample error ≈ 0 for every candidate) |
| `prop:multiview` | posterior precision Σ P_v − m P_0 (should be (m−1) P_0) |

The calibration score is recall (planted errors found with a correct diagnosis) and false alarms (INVALID verdicts on unmutated statements that do not survive scrutiny). Only after calibration is the same prompt run on the real document.

## Results

(Filled in below as the runs complete.)
