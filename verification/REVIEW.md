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

### Calibration (copy with 8 planted errors)

**Recall: 8/8.** Every planted error was found with a correct diagnosis and concrete evidence. The reviewer wrote a line-by-line Python port of `ifit` and the scoring from the C++ files, and settled each disputed point by simulation:

| planted in | verdict | evidence the reviewer produced |
|---|---|---|
| `prop:prior` | GAP (minor) | λ oscillates with x/δ: 0.583 vs 0.504 along two subsequences (β = 3) |
| `lem:leaves` (iii) | INVALID (major) | 9 leaves for 8 distinct instances; the bijection fails in 19/20 streams |
| `thm:single` | GAP (minor) | the tail with ρ = h√(2 log 1/h) is h^{1/2}, absent from the bound |
| `prop:suffclass` | INVALID (major) | Y = (X₁+X₂)²: Δ = 4/9 (Monte Carlo 0.4446) |
| `thm:label` | GAP (minor) | a table where the actual error exceeds the stated bound (0.2475 vs 0.025) |
| `thm:locality` | INVALID (major) | "always keep" errs with probability 0 under P_a |
| `thm:selection` | GAP (major) | an interpolating forest satisfies the hypotheses and wins in-sample |
| `prop:multiview` | GAP (minor) | m = 3: true precision 7, stated 6 |

**False alarms: none.** Every INVALID or GAP outside the planted set pointed at a genuine problem in the text the calibration copy shared with the real draft. These were fixed in v3:

- **Label forests failed when the target has atoms.** Identical (cell, y) instances merge into one leaf, and the count-blind leaves-only `lik` mixture predicts 0.5 in every cell of a Bernoulli target. Fix: count-weighted leaves (leaves-only `prior`), which equal the partitioning estimate exactly. The new check C18 confirms this to 2e-7, and shows the count-blind mixture is off by 0.28.
- **Single-tree consistency assumed exact matching.** The code merges instances within 1e-5. Fix: exact matching or count weighting, a remark on the tolerance, and the depth-limit hypothesis.
- **The crossover exponents were wrong** (random design). Fixed: the threshold is now n > k + 2.
- **Smaller fixes:**
  - the parity corollary now accounts for outcomes that can reveal coordinates;
  - `prop:xy` is now stated asymptotically, with the needed hypotheses;
  - the stage-wise remark is qualified;
  - McShane's extension is used, and the rate is sharpened to G^{-1/d};
  - "CU = I(Z;C)" is replaced by Σ_A I(A;C);
  - the overclaiming phrases in the abstract and the experiments are softened.

### Round 2 (real draft v3)

**No critical or major mathematical error.** The reviewer found no numbered theorem, proposition or lemma false under its own hypotheses. It re-derived every bound, constant and count, and checked the model sections against the two C++ files term by term.

Every finding was addressed in v4:

**Experiments text that contradicted the tables.** The section was rewritten from the tables. Specific fixes:
- Feynman: "every equation" now names the II.6.15a exception, "orders of magnitude" became factors of about 2 to over 100, and "matches or beats" now names its one exception.
- MNIST: "no information" is replaced by the actual accuracies.
- 100-D: "near 1" became "no design below 0.95".
- "1/7" is corrected to 1/10.
- The 14/15 selection count and the seed counts are stated.
- Housing synergy versus the induced groups is reconciled.
- Small errors are now printed to two significant figures instead of 0.000.
- New columns give the best candidate in hindsight, and a new table gives the weighting numbers.

**Experiments presented as instances of theorems whose hypotheses they violate.** The section now says so up front, and per paragraph. A scope note after (A2) says that funnel messages with critical points (unbounded densities) are outside `thm:ideal`.

**Remarks with false claims:**
- `rem:counts`: the tolerance drift is about τ log n_c, not τ, and the regime is pre-asymptotic.
- `rem:structure`: the branch-and-bound bug is caused by queue order, and it also occurs on nominal-only trees. The reviewer's nominal counterexample is now executable check C12; it returns the root at −2.151 when the best is −0.107.

**Statement gaps:**
- `conj:hybrid` now carries a variance-floor schedule. The reviewer's simulation shows the conjecture fails without one.
- `cor:parity` has a correct induction for activation forests.
- The log-loss claim is now a stated open problem.
- `thm:selection` assumes |Y| ≤ B.
- `prop:xy` has full density and bandwidth hypotheses.
- `prop:multiview` assumes m ≥ 2 and nonsingular covariances, and vector outcomes are allowed in `def:task`.
- `lem:bias` requires a non-empty mixture.
- `lem:leaves` states its tolerance test precisely.
- Online label identifiers can drift.
- The synergy deficit vanishes for injective activations; this is now stated in `def:operators`.
- The abstract and discussion no longer claim unproved rates.

### Round 3 (real draft v4)

(Pending.)
