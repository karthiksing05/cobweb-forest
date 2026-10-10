# Calibration review: draft with 8 planted errors

- **Reviewed:** `drafts/calibration_planted.tex`. This is draft v2 (commit d51f534) with eight planted errors; the answer key is the table in `../REVIEW.md`.
- **Reviewer:** a fresh agent with `REVIEW_PROMPT.md`.
- **Scripts:** `scripts/calibration/`. `cw.py` is the reviewer's line-by-line Python port of `ifit` from the two C++ files.
- **Date:** 2026-10-10.
- **Score:** 8/8 planted errors found, no false alarms. Every non-planted finding was a genuine error and was fixed in v3 (commit 82fe29e).

The report follows as the reviewer delivered it, lightly abridged: rows with a VALID verdict are condensed, and every GAP and INVALID finding and its evidence is kept.

---

## Referee report: `review_k7/main.tex` ("Cobweb Forests")

Five stated results are false as written:
- thm:labelcons fails for any target with atoms (it predicts 0.5 everywhere).
- prop:suffclass fails without monotonicity (counterexample with Δ = 4/9).
- lem:leaves (iii) fails: the same instance can end up in two leaves.
- thm:locality (ii): the "≥ 1/4 in each world" claim is false.
- The CU = I(Z;C) identity in the definitions is false.

thm:selection, thm:consistency and nine smaller items have gaps. The main representation results (thm:single, thm:cka, lem:dag, prop:fanin, prop:interweave) and thm:ideal hold.

**How I checked.** Every disputed point was settled by computation. Loading the compiled library would have meant opening other files, so I wrote a line-by-line Python port of `ifit`, the category-utility operators, exact-match and the likelihood from the two .cpp files (`cw.py`). The scripts are `sim_*.py`. `helper.h` was not readable, so `is_close_to_zero` is assumed to be a max-abs test. This only matters for the tolerance experiment, where every coordinate differs by ≤ 1e-6.

### 1. Per-statement table

| Statement | Line | Verdict | Sev. | Reason |
|---|---|---|---|---|
| eq:ka; remark after it | 86–92 | eq. VALID; remark INVALID | minor | "Nowhere differentiable" is wrong: Kolmogorov/Sprecher inner functions are monotone, hence differentiable a.e., and can be chosen Lipschitz. |
| def:concept | 175 | VALID | – | Matches `log_prob_instance` ((K_A+1)α), `predict_probs` (K_Aα) and `model_variance`. |
| Cobweb-tree def: "this is the mutual information I(Z;C)" | 194–198 | INVALID | minor | The code computes Σ_A I(A;C) with Gaussian plug-in entropies, not I(Z;C). |
| ifit description | 201 | VALID | – | Matches `cobweb()`. |
| def:predict | 217 | VALID | – | Matches `expand()`. |
| def:messages | 252 | VALID | – | Label descent equals `get_leaf`. |
| Remark "Why not train stage-wise" | 290 | GAP | minor | "Activation at x_i close to y_i" is false in the regime N h^d → ∞. |
| lem:bias | 308 | VALID | – | g must be Lipschitz at μ_{c,X} (McShane extension). |
| thm:single | 329 | GAP | minor | The claims are true (checked numerically), but the proof's rate step is wrong. **[planted]** |
| prop:prior | 382 | GAP | minor | With β fixed, the PRIOR root weight oscillates as h → 0 (no limit). **[planted]** |
| lem:dag | 409 | VALID | – | |
| thm:cka | 431 | VALID | – | Count is 4n²+4n. |
| prop:fanin | 453 | VALID | – | |
| rem:rates | 462 | VALID | – | The constant also depends on k, the set diameters and the sup norms. |
| thm:label | 477 | GAP | minor | Disagreeing leaves are undercounted by (K−1)^m, so the displayed bound is false. **[planted]** |
| prop:interweave | 503 | VALID | – | |
| lem:leaves | 535 | (iii) INVALID | major | A repeated instance can get a second leaf (NEW chosen over BEST). **[planted]** |
| thm:consistency | 559 | GAP | major | Fails under the implementation's 1e-5 tolerance; the depth-limit hypothesis is omitted. |
| thm:ideal | 682 | VALID (conditional) | note | Step 1 inherits the exact-match caveat. |
| prop:suffclass | 752 | INVALID | major | H_S is injective only if Φ is strictly monotone; Φ(t) = t² gives Δ = 4/9. **[planted]** |
| prop:xy | 773 | GAP | minor | The learned-forest clause is not covered by any theorem and holds only asymptotically. |
| prop:multiview | 790 | GAP | minor | Precision should be Σ_v P_v − (m−1)P_0. **[planted]** |
| thm:labelcons | 822 | INVALID | major | False when Y has atoms; the "uniformly" step is overstated. |
| rem:tradeoff | 842 | GAP | minor | No result gives fan-in learning rates. |
| rem:crossover | 855 | GAP | minor | The noise-free single-tree rate N^{-2/n} is wrong for an i.i.d. design. |
| thm:locality | 884 | (ii) INVALID | major | "Errs with probability ≥ 1/4 in each of the two worlds" is false. **[planted]** |
| cor:parity | 916 | GAP | minor | The parenthetical fails when the trees' outcomes reveal coordinates of J. |
| thm:selection | 937 | GAP | major | The proof is for held-out validation; the statement selects on training-stream error. **[planted]** |
| Abstract claims | 57–71 | GAP | minor | "and so do learned trees" is unproven; "categorization is always sufficient" overstates. |

### 2. Key evidence

- **CU ≠ MI.** Take two perfectly correlated binary attributes with concepts {(0,0)} and {(1,1)}. The gain is 2 log 2, while I(Z;C) = log 2.
- **Stage-wise remark.** At N = 1000 and h = 0.05, the median self-weight is 0.0079. The mean |act − y_i| is 0.412, against 0.051 for |act − m(x_i)|.
- **thm:single.** On a grid tree with d = 1, G·(sup error) is 2.54 → 2.44 (LIK) and 2.09 → 2.38 (PATH) for G = 20…320. So the rate holds, but the proof's tail step gives h^{1/2}.
- **prop:prior.** With x = 0.3 and β = 3, λ = 0.5831 along K = 50, 100, 200, 400 but 0.5043 along K = 51, 101, …; the claimed limit is 0.5209.
- **thm:label.** The actual error exceeds the stated bound:

  | n | K | α | actual error | stated bound |
  |---|---|---|---|---|
  | 1 | 100 | 0.01 | 0.2475 | 0.025 |
  | 1 | 100 | 0.1 | 0.45 | 0.205 |
  | 2 | 20 | 0.05 | 0.2575 | 0.2404 |

- **lem:leaves (iii).**
  - In 1-D, 8 points drawn 60 times give 9 leaves for 8 distinct instances.
  - The trace shows BEST at the root, then NEW at a node with 2 children.
  - The bijection is violated in 19/20 streams (2-D) and 19/20 (3-D).
- **thm:consistency.** X is uniform on five intervals of width 1e-6 and Y ~ Bernoulli(0.9).
  - With the code's tolerance: 10 leaves for both N = 200 and N = 800, and the prediction is 0.500 everywhere.
  - With exact matching: N leaves, and predictions of 0.84–0.97.
  - Merges also happen on ordinary data: a 1-D uniform tree stored 397 leaves for 400 points.
- **prop:suffclass.** Y = (X₁+X₂)² with uniform inputs gives Δ = E(2X₁X₂)² = 4/9; Monte Carlo gives 0.4446.
- **thm:labelcons.** With K = 5 cells, Y ~ Bernoulli(0.2 + 0.6X), N = 1500 and α = 1e-4, the tree predicts 0.500 in every cell (truth 0.26 … 0.74). The leaves-only PRIOR mixture gives 0.282 / 0.422 / 0.539 / 0.630 / 0.743, exactly the partitioning estimate.
- **rem:crossover.** Fitted exponents are −0.95, −0.51 and −0.40 for n = 1, 2, 3, against theory −1, −1/2 and −2/(n+2).
- **thm:locality (ii).** The rule "always keep" has error 0 under P_a.
- **cor:parity.** Tree A has cue X₁ and outcome X₂; tree B has cue X₁ and outcome Y. The joint cue misses X₂, but the paired streams contain (X₁, X₂, Y).
- **thm:selection.** An interpolating forest satisfies the hypotheses, has training error 0, and is selected.

### 3. Most serious issues

1. thm:labelcons is false for targets with atoms.
2. prop:suffclass is false without strict monotonicity.
3. thm:selection is not proved as stated (in-sample selection).
4. thm:consistency holds only for exact matching.
5. lem:leaves (iii) is false.
6. thm:locality (ii): "≥ 1/4 in each world" is false.
7. Minor: the thm:label bound, the thm:single rate step, the prop:prior limit, the rem:crossover exponent, prop:multiview's (m−1), cor:parity's outcomes, the reach of prop:xy and rem:tradeoff, CU ≠ MI, "nowhere differentiable", and overclaims in the abstract.
