# Review round 7: draft v10 (separation, learned-forest rate, recipe, case studies), 2026-10-11

**Setup.**
- Reviewer: a fresh general-purpose agent with the calibrated prompt.
- Reviewed copy: `review_copies/review_v10`. Scripts: `scripts/round7/`.

## Outcome
- No statement false under its own hypotheses.
- **VALID:**
  - thm:separation(a): counting, the inequality, membership of the bump in B_n, and convex
    combinations (checked against the C++);
  - the thm:learn-rate induction and tower step;
  - the restated thm:compose;
  - recipe deficits (Monte Carlo 0.249 vs 0.250);
  - all Section 5 and recipe-table numbers;
  - the cross-view floor 0.6956.

## Findings and resolutions
| id | severity | finding | resolution |
|---|---|---|---|
| A | major | thm:learn-rate's "N^{-2/(k+4)} under thm:learn-tree" does not follow: (iii) needs a uniform rate up to the boundary, for a design law that moves with N | the claim is now "an exponent depending only on k"; the uniform NW rate (log factors, O(h) boundary bias) is explained; Table 1 and Section 5 rephrased (the measurement is interior RMSE) |
| B | minor/major | (ii) for prop:tasks(a) needs Phi and g_j Lipschitz and H_S bi-Lipschitz, extended | added |
| C | minor | separation(b): 2n trees holds only for the bump; (1+2r) factor missing; the empty leaves-only mixture returns 0 or the root mean | at most 2^D-1 <= 8n-1 trees, G=T/(8n); factor added; empty-mixture case added |
| D | minor | recipe: right conclusion, wrong reason | E[Y|x_i,x_j] = E[Y|a_i,a_j]; product-pair activations are constant |
| E | minor | Bernstein "sup" values are sample maxima; "never does better" is wrong; kappa's A_v ambiguous; I.29.16's sqrt is not Lipschitz | "maxima over 300 points"; triangle inequality; A_v = downstream product excluding v (as computed); convergence cites lem:stable; "five of seven formulas with n>2" |
| F | minor | rem:rates and rem:tradeoff stale | updated to cite thm:separation and thm:learn-rate |
| G | minor | "-0.74" should be -0.75; margin checked at query points only | fixed; "at all 300 query points" |
