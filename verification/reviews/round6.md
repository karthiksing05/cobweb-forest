# Review round 6: draft v8 ("Cobweb forests are universal approximators"), 2026-10-10

**Setup.**
- Reviewer: a fresh general-purpose agent with the calibrated prompt, focused on the new
  main-text results with full proofs and on Section 5's numbers.
- Reviewed copy: `review_copies/review_v8`. Scripts: `scripts/round6/`.

## Outcome
- **VALID:**
  - `thm:univ-tree` (the rho-net proof, also for unequal counts);
  - `thm:univ-forest` (two-input trees, constants and the sets K_v^rho all correct);
  - `prop:fanin-main`, `rem:ka-main`, `thm:learn-forest`;
  - all Section 5 numbers against `tables/equivariance.tex`;
  - spot-checked appendix proofs: lem:dag, thm:single, thm:cka, lem:leaves (matches the C++),
    thm:consistency, thm:ideal given (A5), prop:suffclass, prop:xy, prop:multiview (Delta=0),
    thm:labelcons, thm:locality, cor:parity, thm:selection.
- **INVALID:** `prop:tasks(d)`, the claim "an explicitly computable Delta>0 when it does not".
  With one-dimensional views Delta = 0 for a view-specific target (computed: 1.8e-31).
- **GAPs:** listed with their resolutions below.

## Resolutions
| item | finding | resolution |
|---|---|---|
| prop:tasks(d) | Delta>0 half false | restricted to Delta=0; "otherwise Delta can be positive, a closed-form projection, computed for one instance" |
| thm:compose | Euclidean vs max-norm neighbourhood; no extension of g_v; depth undefined; k_v<k; margin hypothesis unmet in R2 for n=4,8; constant grows with depth | restated with max-norm B^r, McShane extension, d(v), k_v<=k, C=max C_j; a-posteriori form "if every vertex error stays <= r"; R2 now measures per-level errors: for G>=64 every input stayed inside its margin, so the slopes are fitted on in-regime runs (-0.60, -0.47, -0.55) |
| thm:learn-tree | needs i.i.d., bounded Y, depth limit as event, C^2 with bounded second derivatives, density C^1, bias/variance statement | all stated; depth_max = infinity |
| prop:rate | extension of g to the cube; C depends on beta; name the rule | McShane extension, C_{d,beta}, count-weighted leaf mixture |
| lem:stable / lem:dag | domain of g_v; lem:dag only on [0,1]^n; strict vs non-strict | g_v continuous on an open set containing a closed neighbourhood; lem:dag restated for compact K; "<= eps" with a tolerance below eps/2 downstream |
| thm:univ-forest | repeated inputs (x*x); trivial polynomials | a -> a^2 one-input trees; identity vertex |
| prop:tasks(a)-(c) | dropped hypotheses | compact support and centred integrable noise; finite variance; bounded Y, exact matching, count-weighted |
| prop:local | "a tree reading X_2" too broad | "a tree whose only input is X_2" |
| prior/posterior sentence | overstated | "not universal by this construction (flat trees); see rem:hierarchy" |
| Sec. 5 smooth sentence | contradicted at n=8 single | restricted to the forest; n=8 single tree noted as pre-asymptotic |
| L3 floor | 0.696 is sqrt(Delta/Var f); Delta/Var f = 0.484 | both stated |
| L3 label forest | fixed alpha and depth are not asymptotic | described as finite-N illustrations |
| Lean row | only the discrete skeleton | "equal; exact on the sign skeleton (Lean)" |
