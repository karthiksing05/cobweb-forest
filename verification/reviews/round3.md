# Round 3 review: real draft v4

- **Reviewed:** `drafts/round3_v4.tex` (commit 21eb0d8), with tables as of that commit.
- **Reviewer:** a fresh agent with `REVIEW_PROMPT.md`, plus item 7.
- **Scripts:** `scripts/round3/`.
- **Date:** 2026-10-10.
- **Outcome:** no critical or major error. The most consequential finding, about the weightings on hierarchies, was verified independently by executable check C19 before the text changed. Everything was fixed in v5 (commit 7f30aaa).

The report follows as the reviewer delivered it, lightly abridged: rows with a VALID verdict are condensed, and every GAP and INVALID finding and its evidence is kept.

---

No numbered theorem, lemma or corollary is false under its stated hypotheses. The problems are:
- the paper's account of the PRIOR and POSTERIOR weightings;
- one wrong claim about the implementation in rem:structure;
- small proof slips;
- several experiments sentences that overreach the tables.

## Verdicts

Everything is VALID except the following:

| Statement | Verdict | Sev. | Reason |
|---|---|---|---|
| prop:prior (title/framing) | INVALID | minor | The body holds for the depth-one tree. On a hierarchy the PRIOR all-concept mixture is consistent. |
| Prose around prop:prior | INVALID | minor | "Both are all-concept pathologies" is false: POSTERIOR keeps a bias even leaves-only on a hierarchy. |
| rem:structure (last sentence) | INVALID | minor | The proposed repairs fail for continuous attributes. |
| thm:ideal | GAP | minor | The Step 1 tolerance parenthetical (τ) contradicts rem:counts (τ·log n). |
| prop:suffclass | GAP | minor | Integrability of Φ(s+Z) is not assumed. |
| rem:crossover | GAP | minor | The rates are interior; boundary bias changes the exponents but not the conclusions. |
| proc:induction claim | GAP | minor | thm:selection's hypothesis is not established for the HYBRID, LABEL or CROSS candidates. |
| conj:hybrid | UNCHECKED | – | A conjecture. |

**(a) rem:structure.** Two points {0, 0.1}, h² = 1e-6, query 0. The root has π = 0 and score 1.577; the leaf at 0 has score 5.989. The C++ rule, per-subtree pruning and π-ordering all return the root. Over 2999 random 1-D trees, each variant fails on the same 128. Fix: prune only when best > π_c + ℓ_max, with ℓ_max = −(|O|/2)·log(2πh²).

**(b) Weightings on a hierarchy.** Balanced binary tree over G = 2^D grid points, h equal to the spacing:

| Mixture and target | Error as G grows | Behaviour |
|---|---|---|
| PRIOR all-concept, sin | 0.269 (G=16), 0.141 (256), 0.098 (4096), 0.075 (65536), 0.068 (262144) | falls like ≈1.2/D |
| Depth-one tree, sin | 0.3757, constant | reproduces Prop prior |
| POSTERIOR leaves-only, sin | 0.0319 … 0.0303 (D = 8…18) | floor |

Tab weighting's PRIOR error × ln N is roughly constant (0.91–1.00), so it decays rather than plateaus.

**(c) thm:ideal tolerance.** The first instance ends 1.93τ, 4.18τ, 6.48τ and 8.78τ from the leaf mean at n = 10, 10², 10³, 10⁴, which is ≈ τ(H_n − 1).

**(d) prop:suffclass.** X ~ N(0,1), g = x², Φ = exp gives E[Y|X₁] = ∞.

**(e) rem:crossover.** On a compact support, boundary bias gives RMSE ~ N^{−3/(2(n+1))} noise-free. The conclusions survive.

**(f) Selection claim.** HYBRID is only conjectured; fixed-depth LABEL and CROSS are not covered.

**Smaller notes:**
- `is_exact_match` compares nominal proportions.
- "Gives 1/2" holds only as α → 0.
- Tie-break streams are shared unless each tree is seeded.
- The McShane sup norm is not preserved (harmless).
- A joint outcome (Y, X_{3−j}) would distinguish the two sign worlds.

## Experiments text vs. tables

- **E1.** The "plateau" claims are wrong: PRIOR falls 0.197 → 0.112 (sin).
- **E2.** "About an order of magnitude behind the MLP and further behind the KAN": at N=250 the KAN is worse than the MLP.
- **E3.** "0.99–1.01": xy is 1.011 at N=1000.
- **E4.** "As prop:xy predicts in the limit": the runs are outside its hypotheses.
- **E5.** The crossover reading is wrong: the measured slopes run against the predicted exponents (forest −0.16 against single −0.23 noisy, −0.15 against −0.25 noise-free).
- **E6.** diag100: the κ = 1 row is 3.16; at N=16000 with 1000 leaves, level 0 is 0.90.
- **E7.** "Guarantee concerns only the other candidates" overstates.
- **E8.** Housing rooms–bedrooms vs Tab induce.
- **E9.** "Deficit predicted" is a sign only, not a size.
- **E10.** MNIST at N=20000 is a tie.
- **E11.** The MNIST cross-view explanation is untested.
- **E12.** OWT: only the funnel label forest reaches 5.77.
- **E13.** Universal consistency in the abstract needs the count-weighted rule.

## Most serious issues

No critical or major errors. In order:
1. The account of the weightings.
2. rem:structure repairs.
3. The crossover reading.
4. Selection guarantees and synergy.
5. thm:ideal tolerance.
6. Missing hypotheses.
7. Abstract overreach.
