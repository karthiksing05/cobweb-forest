# Round 4 review: real draft v5

- **Reviewed:** `drafts/round4_v5.tex` (commit 7f30aaa), with tables as of that commit.
- **Reviewer:** a fresh agent with `REVIEW_PROMPT.md`, plus item 7. The run was interrupted by a session restart and resumed.
- **Scripts:** `scripts/round4/hier.py`, an independent simulation of `rem:hierarchy`.
- **Date:** 2026-10-10.
- **Outcome:** no INVALID statement, and no main result fails. Four minor theory gaps and five experiments-text items, all fixed in v6.

The report follows as the reviewer delivered it, lightly abridged: rows with a VALID verdict are condensed, and every GAP and its evidence is kept.

---

I checked every numbered statement. No main result is false. Every proof I could redo closes, and the arithmetic checks out: the 4n²+4n count, the (1+Kα)^n−1 bound, the Chernoff and partition bounds, and the constants of thm:single. The C++ matches the paper's description.

**VALID:**
- def:concept, the PU definition, def:predict, def:messages
- lem:bias, thm:single, prop:prior
- rem:hierarchy. Confirmed numerically: PRIOR-all error × D stays near 1.05–1.16 (sin) and 2.9–3.1 (cos 20x) for D = 4…14; POSTERIOR-leaves floors at 0.0316 and 0.148; LIK converges.
- lem:dag, thm:cka, prop:fanin, rem:rates, thm:label, prop:interweave
- lem:leaves (the helper.h details are unchecked), rem:counts
- rem:structure (the π_c+ℓ_max prune is sound)
- def:ideal, cor:iff, prop:suffclass, prop:xy, prop:multiview
- rem:crossover (the boundary-layer version keeps the crossover at n > k+2)
- thm:locality, thm:selection, def:operators, proc:induction

## Gaps

- **G1, thm:consistency.** The hypothesis "X has a density" is unused. Devroye–Krzyżak hold for every distribution of X, and with τ = 0 atoms only create copy-holding leaves, so the count-weighted mixture is still exactly Nadaraya–Watson. Fix: drop it.
- **G2, thm:ideal Step 2.** The uniform Nadaraya–Watson law up to a Lipschitz boundary, for a Gaussian kernel, is cited, not derived. The conclusion survives via the cone condition, a uniform KDE law and a bias bound. Fix: state it as a lemma or as an explicit given.
- **G3, thm:labelcons (a).** "Largest diameter of a cell meeting 𝒳" cannot hold: label maps partition all of R^n, so the edge cells are unbounded. In thm:label's grid tree, the cell of 1/(2K) is (−∞, 1/K]. Fix: use diam(C ∩ 𝒳).
- **G4, cor:parity.** "A tree whose cue misses a coordinate of J" is ambiguous once cues include messages. A child with an empty raw cue can receive E[Y|X_J] = Y. Fix: use the upstream raw set R(τ).

## Experiments text vs. tables

1. The sin posterior range "0.08–0.20": the table gives a maximum of 0.135.
2. diag100 "improves with N" holds only at the 4000-leaf budget; at 1000 leaves, level 0 goes 0.57 → 0.90.
3. Locality: "the funnel fails in both worlds" is wrong under P_b, where f ≡ 0 and the constant predictor is optimal. The normalisation under P_b must be sd(Y).
4. Induction on housing at N = 16000: the τ = 0.05 groups are the single tree, so the gain comes from another threshold. (Checked: the selected forests at τ = 0.1/0.2 merge only longitude–latitude in all three runs.)
5. "Categorization is sufficient whenever its cells shrink and stay few" omits (c), α_N M_N → 0.

## Most serious issues

1. G3, which as written empties the label-consistency theorem.
2. The housing induction attribution.
3. diag100.
4. G2.
5. G1.
6. G4.
7. The remaining text items.

No statement is INVALID.
