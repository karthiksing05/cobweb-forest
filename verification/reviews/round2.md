# Round 2 review: real draft v3

- **Reviewed:** `drafts/round2_v3.tex` (commit 82fe29e), with tables as of that commit.
- **Reviewer:** a fresh agent with `REVIEW_PROMPT.md`, plus item 7 (experiments text versus tables).
- **Scripts:** `scripts/round2/`.
- **Date:** 2026-10-10.
- **Outcome:** no critical or major mathematical error. Every finding was fixed in v4 (commit 21eb0d8).

The report follows as the reviewer delivered it, lightly abridged: rows with a VALID verdict are condensed, and every GAP and INVALID finding and its evidence is kept.

---

**Bottom line.** None of the numbered theorems, propositions or lemmas is false under its own hypotheses, so I found no critical or major errors. I re-derived every bound, constant and count. I checked the model sections against `cobweb_hybrid_node.cpp` and `cobweb_hybrid_tree.cpp` and they match.

The real problems are elsewhere:
- **Experiments text vs. tables:** several sentences contradict the paper's own tables.
- **Summaries:** the Discussion, Limitations and Abstract overstate or misdescribe the theory, including one false "exactly when".
- **Remarks and sketches:** a few have gaps, and two remarks make claims that are false as stated.

## Statement verdicts (abridged)

All of the following are VALID:
- def:concept, the PU definition, def:predict, def:messages
- lem:bias, thm:single, prop:prior, lem:dag, thm:cka, prop:fanin, rem:rates, thm:label, prop:interweave
- lem:leaves, thm:consistency, def:ideal, thm:ideal, cor:iff, prop:suffclass, prop:multiview
- rem:crossover, thm:locality, thm:selection, def:operators

The issues:

| Label | Verdict | Sev. | Reason |
|---|---|---|---|
| def:forestlearn | GAP | minor | Label identities are not stable over time in the online protocol. |
| rem:counts | INVALID | minor | Drift reaches 7.1τ (about τ(1+H_{n−1})); "h_N ≫ τ" is incompatible with h_N → 0; the Bernoulli 1/2 is conditional. |
| rem:structure | GAP | minor | Branch-and-bound also fails on nominal-only trees. The queue is ordered by score, not by π. |
| prop:xy | GAP | minor | "Source trees satisfy (A2)" is not implied by the stated hypotheses. |
| thm:labelcons (sketch) | GAP | minor | The cited theorem's conditions differ; the missing-mass step and the weight-proportionality factor are omitted. |
| conj:hybrid | GAP | minor | Needs a variance-floor condition: with h = N^{−3} it fails (RMSE/sd ≈ 1.05) while h = N^{−1/4} works (0.165 → 0.068). |
| cor:parity | GAP | minor | The last clause is unproved and not qualified as asymptotic. |
| Log-loss paragraph | GAP | minor | Unproved; it needs convergence of predictive distributions. |
| Discussion "exactly when monotone-additive" | INVALID | minor | Counterexample: f = x₁x₂ + x₁ with X₁ ~ U[−1,3]; the funnel is sufficient but f is not monotone-additive. |
| Limitations "LIK mixture" | INVALID | minor | (A3) specifies the count-weighted mixture. |
| Abstract/Intro | GAP | minor | Unproved label-forest "rate"; multi-view proved only for the Gaussian model; "not for x₁x₂" needs symmetric inputs. |
| Scope | note | – | (A2) excludes the paper's own funnel forests: messages with critical points have unbounded densities. |

**rem:structure counterexample** (nominal only, α = 0.01, query a = 0):

| Node | Contents | PATH score |
|---|---|---|
| root | children `curr`, `q` | −2.151 |
| `curr` | leaf, 1 copy of a=0 | −2.341 |
| `q` | children `d`, `e` | −2.348 |
| `d` | leaf, 9 copies of a=0 | −0.107 |
| `e` | leaf, 76 copies of a=1 | −15.84 |

`categorize` returns the root although `d` scores higher.

## Experiments text vs. tables

1. "Far more accurate on smooth low-dimensional targets" is contradicted on J0: at N=250 the single tree scores 0.104 against MLP 0.679 and KAN 0.283.
2. The weighting figure was presented as testing thm:single and prop:prior, which concern constructed trees, on learned trees with h ≍ 1/N.
3. "As thm:consistency predicts": the runs use PATH, a budget and τ > 0, all outside the theorem.
4. "Exactly as prop:xy predicts": prop:xy assumes the split protocol and count-weighted full expansion.
5. The crossover attribution is wrong: n = k+2 gives equal exponents.
6. "Every Cobweb design stays near 1": the exp100 single tree is at 1.58–1.61.
7. "1/7" should be 1/10.
8. "(A4) failing at finite N": (A4) is asymptotic.
9. Locality: the funnel under P_a rises 1.008 → 1.164.
10. Selection at nested4 N=500 picked "funnel-pairs (4), funnel (1)"; "within one percent" was not checkable from the table.
11. Housing: rooms–bedrooms 0.13 > τ, yet Tab induce keeps them apart.
12. Induction: "the groups … are the interactions" overreaches (q kept apart); II.6.15a: 0.546 against 0.519.
13. Feynman: "every equation" is false (II.6.15a hybrid 0.421 < 0.486); "orders of magnitude" is false (1.9× on I.29.16); "matches or beats" is false for II.6.15a.
14. Housing: "consistent with synergy" holds only in part.
15. MNIST: "categorization passes no information" is false (70.7–86.2% against 10% chance).
16. Text: "the forests do not help at N=20000" is false (label-funnel 5.77 against 5.88).
17. "Five seeds for sec:exp-tasks", but tab:induce uses 2–3.
18. The label depth from {2,3,4} does not grow with N, so condition (a) fails.
19. Single-tree numbers differ between tables without explanation (κ fixed vs tuned).

## Most serious issues

No critical or major mathematical error. In order:
1. Experiments text contradicts the tables.
2. False "exactly when" in the Discussion.
3. Limitations misstates thm:ideal's mixture; "1/7".
4. Experiments presented as confirming theorems whose hypotheses they violate; (A2) excludes the paper's funnel forests.
5. Gaps in the thm:labelcons sketch; conj:hybrid needs a floor condition.
6. rem:counts tolerance; rem:structure diagnosis.
7. Smaller gaps.
