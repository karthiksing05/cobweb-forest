"""Run every executable check; write theory/verification.json and theory/tables/verification.tex."""
import json
import os
import sys
import time

from cobweb_forest import code_hashes
from verification import checks as C

TITLES = {
    "def:predict": "Def.~multi-node prediction and Def.~concept equal the implementation",
    "lem:leaves": "Leaf preservation (no leaf deleted; additivity; leaves hold identical copies)",
    "lem:bias": "Concept-bias certificate holds for every weighting and budget",
    "thm:single+prop:prior": "Grid tree: error and root weight vanish (lik, path, leaves); prior plateaus",
    "thm:label": "Label-forest construction meets its error bound; labels are nearest grid points",
    "prop:interweave": "Interwoven tree = convex combination of the subtrees (lik, path)",
    "lem:dag+thm:cka": "Composed constructed trees approximate $\\Phi(\\varphi_1+\\varphi_2)$",
    "thm:consistency": "Learned tree, leaves-only lik = Nadaraya--Watson (path is not)",
    "prop:xy": "$xy$: funnel activations vanish, funnel forest fails; label error falls with $N$ at every depth",
    "thm:locality": "Sign construction: local laws equal; composition separates the worlds",
    "prop:multiview": "Posterior given all views = function of per-view posterior means",
    "rem:structure (branch-and-bound)": "Branch-and-bound stop returns a sub-optimal concept (continuous and nominal-only trees)",
    "prop:suffclass": "$H_S$ strictly increasing; funnel forest on a monotone-additive target converges",
    "cor:parity": "3-way parity: pairs independent of $Y$; label forest learns it, funnel does not",
    "thm:ideal": "Split protocol: sup error to the ideal forest falls with $N$",
    "thm:selection": "Validation at the output never selects the funnel forest on $xy$",
    "app:verify (provenance)": "Every reported result comes from the fixed build or a path the bug cannot reach",
    "thm:consistency+thm:labelcons (counts)": "Count-weighted leaves = Nadaraya--Watson and partitioning estimate with repeats; count-blind leaves fail",
}


def main():
    only = sys.argv[1:]
    out = dict(code=code_hashes(), checks=[])
    for fn in C.ALL:
        if only and fn.__name__ not in only:
            continue
        t0 = time.time()
        try:
            r = fn()
        except Exception as e:                       # a crash is a failed check, not a skipped one
            r = dict(label=fn.__name__, passed=False, error=repr(e))
        r["seconds"] = round(time.time() - t0, 1)
        r["check"] = fn.__name__
        out["checks"].append(r)
        print(("PASS " if r["passed"] else "FAIL ") + r["label"], f"({r['seconds']}s)", flush=True)
    os.makedirs("theory/tables", exist_ok=True)
    with open("theory/verification.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    rows = [r"\begin{tabular}{@{}llp{9.2cm}@{}}\toprule", r"result & statement & check\\\midrule"]
    for r in out["checks"]:
        lab = r["label"].split(" ")[0]
        refs = ", ".join(r"\cref{" + x + "}" for x in lab.split("+"))
        rows.append(("pass" if r["passed"] else r"\textbf{fail}") + f" & {refs} & {TITLES.get(r['label'], r['label'])}\\\\")
    rows.append(r"\bottomrule\end{tabular}")
    with open("theory/tables/verification.tex", "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows))


if __name__ == "__main__":
    main()
