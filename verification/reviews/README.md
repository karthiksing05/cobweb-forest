# Reviews of theory/main.tex

Each round is a fresh, independent reviewer agent. All rounds use `REVIEW_PROMPT.md`; from round 2 on, the prompt adds a check of the experiments text against the tables. The reviewer may read only the draft, its bibliography and tables, and the two C++ files the model formalises.

| file | draft reviewed | outcome |
|---|---|---|
| `calibration.md` | `drafts/calibration_planted.tex` (v2 with 8 planted errors) | 8/8 found; no false alarms |
| `round2.md` | `drafts/round2_v3.tex` | no theorem false; text/table and remark issues |
| `round3.md` | `drafts/round3_v4.tex` | no theorem false; weightings on hierarchies (verified by check C19) |
| `round4.md` | `drafts/round4_v5.tex` | no statement invalid; 4 minor gaps + 5 text items |

- **`scripts/<round>/`** holds the reviewer's own simulations, copied from its scratch directory. `scripts/calibration/cw.py` is an independent Python port of `ifit` written from the C++ source.
- **The calibration answer key** is the planted-error table in `../REVIEW.md`.
- **`../REVIEW.md`** records how each finding was resolved.

**`review_copies/`** holds the exact working directories each reviewer was pointed at, moved here from the flash root:
- `review_k7`: calibration copy with planted errors;
- `review_m5`: round 2;
- `review_n7`: round 3;
- `review_p8`: round 4;
- `review_r3`: an early copy of v2, staged but never reviewed.

The reviews' path references (`/coc/flash10/ksingara3/review_*`) refer to these.
