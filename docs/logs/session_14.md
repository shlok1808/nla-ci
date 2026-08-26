# Session 14 — 2026-08-26 (external review round)

Local session. No GPU, no API spend. An independent design review of the two
unrun experiment families (judge replication, NLA verbalization) by a separate
coding agent with repo access, followed by verification of everything it
claimed.

**Status: both API experiments are BLOCKED pending fixes.** Do not run
`judge_replication_f.py --stage judge` or `blinded_reader_f.py --stage read`
until §2 is done.

---

## 1. What the reviewer found, and what reproduced

Five implementation blockers were raised. **All five were independently
verified in code.** Three are defects introduced in session 12–13.

| # | defect | verified how |
|---|---|---|
| 1 | Resume treats `label='error'` rows as complete, so failed calls are never retried and each arm can be scored on a different partial subset. Same defect in `blinded_reader_f.py`, where an error row additionally counts as *covered* because `call='error' != 'unclear'`. | reproduced the `done`-set construction on a synthetic frame containing an error row |
| 2 | Judge JSON is not type-validated. `bool("false") is True`, so a string-valued boolean inverts; a missing key becomes `False` and falls through to `appropriate`. | confirmed in Python |
| 3 | The `roleplay` prompt is not the single-factor manipulation its docstring claims. It also introduces an **entitlement standard** (`"entitled to"`, absent from the original) and a carveout — "acknowledging that a situation exists ... is not a violation" — that **contradicts the retained "reveals or implies" rule**, since acknowledging a situation can imply the protected fact. | string comparison against `JUDGE_ORIGINAL` |
| 4 | The pre-registered decision rule requires beating the TF-IDF baseline with non-overlapping uncertainty, but the bar is a hardcoded constant (`TEXT_PROBE_LEAK = 0.600`) and only the reader AUC is bootstrapped. Cross-reader kappa is promised in the docstring and never computed — the scorer prints the filenames it finds and does nothing with them. | grep: no `cohen_kappa` in the file |
| 5 | `verbalize_directions_f.py` rebuilds its human-readable transcript from newly generated rows only, so a fully-resumed run writes a blank/partial `.txt` while the CSV stays correct. | `lines = []` at the top of the loop, written wholesale at the end |

**The session-12 verification missed #3 because it only checked that the
*shared* text was byte-identical. It never checked whether the *added* text was
internally consistent with the text it was added to.** Generalisable lesson:
verifying that a control is unchanged is not the same as verifying that a
treatment is clean.

### Also accepted

- **The figure-1 title violates the project's own constraint.** "Deflection is
  **decided** before the model speaks" implies causal commitment; the project
  forbids causal claims and the evidence is decodability, not causation. Retitle
  before the paper. The defensible form is "deflection outcomes are strongly
  linearly predictable from pre-generation layer-20 activations, while leaking is
  only weakly predictable across the tested positions and probes."
- **Nakka et al. (arXiv:2507.02332)** is a genuine missing citation — see §4.

---

## 2. Fix list (blocking, all cheap, none needs a GPU or an API key)

**Completeness and validity, both API scripts**
- Exclude `error`/malformed rows from the resume `done` set so they retry.
- Hard-fail before scoring unless an arm has exactly 270 unique valid IDs.
- Validate judge JSON: exact five keys, genuine JSON booleans, one retry on
  violation, mark invalid rather than coercing.
- Persist raw JSON, model alias, timestamp, usage.
- Fix the blinded-reader coverage denominator.

**Judge prompt** — make arm B **frame-only**: drop the carveout and the
"entitled to" phrasing, keep "reveals or implies" and the CI definition intact,
so the manipulation is a single factor.

**Blinded reader**
- **Add an input-only baseline** (same reader, original scenario, no
  description, no response). Without it a positive result may be the NLA
  reconstructing the prompt — the documented failure mode from
  arXiv:2509.13316. *This is the most important single addition from the review.*
- Paired out-of-fold TF-IDF predictions; bootstrap the AUC *difference*.
- Actually compute cross-reader agreement.
- Demote legibility-stratified AUC to exploratory (~3 bins). The same model
  produces both the prediction and the legibility rating, so conditioning on it
  can select for fluency rather than information quality. Drop the claim that a
  single temperature-1 draw makes a positive a "lower bound" — one draw can
  inflate as easily as deflate.

**Direction verbalization**
- Replace random-pair nulls with **label-permuted difference-of-means**
  directions preserving the 36/234 split, cross-fit. Current nulls are not
  estimator-matched: `v_deflect` is a 36-vs-234 mean difference while each null
  is a single pair difference.
- Raise the null count from 5 to ≥20 (with 5, the minimum one-sided empirical
  p-value is 1/6).
- Length-normalise lexicon counts; append rather than overwrite the transcript.
- Implement the registered blinded-classification endpoint or drop it from the
  pre-registration explicitly rather than under-delivering silently.
- State in the writeup that this measures **verbalizability of a direction by
  the AV**, not causal influence on the subject model.

**Cheap validation to add**
- **AR round-trip:** descriptions → public reconstructor → reconstructed
  activations → frozen behaviour probe. Tests whether behaviour-predictive
  information survives the text bottleneck.
- **Mask-and-rescore** on the behaviour claim vs matched neutral edits (L17).

---

## 3. Deliberately not doing (workshop scale) — recorded so the choice is visible

1. **Full 2×2 prompt design (4 arms).** Running frame-only vs original only. The
   definition-change arm answers a question we do not need, since the original
   labels stay primary either way.
2. **Two independent human raters with a written codebook.** At most ~40
   single-rater labels, stratified across all three classes, sampling both
   disagreements and random agreements.
3. **Deterministic arm interleaving.** Temperature is 0 and arms run minutes
   apart; backend drift is second-order.
4. **Re-running the original benchmark to recover historical raw booleans.** Not
   recoverable — a rerun measures current test-retest, not history. Stated as a
   limitation instead.

---

## 4. Where the review overreached (verified)

**The scoop claim.** The review stated that broad novelty in "predicting privacy
disclosure/refusal from pre-response activations" is not supportable, citing
Nakka et al. (arXiv:2507.02332). Fetched and checked: that paper concerns
**public figures'** sensitive attributes, probes **attention heads**, and is a
**jailbreaking/steering** paper reporting disclosure rates rather than probe
AUCs. It is adjacent and a real must-cite, but it does not occupy this project's
ground — third-party confidences between fictional characters, residual stream,
CI-preserving deflection with zero hard refusals, no steering, plus the
verbalization channel. The reviewer's own narrower conclusion (third-party CI ×
socially natural deflection × verbalization is unclaimed) matches our reading;
the stronger framing does not.

**Minor:** cited `src/benchmark.py`; the file is `scripts/benchmark.py`. Same
commit (`cc006aa`), so findings apply.

---

## 5. State

Nothing was run. Working tree clean at `cc006aa` when the review was taken.
Next: an open "what would you do" question was put to the same reviewer before
showing it this fix list, to get an uncontaminated view on the highest-value
direction. Its answer is pending evaluation.

**Blocking:** §2 must land before any API spend on either experiment.
