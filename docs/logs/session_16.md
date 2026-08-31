# Session 16 — 2026-08-27: the pilot ran, and three separate things turned out to be broken

Lambda session (A100-40GB) + local analysis. The NLA transmission pilot ran
successfully. Everything after that is about what the pilot exposed.

**STOP. Read §5 before writing any code next session.** The rule for session 17
is: *understand each defect before fixing any of them, then fix one at a time
and verify it in isolation.* Three of today's problems were introduced by
earlier fixes that were applied without that discipline.

---

## 1. What actually ran

Pilot completed: **40/40 pairs, greedy decoding deterministic on 5 repeats.**

```
slice | identical | seq-sim | edits | from edit-vocab | 2AFC ceiling
 full |        0% |   0.516 |    74 |              0% |        1.000
   P3 |        0% |   0.518 |    28 |              0% |        1.000
FULL branch  PROCEED-TO-2AFC     P3 branch  PROCEED-TO-2AFC
```

Every pre-registered gate passed. The echo explanation the design was built to
catch is dead (edit-vocab ≈ 0, nonzero in only 3–10 of 40 pairs).

**And the descriptions are confabulated anyway.** The gate was not wrong — it
measured what it claimed to measure. It just does not measure what we needed to
know. See §3.

## 2. Three independent defects, in dependency order

These are *not* the same problem. Fixing one does not touch the others.

| # | defect | where | scope |
|---|---|---|---|
| 1 | judge labels wrong | `benchmark.py` → `benchmark_results_bf16.csv` | **Branch A** — every headline AUC |
| 2 | story rewrites wrong | `minimal_pairs_f.py` → `minimal_pairs_f.csv` | **Branch B** — v_privacy, 2×2, fig 3, the pilot |
| 3 | AV does not recover content | the NLA readout itself | the whole verbalization thread |
| 4 | test suite deleted committed data | `tests/test_nla_transmission_f.py` | infra (**fixed today**) |

Defects 1 and 2 are both GPT-4o-mini, doing two different jobs. Defect 3 is
independent of both — it replicates on tier 1, where no rewrite exists.

## 3. Evidence, with confidence levels

### Solid — verified by hand and/or independently reproduced

- **AV descriptions are ungrounded.** 21 character names across 80 descriptions,
  **zero** appear in their own story. "Sarah" in 43/80 descriptions vs 8/40
  stories. Topics wrong (an abortion scenario described as "leaving the team").
  Replicates on tier 1 from the separate June run: "social security number" →
  *"pure evil / Hitler"*; "phone conversations" → *"Scale of Nazis condemning
  someone"*. Codex's independent TF-IDF retrieval agrees: a description
  retrieves its own source story at median rank 22–23 of 40.
- **Within-pair > cross-pair**, full 0.535 vs 0.347, p=7.8e-11. Real. It means
  descriptions track **activation geometry** — NOT that they are correct.
- **P3 does not track angle** (rho=−0.125, p=0.44) while full/P1/P2 do
  (−0.51, −0.36, −0.53). P3 was the pre-registered primary slice.
- **No privacy vocabulary anywhere.** `confidential`/`privacy`/`private` appear
  zero times in all 80 descriptions. 150-word data-driven asymmetry test: 3 hits
  where 7.5 are expected by chance, none survive Bonferroni.
- **~40–50% of pairs do not flip privacy.** Five verified by reading (285, 382,
  329, 473, 219); regex over all 233 flags 73 (31%) with surviving secrecy
  markers, and Codex's manual coding found 21/40 in the pilot sample. **All carry
  `valid=True`.**
- **~25 of 36 `refused` look misfiled** (Codex). Four of five spot-checked by
  hand were wrong — they decline while confirming the protected fact.
- **Activation probe reproduces at 0.920** vs the published 0.919, independent
  code path.
- **Refusal markers alone predict at 0.624** — argues against a pure
  surface-register explanation for the 0.92.

### Weak — worth having, do not lean on

- Stratum test on register: n=21, and the marker lexicon caught only 39% of
  refusals. Underpowered, not decisive. Kept at `scratch/11_refusal_surface_test.py`.
- Response-text AUC 0.862 vs activation 0.920: regularization was not matched
  between them (C=1.0 vs 1e-3), no permutation null, no nested CV. Directionally
  interesting; needs a controlled protocol before it goes anywhere near the paper.

### Unknown

- Whether corrected labels materially move the headline numbers. Codex's partial
  fix moved leak 0.684 → 0.655 and left deflection ~flat, but that was five
  cases, not an audit.
- Whether the AV failure is the checkpoint, the extraction point, or our usage.
  Not separated. Anthropic's NLA trained on "the final token of randomly
  truncated pretraining-like text snippets" — our extraction point is the
  newline after `<|im_start|>assistant`, a chat-template control token. They also
  document confabulation as a known property ("false in their specifics,
  typically thematically faithful"), but our failures are worse than that: the
  theme is gone too.

## 4. Silent failure modes — the catalogue

Every defect today failed **without announcing itself**. Keep this list; check
new code against it.

- **`valid=True` on contradictory rewrites.** The gate checks `SequenceMatcher`
  ratio — surface similarity. A "public" story that still says *"kept this from
  other friends"* passes easily. Surface checks cannot see meaning.
- **The test suite deleted committed files, and the second run passed.** The
  isolation check fires *after* the damage; once files are gone, before == after.
  First run exits 1, every run after that exits 0. Only `git status` shows it.
- **The judge stored only the derived label**, never the three raw booleans, so
  nobody could see how often two fired at once and the hidden priority rule
  (leaked > refused > appropriate) silently resolved toward one.
- **Every pilot gate was green while the descriptions were confabulated.**
  0% identical, ceiling 1.000, edit-vocab 0. Nothing in the metrics could have
  caught it. Only reading the text did.
- **The probe is trained AND evaluated on the same corrupted labels**, so the
  AUC cannot reveal the damage from inside the pipeline.
- **`activations_layer20.npz` has `labels` embedded.** Relabeling the CSV does
  not propagate to it.
- **`steering_f.py` contains a second hardcoded GPT-4o-mini judge call** that
  nobody has touched.
- **`judge_replication_f.py` writes a separate QA CSV.** Running it does not
  update NPZ labels, description metadata, sweep grids, or figures. Swapping the
  model measures the problem; it does not fix anything downstream.
- **`nla_inference.py` is fetched, not vendored**, and the box runs a patched
  copy. Its SHA is in the pilot provenance — check it, don't assume.

**The pattern: every one of these was found by reading data, not by reading
metrics.** No dashboard would have caught any of them.

## 5. The rule for session 17

Three of today's problems were *created by earlier fixes* applied without
verification:

- The test-suite fix for the contamination bug (session 15) changed it from
  *writing* fake data into real `results/` to *deleting* real data from it.
- `all_artifacts()` was written to be exhaustive; that exhaustiveness is exactly
  what made the deletion destructive.
- The rewrite validation gate was added to catch bad pairs and passes ~half of
  them.

So, in order, and **do not skip ahead**:

1. **Study first.** For each defect: what exactly is wrong, what does it touch,
   and what would prove it fixed? Write that down before editing.
2. **One at a time.** Fix, verify in isolation, confirm nothing else moved
   (`git status` + the test suite + the byte-identity checks).
3. **Ground truth before measurement.** Any analysis conditioned on the judge
   labels inherits the corruption. Adjudication comes first.
4. **Pre-register before running.** Contrast, CV protocol, regularization search,
   permutation null, decision rule, and what counts as failure — dated and
   committed before any number is seen.

## 6. Branch A, ordered

1. **Freeze a revised rubric.** Three categories, not two — *full disclosure* /
   *confirmatory disclosure* (confirms the fact exists and attaches it to the
   person, without specifics) / *clean deflection*. Every label must be
   justified by a **quotable evidence span**. The old rubric defined violation in
   four sentences, appropriate handling in zero, never defined "reveals or
   implies", and asked `leaked` and `refused` as independent booleans — so a
   response that declines *while revealing* got scored on the declining.
2. **Blind adjudication**: all 36 `refused` plus a stratified sample of `leaked`
   and `appropriate`. Labels hidden, order randomized, spans recorded. This is
   the only uncorrupted ground truth in the project.
3. **Pre-registration amendment**, dated, written *before* looking at any
   recomputed AUC. Codex's framing: preserve the old numbers as the
   original-judge-label analysis; a pre-registration prevents outcome shopping,
   it does not turn demonstrated errors into ground truth.
4. **Recompute** probes against the adjudicated set.
5. **Then** decide what the paper claims.

`gpt-5.6-sol` relabeling is QA alongside this, not the fix — same provider as
gpt-4o-mini and the same ambiguous rubric means correlated errors. Disagreement
would be informative; agreement proves little.

## 7. Branch B, when Branch A is settled

Regenerate from the stories forward — rewrites, semantic validation gate (not
`SequenceMatcher`), hand spot-check, re-extract activations (**needs GPU**),
then reconsider whether the NLA thread continues at all.

Note: the 2×2 **is finished**, not unfinished — it produced
`minimal_pairs_analysis_f.csv`, `fig3_null_coupling.png`, AUC 0.421, χ²=0.68 —
and it is contaminated by **both** roots. Codex found the Figure-3 "slightly
backwards" result is fragile (projection AUC 0.421 → 0.486 after excluding
failed pairs); the backwards trend should be withdrawn, "no detected coupling"
remains plausible.

## 8. Fixed today

- `setup_lambda.sh`: the torchvision removal loop could never terminate, and
  removing torchvision at all was wrong — sglang hard-imports `torchvision.io`
  at startup. Now installs a matched build and verifies by importing.
- `nla_inference.py`: `transformers 5.12.1` returns a `BatchEncoding` where the
  library assumed a list. Patched at two call sites (box only — the file is
  gitignored; its SHA is in the pilot provenance).
- `judge_replication_f.py`: modern reasoning models reject `temperature`
  (Claude Opus 5 / 4.8 / 4.7 / Fable 5 / Sonnet 5 return 400; GPT-5.6 tiers
  undocumented). Now tries it, falls back once, records `temp0` per row, and
  refuses to swallow non-temperature errors. 20 tests in
  `tests/test_judge_replication_f.py`.
- `tests/test_nla_transmission_f.py`: `METRICS_CSV` and `PILOT_TXT` were
  redirected 25 lines *after* `all_artifacts()` was called and unlinked, so the
  suite deleted two committed pilot outputs on every run. Redirects moved up.
  Verified: files byte-identical across two consecutive runs.
