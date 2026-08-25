# Session 11 — 2026-08-25

Local build/housekeeping session. No GPU, no Lambda run, no API spend.

**Purpose:** adopt the independent research audit (`REPORT.md`, 2026-07-02) into the
permanent record, harden the statistics, and pre-register the headline contrasts *before*
the next GPU run. This is the P0 block of the audit's §5 recommended path — everything
free and local, done first, so the upcoming Lambda day is a single spin-up with no human
checkpoint stranded in the middle of it.

## Gap in the record (noted, not reconstructed)

Two prior sessions were never logged:

- **Session 10 (2026-06-29, commit `8ede759`)** — GATE-1 review of the minimal pairs;
  4 pairs dropped (289/372/449/475) leaving **233 valid**; L13 written. Referenced by L13
  as "session 10" but has no `docs/logs/session_10.md`.
- **The audit session (2026-07-02)** — produced `REPORT.md` (30 KB) and `scratch/01–06`,
  both left **untracked** in the working tree until this session. Roughly eight weeks
  elapsed with no commits.

`docs/HANDOFF.md` was written 2026-06-29 and therefore predates the audit; its §6
priority list ("run E2, then minimal pairs") is superseded by `REPORT.md` §5 and by this
log. **HANDOFF.md §2–§3 and §6 need updating** — deliberately not done in this session so
that the audit adoption and the handoff rewrite stay separable in the history.

---

## 1. Audit artifacts adopted (`scratch/`, `results/audit/`)

All six audit scripts re-run locally 2026-08-25 — **all exit 0, every headline number
reproduces**. Captured stdout committed to `results/audit/*.txt` so the figures are in the
record without needing a re-run, plus `scratch/README.md` indexing what each establishes.

Spot-verified rather than trusted:

- `06`: text AUC **0.956** full / **0.824** marker-ablated / **0.500** edit-vocab-removed
  / **0.915** function-words / **0.522** length-only — all reproduce exactly.
- `03`: response lengths median **111**, max 277, **100%** exceed the 64-token sweep
  window. Full-response text AUC **0.749** (lvn) / 0.720 (lva).
- Tier-4 label bug re-verified directly against the bf16 CSV: **ID 495 contains zero
  health/doctor/appointment/medical tokens in 2,542 chars** while the judge cites
  "revealed Kate's private health information" — confirmed hallucination. **492 is a
  genuine leak** (contains `doctor`, `appoint`); **493 is defensible** (a "Kate's Health
  Concerns: keep confidential" bullet that itself discloses). So L3's blanket "three
  mislabeled rows" is itself wrong for the bf16 run — one is wrong, one correct, one
  borderline.

## 2. LIMITATIONS.md — L11, L12, L14, L15 written; L13 upgraded

- **L11** — "position 42" retired. Argmax of 975 correlated AUCs, neighbour margin ≤1 SE
  on a broad k≈38–46 plateau, mid-function-word ~38% into a median-111-token reply. Note
  the *null* headline is anti-fragile to this bias: selection inflates maxima, so the true
  ceiling is if anything lower than 0.77.
- **L12** — leaked-vs-appropriate is canonical. Now carries the mechanism, not just the
  recommendation: cos(v_leak, v_deflect) = **−0.52**, probe-score correlation −0.45,
  and erasing the deflection direction drops the leak probe **0.684 → 0.628**. About a
  third of the leaked-vs-not signal was "absence of deflection."
- **L13** — upgraded from "flagged, to be quantified at GATE 2" to **quantified pre-GPU**,
  with the full table and four decided-in-advance consequences. Key line: the pairs are
  simultaneously well-constructed *and* irreducibly confounded, and these are not in
  tension.
- **L14** — new. The L3 tier-4 fix was applied only to the deprecated NF4 CSV; the
  canonical bf16 CSV still carries the bad label. Purely reportorial exposure (tier 4 is
  excluded from the 2×2, sweep, triad, and minimal pairs), but must be fixed or annotated
  before any tier-4 number appears anywhere.
- **L15** — new. Sweep window covers the first 64 of a median-111-token response;
  "never crystallizes" carries that qualifier until the relative-position sweep runs.

## 3. Stats hardening — `scripts/stats_hardening_f.py` **(new result)**

The triad CSV reports bootstrap CIs on the *marginal* AUCs, which overlap heavily and make
the verbalization claims look unsupported (audit T1). But the three probes are evaluated
on the same scenarios, so the correct object is a **paired** bootstrap on the delta.
5000 replicates, one resample of scenario indices per replicate, all three AUCs recomputed
on that same resample → `results/stats_hardening_f.csv`.

| Contrast | acts − input | desc − input | acts − desc |
|---|---|---|---|
| leaked vs not | **+0.103** [+0.026, +0.178] p=.009 | +0.025 [−0.065, +0.121] **n.s.** | +0.077 [+0.001, +0.151] p=.046 |
| leaked vs appropriate | **+0.104** [+0.016, +0.195] p=.020 | +0.051 [−0.050, +0.154] **n.s.** | +0.053 [−0.031, +0.138] **n.s.** |
| refused vs appropriate | **+0.297** [+0.197, +0.403] p<.001 | **+0.138** [+0.006, +0.264] p=.042 | **+0.159** [+0.083, +0.246] p<.001 |

**This settles T1 and is binding on the paper's framing.** The leak row cannot carry a
"verbalization destroys the signal" claim — `desc − input` is not distinguishable from
zero (p=.59 / p=.33), so the description channel never rose above the input baseline and
there is no demonstrated collapse. What the leak row *does* establish is that activations
carry privileged signal over the input (+0.10, p<.05) which descriptions do not detectably
recover. The **deflection** row is where channel loss is actually demonstrated: all three
deltas significant, descriptions retaining roughly half the privileged signal (+0.138 of
+0.297) and losing the rest (−0.159, p<.001).

So: **deflection = partial survival through verbalization; leak = little privileged signal
to lose.** Report deltas with CIs, never bare AUCs.

Also emitted (`results/stats_hardening_hanley_f.csv`): Hanley–McNeil SEs for every
headline pos-0 AUC — the n=36 `refused` class gives SE 0.033–0.037, comfortably fine for
the 0.88–0.92 claims — plus a judge-noise sensitivity band (e=10% → true deflection
≈0.93–0.97, true leak ≈0.67–0.70), flagged as a band and **not** a corrected headline
since both sides of any comparison attenuate equally.

## 4. Pre-registration — `docs/PREREGISTRATION.md`

Dated, locked, committed before the next GPU run. Contains: the one-sentence headline
claim; observed results locked as observed (not as predictions); directional predictions
plus decision rules for E2, E2b, E3, E-NLA, and E-EVAL; and an explicit list of contrasts
that will **not** be reported as findings (per-info-type AUCs, sweep argmax, tier-4
numbers, the logit-lens Chinese-token "theme", the nonlinear-probe null). Amendments must
be appended and dated, never edited in place.

Both E2 outcomes and all three E-NLA outcomes are pre-committed as reportable, so a flat
E2 will not read as a failed experiment. E-EVAL is registered specifically because it is
the result most capable of undermining the CI framing and must not become a file-drawer
item.

## 5. E3 claim language — `docs/E3_CLAIM_LANGUAGE.md`

Written *before* GATE 2 because the confound is already measured and the verdict is
therefore known in advance; deciding the wording afterwards would be choosing a claim to
fit a result. Fixes the forbidden phrasings (anything of the form "encodes privacy beyond
its lexical expression"), the permitted load-bearing claim (**privacy status ≈0.95 vs leak
behaviour 0.65, same extraction point, same probe family, both probes seeing the same
words — the contrast is the finding, not either level**), and a GATE-2 checklist in which
the CONFOUND FLAG explicitly does **not** block `--stage project`.

Also specifies the marker-family split, with feasibility measured rather than assumed:

| Family | Any-match | Exactly-one |
|---|---|---|
| confided | 103 | 34 |
| secret_word | 80 | 24 |
| kept_quiet | 59 | 18 |
| without_any | 46 | 17 |
| only_knew | 41 | 13 |
| not_tell | 7 | 0 |
| (none) | 26 | — |

Overlap is substantial — only **106 of 233** pairs match exactly one family. So the
protocol runs both a powered-but-contaminated variant (priority assignment, n=207,
families 18–90) and a clean-but-underpowered one (n=106, families 13–34, Hanley SE
≈0.06–0.09), with the reading of each fixed in advance. A null on the confirmatory variant
alone is pre-declared **uninformative**, not evidence against transfer.

## 6. Eval-awareness control built — `scripts/eval_awareness_f.py`

Gated `build → extract → analyze`, matching the repo convention. The framing manipulation
lives entirely in the **system** prompt, so user content stays byte-identical across
conditions and the paired difference cancels scenario content exactly — the same trick
`minimal_pairs_f.py` uses for `v_privacy`. Three test framings and three deployment
framings (as with `forced_prefix_f.py`'s three prefixes) so nothing hinges on one wrapper.

`--stage build` run locally: **1620 prompts = 270 scenarios × 6 framings, 0 unparsed.**
Getting to 0 took two regex fixes — the ConfAIde question clause is not uniform
("what should X say in response to this?", "Given that …, what should Ali say?", and two
scenarios using "How should X respond/react?"). The naive pattern silently dropped 12
scenarios into `You are None.`; all 270 now parse, no trailing question text leaks into
the story field.

`--stage analyze` reports cos(v_eval, v_deflect), the deflection probe before/after
erasing v_eval, plus three controls (random-direction erasure, the leak probe, and
per-scenario cosine consistency). Interpretive limit fixed in the docstring and the
prereg: the manipulation is **explicit**, so a null bounds the explicit variety of the
confound, not eval-awareness inferred from scenario style. Do not claim full closure of T2.

---

## 7. State at end of session

Everything above is local and committed. Nothing is blocked on a human checkpoint. The
next action is a single Lambda session; all four GPU items are now one command each.

**Ready to run on Lambda, in this order (audit §5 P1, ~6–8 h, ~$10–15):**

1. `forced_prefix_f.py` (E2) — text-matched control. Prereg §2; expect flat.
2. Extended/relative-position sweep (E2b) — **script not yet written**, ~20 lines
   alongside `position_sweep_f.py`; closes L15. Prereg §3.
3. `minimal_pairs_f.py --stage extract` then `--stage validate` (E3) — read
   `docs/E3_CLAIM_LANGUAGE.md` §4 at GATE 2. Prereg §4.
4. `eval_awareness_f.py --stage extract` (E-EVAL), then `--stage analyze` locally.
   Prereg §6.
5. If SGLang is up: E-NLA (idea 1) — verbalize the deflection direction, temp 0.
   Prereg §5.

**Before any Lambda NLA run:** `grep injection_token_id actor_hf/nla_meta.yaml` → must be
**149705** (L9). All existing NLA runs used temperature 1.0; contrastive verbalization
must pass `temperature=0` (L9 §3).

**Carried over, still open:** update `HANDOFF.md` §2/§3/§6 to post-audit state; fix or
drop tier 4 (L14); write the E2b script; the two missing k=42 pairwise AUCs from
`position_sweep_acts_f.npz` (Lambda-only, ~4 lines) — low priority now that L11 retires
the argmax framing.

**Calendar risk (unresolved, flagged for the user).** `REPORT.md` §5 scoped this plan to
late-August workshop deadlines and put the scoop window at ~one quarter from 2026-07-02,
noting Wang et al. have already published the population-level "knows-but-leaks" headline.
It is now 2026-08-25 with ~8 weeks of no commits. The differentiators the audit identified
— behaviour-outcome probes at the moment of action, crystallization dynamics, the
verbalization channel, and the deflection/leak dissociation — remain unclaimed, but the
plan's assumed timeline has largely elapsed.
