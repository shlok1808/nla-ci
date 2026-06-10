# Session 06 — 2026-06-10

Build-out session with Fable: ran the free local analyses, resolved the injection
mechanics from source, scaffolded the entire next experimental phase (8 scripts, `_f`
suffix), and wrote `docs/METHODOLOGY_f.md`. All new files this session carry `_f`.

## 1. Injection mechanics resolved from source (L9)

Fetched `nla_inference.py` (kitft/nla-inference) and the shipped `nla_meta.yaml` from
HuggingFace and read the injection code:

- **Session 04's token-ID "fix" was misrecorded.** Shipped yaml says
  `injection_token_id: 149705`, and the client **hard-asserts**
  `tokenizer.encode('㈎') == [id]` at init — a mismatch crashes before any request.
  Both logged runs printed 149705 and completed, so everything was consistent at run
  time. The "tokenizer maps to 149785" observation was likely a measurement error
  (`convert_tokens_to_ids('㈎')` returns None for Qwen; `encode` is the right call —
  the source comments say exactly this). **Check on Lambda:** if the sed'd yaml still
  says 149785, the next run will crash with "tokenizer drift" — restore 149705.
- **`injection_scale=150` is an L2-renormalization of every injected vector** →
  the NLA is **direction-only**; magnitude is always discarded. This *revises L7's
  mechanism*: the raw diff didn't fail because its norm (~4.3) was small — it fails
  because its *direction* is off-manifold. And the α=2 counterfactual was a ~5°
  rotation → unchanged descriptions were geometrically guaranteed.
- **All previous NLA calls sampled at temperature 1.0** (client default). Contrastive
  verbalization must pass `temperature=0`. Documented as L9.

## 2. Verbalization-survival triad (run locally, results in)

`scripts/verbalization_survival_f.py` — for each attribute: input-text TF-IDF AUC
(privileged-information baseline per arXiv:2509.13316), activation probe AUC, NLA-
description TF-IDF AUC. 5-fold CV, bootstrap CIs, figure + CSV committed.

| Attribute | Input | Acts | Desc |
|---|---|---|---|
| tier (t3 vs t1/2) | 1.00 | 1.00 | 1.00 |
| topic cluster (t3, circular input baseline) | (1.00) | 0.85 | 0.61 |
| scenario length | 0.63 | 0.92 | 0.60 |
| **leaked vs not** | **0.58** | **0.68** | **0.61** |
| leaked vs appropriate (no refused) | 0.55 | 0.65 | 0.60 |
| **refused vs appropriate** | **0.62** | **0.92** | **0.76** |
| response length | 0.59 | 0.72 | 0.58 |

Headline readings:

1. **Activations carry privileged information about upcoming behavior** (+0.10 leak,
   +0.30 deflection over input text). **The NLA collapses the leak signal back to the
   input floor** — verbalization destroys precisely the privileged part. Deflection
   partially survives (0.76) — strong signals can pass the channel.
2. **Leak/deflection dissociation:** at the last prompt token the model has largely
   decided *whether to deflect* (0.92) but barely *whether a non-deflecting reply will
   leak* (0.65). Working hypothesis: **leakage is a default, not a decision** —
   analogous to the harmfulness-vs-refusal encoding dissociation in the refusal
   literature. The position sweep tests when (if ever) the leak outcome crystallizes.
3. CI-term audit: 21/496 descriptions contain privacy/secrecy terms, concentrated in
   tiers 1/2 where the prompt text is explicitly about privacy (tier-3: 2/3/2 across
   labels — nothing). *The NLA says "privacy" when the input says privacy, never when
   the privacy reasoning is latent.*

## 3. Related-work sweep (details + links in METHODOLOGY_f.md §9)

- **Wang et al. = arXiv:2604.00209.** Crucial detail: they probe at the last token of
  *judgment templates* ("is this flow appropriate?") — the model is asked to judge.
  We extract where it is about to *act*. They steer along CI-norm PCA directions
  (top-5 layers, α∈{0.5,1,2,4}) and never probe actual leak behavior. Our behavioral
  probe + acting-frame focus stay novel; steering must cite them and differentiate.
- **arXiv:2509.13316** tests whether Patchscopes/LatentQA/SelfIE verbalizations convey
  privileged info (QA/retrieval domains) — closest critique-adjacent work; doesn't
  cover NLAs or behavioral decisions. Borrowed their input-only baseline into our triad.
- **Refusal direction** (Arditi et al. 2024 + follow-ups): diff-in-means over
  contrastive sets, (layer, position) selected by validation, causal validation by
  ablation AND addition; later work splits harmfulness from refusal and finds
  multi-direction structure. Borrow all three practices for v_privacy.
- NLA paper documents posterior-collapse-style failures and the unverbalized-eval-
  awareness use case (Opus 4.6 audit) — the exact capability our domain tests.

## 4. Scripts scaffolded this session (all `_f`, all compile, none run yet except survival)

| Script | What it does | Cost |
|---|---|---|
| `verbalization_survival_f.py` | triad analysis (RUN — results above) | local, free |
| `logit_lens_f.py` | all directions through RMSNorm+lm_head, top±40 tokens | 5 min |
| `position_sweep_f.py` | teacher-forced crystallization curves, layers 10/15/20/24/28 × 65 positions × 3 targets | ~1 h |
| `forced_prefix_f.py` | Johnny's text-matched control (3 fixed prefixes) | ~30 min |
| `minimal_pairs_f.py` | 3-stage: GPT-4o-mini secret→public rewrites → paired extraction → v_privacy + non-circular 2×2 | ~$1 + 30 min |
| `patchscopes_f.py` | targeted Yes/No readout of stored activations, 4 questions (incl. placebo) × target layers {6, 20} | ~30 min |
| `introspection_f.py` | norm-awareness + self-judgment logit readouts; three-level dissociation table | ~15 min |
| `steering_f.py` | ±α·v̂ at layer 20 during generation, judge w/ coherence flag, leak rate vs α | ~2–4 h + $3 |
| `alpha_sweep_f.py` | denoised directions (PCA-50, bootstrap-mask, v_privacy), rotations to ~65°, temp 0, CI-term audit | SGLang, ~15 min |

Run order is in METHODOLOGY_f.md §2. Big npz stores (`position_sweep_acts_f`,
`forced_prefix_acts_f`) are gitignored — commit the derived CSVs/plots.

Design decisions worth remembering:
- Position sweep flags its textual-divergence confound up front; forced-prefix is the
  control; report the (E1 − E2) gap, not E1 alone.
- Minimal pairs: similarity-ratio gate (0.55–0.999) + strict-retry; **manual
  spot-check of ~10 rewrites is mandatory** before trusting v_privacy; per-pair
  cosine histogram diagnoses direction-vs-bundle; projection axis never sees leak
  labels (2×2 stays non-circular).
- Steering judge gained a `coherent` boolean — leak-rate shifts among incoherent
  responses are not causal evidence.
- Patchscopes includes a placebo question; readouts are AUCs against the existing
  reference ladder (probe 0.68/0.92, NLA 0.61/0.76, input 0.58/0.62).

## 5. METHODOLOGY_f.md

New doc covering: question evolution (3 phases), full pipeline table with status,
established findings incl. the triad, current+anticipated limitations, per-experiment
designs with hypotheses AND decision rules, unscripted ideas (AR round-trip loss is
the best of them), paper plan (title, skeleton, venues, differentiation one-liners),
and a separate final section on fundamentally different approaches (Patchscopes/
SelfIE/LatentQA, SAEs, logit/tuned lens, activation+attribution patching, probing
variants incl. LEACE erasure, RepE, crosscoders, introspection) each with tradeoffs
vs NLA, plus related-work borrow/avoid notes.

## 6. What's next (Lambda session)

1. `grep injection_token_id actor_hf/nla_meta.yaml` — restore 149705 if sed'd (L9).
2. Run order: logit_lens → position_sweep → forced_prefix → minimal_pairs (export
   OPENAI_API_KEY; then SPOT-CHECK 10 rewrites) → patchscopes → introspection →
   [SGLang up] alpha_sweep → steering overnight.
3. Commit derived CSVs/plots after each; update LIMITATIONS as issues surface.
4. Then: rebuild the 2×2 analysis notebook around v_privacy projections, draft the
   triad + crystallization figures into the paper skeleton.

## Pipeline status

| Step | File | Status |
|------|------|--------|
| 1–5 | setup / benchmark / extraction / NLA / diff-of-means | done (sessions 1–4) |
| 6 | `probe_diagnostics.py` | done (session 5) |
| 7 | `verbalization_survival_f.py` | **done (this session)** |
| 8–15 | `position_sweep_f` / `forced_prefix_f` / `minimal_pairs_f` / `steering_f` / `patchscopes_f` / `introspection_f` / `alpha_sweep_f` / `logit_lens_f` | scaffolded, awaiting Lambda |
