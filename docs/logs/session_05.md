# Session 05 — 2026-06-09

## What we did

- Full repo review with Fable 5 (Mythos-class): read README, CLAUDE.local.md, all four
  prior session logs, LIMITATIONS.md, every script (`benchmark.py`, `extract_activations.py`,
  `run_nla.py`, `diff_of_means.py`), and both NLA output files
  (`diff_means_output.txt`, `counterfactual_output.txt`).
- Goal: resolve the "core tension" from session 04 — Wang et al. proved CI norms are
  linearly encoded at layer 20 with near-perfect probe AUROC, but our NLA verbalization
  surfaced nothing CI-related (L6/L7).
- Wrote `scripts/probe_diagnostics.py` — runs entirely locally (no GPU/SGLang needed) on
  `results/activations_layer20.npz`. Five checks: CV logistic probe AUC, class-mean
  geometry (cosine similarity, diff norm vs within-class spread), permutation test on
  the diff vector, held-out projection onto the diff direction, and a tier-3-vs-tier-1/2
  control probe.
- Also re-verified the TF-IDF claims from session 04 against `results/nla_descriptions.csv`
  directly (leak-vs-not-leaked AUC, tier-vs-tier AUC).
- Added **L8** to `docs/LIMITATIONS.md` documenting the finding below.
- Committed and pushed: `652fa7e` — "add probe diagnostics + L8 (leak signal is AUC 0.68,
  not near-perfect)".

## Key finding: the "core tension" was a conflation of two different probe targets

**Wang et al.'s near-perfect AUROC is for CI norm *attributes*** (information type,
recipient, transmission principle) — these are properties of the *input scenario text*
and are trivially decodable (our own tier-3-vs-tier-1/2 control probe also hits AUC 1.00,
for the same reason — surface task format dominates).

**That is a different probe target from the leak/no-leak *behavioral outcome***, which is
what `diff_of_means.py` actually computed (`leaked_mean - appropriate_mean` over tier 3
labels). Nobody — not Wang et al., not us — ever established that the leak decision
itself is near-perfectly linearly decodable at the last prompt token. In fact Wang et
al.'s own privacy awareness gap (correct norm encoding, but 38.5% leakage anyway)
*requires* these two signals to dissociate — if the norm encoding fully determined the
leak decision, there would be no gap.

### Measured numbers (`scripts/probe_diagnostics.py`, tier 3, n=270, leaked=151/not-leaked=119)

| Quantity | Value |
|---|---|
| 5-fold CV logistic probe AUC (leaked vs not-leaked) | **0.68** (best at C=1e-3; 0.65–0.68 across C=0.001–1.0) |
| Activation L2 norm (mean ± std) | 89.61 ± 1.40 |
| cosine(leaked_mean, not_leaked_mean) | 0.99896 |
| ‖diff‖ vs mean within-class distance | 4.016 vs 18.48 |
| ‖diff‖ as fraction of mean activation norm | 4.48% |
| Permutation test on ‖diff‖ (500 shuffles) | observed 4.016, null mean 2.292 ± 0.284, max 3.274 → **p < 0.002** |
| Held-out projection onto train-fold diff direction | AUC **0.665** |
| Control: tier_3 vs tier_1/2 raw-activation probe | AUC **1.00** |

### What this means

- The diff-of-means direction is **statistically real** (passes the permutation test
  decisively — null distribution tops out at 3.27, observed is 4.02) but carries at most
  ~AUC-0.68 worth of separability. Roughly a third of its apparent magnitude is
  label-sampling noise relative to the permutation null.
- The NLA "failure" from L6/L7 is therefore **largely expected**, not a mystery to be
  solved. There was never a strong leak-decision signal at this extraction point for the
  NLA to surface. The honest framing is "weakly decodable, not verbalizable" — not
  "strongly decodable but unverbalizable."
- This also explains why counterfactual interpolation at α=2 produced identical
  descriptions (session 04, L7): `‖2·diff‖ ≈ 8` against a mean vector of norm ~88 is only
  a ~5° rotation. Combined with `NLAClient` renormalizing all injections to
  `inj_scale=150` (direction-only sensitivity), identical output at α=2 was
  **geometrically guaranteed**, not evidence the direction carries no signal.

### TF-IDF re-verification (on `results/nla_descriptions.csv`, tier 3)

| Classifier | AUC |
|---|---|
| TF-IDF on descriptions, leaked vs not-leaked | 0.603 |
| TF-IDF on descriptions, tier_3 vs tier_1/2 | 1.000 |

Confirms session 04's finding: descriptions carry the (trivial) task-format signal
essentially perfectly but carry almost nothing about the leak decision — consistent with
the activation-level numbers above (1.00 vs 0.68 → 1.00 vs 0.60).

## Reframe of the research question

Old framing (session 04): "Wang et al. proved the signal is there with near-perfect
accuracy; why can't our NLA read it?"

New framing: **"Decodability ≠ verbalizability, and we can now quantify both."** Two
ladders:
- **Format/tier signal:** activation-probe AUC 1.00 → survives into NLA descriptions
  (TF-IDF AUC 1.00). Fully verbalizable.
- **Leak-decision signal:** activation-probe AUC 0.68 → does not survive into
  descriptions (TF-IDF AUC 0.60, zero spontaneous privacy-violation language across all
  496 descriptions). The NLA behaves like a lossy compressor that transmits the dominant
  variance directions and drops weak task-relevant ones.

This is a real, quantified characterization of a brand-new method's limits (NLA paper is
from earlier in 2026), published with good timing, and doesn't require rescuing the
original (overclaimed) hypothesis.

## Open item to check before trusting future SGLang/NLA runs

Both `results/diff_means_output.txt` and `results/counterfactual_output.txt` print:

```
[NLAClient] actor_hf: d_model=3584 inj_scale=150.0 embed_scale=1.00 inj_char='㈎'(id=149705)
```

Session 04 says the injection token ID mismatch was fixed via `sed` in
`actor_hf/nla_meta.yaml` (149705 → 149785), but both logged runs still show **149705** —
the value session 04 said was wrong. Descriptions clearly track scenario content
correctly (so injection is *working*), but the discrepancy between the session-04 fix
and the printed value should be reconciled — `grep -i inj actor_hf/nla_meta.yaml` on
Lambda, 30 seconds — before trusting injection-scale-sensitive results in future runs.

## Concrete next steps (priority order)

1. **Free, do first:** logit-lens the diff vector through Qwen's unembedding (one matmul
   on Lambda) — check what tokens it's closest to. Also resolve the injection-token
   discrepancy above.
2. **~1 GPU-hour — highest immediate value:** **Teacher-forced position sweep / decision
   crystallization curve.** All 270 tier-3 responses are already in
   `results/benchmark_results_bf16.csv`. Feed `prompt + stored_response` through Qwen in
   one forward pass per scenario (no generation needed), hook all positions in layer 20,
   and run the CV probe at each response-token index. Plot AUC vs position. Hypothesis:
   AUC climbs steeply over the first ~10 generated tokens as the leak decision
   crystallizes. Then re-run NLA verbalization at the position with highest AUC instead
   of the last prompt token.
3. **One Lambda session — single highest-ROI experiment:** **Minimal pairs / paired
   counterfactuals.** Current diff-of-means contrasts *different scenarios* (topic
   confounds dominate, residual is mostly noise → explains the 0.68 ceiling). Instead:
   for each of the 270 tier-3 scenarios, use GPT-4o-mini (~$1 total) to generate a
   counterfactual where the secret is already public knowledge (e.g. "Anthony was the
   only one who knew" → "everyone at the office knew"), changing nothing else. Extract
   layer-20 activations for both versions at the same position. Compute
   `v_privacy = mean(secret_acts - public_acts)` over the 270 *paired* differences —
   topic cancels exactly this time, so sampling noise should collapse and this should
   probe near-ceiling (this is the actual Wang-et-al.-style norm direction). Then:
   (a) verbalize counterfactual-shifted activations along `v_privacy` — NLA's best shot;
   (b) project each *original* tier-3 activation onto `v_privacy` and test whether weak
   privacy-norm encoding predicts leaking — this becomes the non-circular axis for the
   2×2 (replaces the dead "NLA CI Signal present/absent" axis from the original design).
4. **Same/next session:** α-sweep (α ∈ {5, 10, 20, 40}) on the *denoised* diff direction
   (project onto top ~50 PCs of the activation cloud first to remove off-manifold noise),
   verbalizing at each step to find where descriptions transition from
   format-dominated to gibberish — characterizes the "readable window," if any.
5. **Local, free, no GPU:** the "what survives verbalization" figure — for several
   attributes (tier, leak/not-leaked, refusal, topic cluster, response length, etc.),
   plot activation-probe AUC vs description-TF-IDF AUC on one chart. This is the
   conceptual figure for the paper and costs nothing — can be built today from existing
   `.npz`/`.csv` files.
6. **If time allows — outside-the-box additions for a stronger workshop paper:**
   - **Patchscopes** (Ghandeharioun et al. 2024) as a *targeted* readout: patch the
     activation into an interpretation prompt of our choosing ("Does this situation
     involve a secret that should not be shared? Answer:") and read Yes/No logits.
     Question-conditioned, no trained checkpoint, works at any layer. A
     probe-vs-Patchscopes-vs-NLA comparison on the same activations is itself a
     "readout ladder" result.
   - **Steering as a semantics test:** add ±α·v_privacy at layer 20 during generation on
     tier 3, re-judge with GPT-4o-mini, measure leak-rate shift vs α. If leak rate moves
     monotonically, the direction is *causally* implicated even if no verbalizer can read
     it — "causally potent but unverbalizable" is a stronger and more memorable claim
     than "unverbalizable" alone.
   - **Introspection baseline:** ask Qwen directly, after each response, "did your reply
     reveal private information?" Compare accuracy to the activation probe. Gives a
     three-layer story: what the model *encodes* (probe) vs what it *says about itself*
     (introspection) vs what verbalizers *recover* (NLA/Patchscopes), against what it
     actually *does* (leak labels).

## Pipeline status (updated)

| Step | File | Status |
|------|------|--------|
| 1 | `notebooks/01_setup.ipynb` | Done |
| 2 | `notebooks/02_benchmark.ipynb` | Done (NF4, deprecated) |
| 2b | `scripts/benchmark.py` | Done (bf16) |
| 3 | `scripts/extract_activations.py` | Done |
| 4 | `scripts/run_nla.py` | Done — negative result on per-scenario, now explained (L8) |
| 4b | `scripts/diff_of_means.py` | Done — diff is real but weak (AUC 0.68), now explained (L8) |
| 5 | `scripts/probe_diagnostics.py` | **Done — local, no GPU needed** (this session) |
| 6 | Position sweep (teacher-forced) | **Next** |
| 7 | Minimal-pairs / `v_privacy` | Pending, depends on step 6 |
| 8 | "What survives verbalization" figure | Pending — can be done locally any time |
| 9 | `notebooks/07_analysis.ipynb` (2×2, restructured around `v_privacy` projection) | Pending |

## Run instructions

`scripts/probe_diagnostics.py` runs locally — no Lambda needed:

```bash
cd ~/nla-ci
pip3 install scikit-learn
python3 scripts/probe_diagnostics.py
```

For the position sweep and minimal-pairs experiments (steps 6–7 above), Lambda A100 is
needed (model forward passes). No SGLang/NLA server needed for the position sweep itself
— only for the final verbalization step.
