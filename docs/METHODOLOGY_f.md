# METHODOLOGY — nla-ci

End-to-end methodology, established findings, limitations, planned experiments, and
alternative approaches. Written session 06 (2026-06-10). Companion docs:
`docs/LIMITATIONS.md` (L1–L9, per-issue detail), `docs/logs/` (chronology),
`README.md` (public framing — currently lags this doc; update before submission).

---

## 1. Research question and how it evolved

**Phase 1 (session 01):** Do LLMs internally represent contextual-integrity (CI)
violations? → Killed by Wang et al. ([arXiv:2604.00209](https://arxiv.org/abs/2604.00209)):
CI norm parameters (information type, recipient, transmission principle) are linearly
encoded with near-ceiling probe AUROC in Qwen2.5-7B.

**Phase 2 (sessions 02–04):** Given the signal exists, what does it *say*? Verbalize
CI-relevant activations with an NLA. → Produced two null results (L6: per-scenario
descriptions are format-dominated; L7: diff vectors verbalize as gibberish).

**Phase 3 (sessions 05–06, current):** The phase-2 premise conflated two probe targets.
Wang et al.'s ceiling is for **norm attributes probed in a judgment frame** (scenario
wrapped in "is this appropriate?" templates — properties of the input). Our extraction
point is the **acting frame** (model about to produce an in-character reply), and our
target is **behavior** (will it leak). At that point leak behavior probes at only
**AUC 0.68**. The honest, sharper question:

> **Which internal signals are decodable at the moment of action, which of those are
> *privileged* (beyond input text), and which survive natural-language verbalization?**

Decodability ≠ verbalizability, quantified. CI/privacy is the case study; the
methodological claim generalizes.

---

## 2. Pipeline end to end

| # | Step | File | Output | Status |
|---|------|------|--------|--------|
| 1 | Data download/parse (ConfAIde, 5 tier files, 496 scenarios) | `notebooks/01_setup.ipynb` | `data/*.txt` | done |
| 2 | Behavior benchmark: Qwen2.5-7B-Instruct bf16, greedy, 512 tok; GPT-4o-mini judge (leaked/refused/appropriate + reasoning) | `scripts/benchmark.py` | `results/benchmark_results_bf16.csv` | done |
| 3 | Layer-20 residual extraction, last prompt token (post `add_generation_prompt`) | `scripts/extract_activations.py` | `results/activations_layer20.npz` (496×3584 fp32) | done |
| 4 | Per-scenario NLA verbalization (SGLang + `kitft/nla-qwen2.5-7b-L20-av`) | `scripts/run_nla.py` | `results/nla_descriptions.csv` | done — L6 |
| 5 | Behavioral diff-of-means + α=2 counterfactuals → NLA | `scripts/diff_of_means.py` | `results/diff_means_output.txt`, `counterfactual_output.txt` | done — L7/L8/L9 |
| 6 | Probe diagnostics (local) | `scripts/probe_diagnostics.py` | printed table (L8) | done |
| 7 | Verbalization-survival triad (local) | `scripts/verbalization_survival_f.py` | `results/verbalization_survival_f.{csv,png}` | done |
| 8 | Position sweep / crystallization curve | `scripts/position_sweep_f.py` | `position_sweep_aucs_f.csv` + plot | scaffolded |
| 9 | Forced-prefix control (text-matched) | `scripts/forced_prefix_f.py` | `forced_prefix_aucs_f.csv` + plot | scaffolded |
| 10 | Minimal pairs → v_privacy → non-circular 2×2 | `scripts/minimal_pairs_f.py` | `data/minimal_pairs_f.csv`, `results/v_privacy_f.npz`, analysis CSV | scaffolded |
| 11 | Steering (causal test) | `scripts/steering_f.py` | `steering_results_f.csv` | scaffolded |
| 12 | Patchscopes targeted readout | `scripts/patchscopes_f.py` | `patchscopes_results_f.csv` | scaffolded |
| 13 | Introspection baseline | `scripts/introspection_f.py` | `introspection_results_f.csv` | scaffolded |
| 14 | Denoised α-sweep → NLA (temp 0) | `scripts/alpha_sweep_f.py` | `alpha_sweep_f.{csv,txt}` | scaffolded |
| 15 | Logit lens on all directions | `scripts/logit_lens_f.py` | `logit_lens_f.txt` | scaffolded |

Conventions: scripts not notebooks for pipeline steps; crash-resume + incremental saves
everywhere; parsers duplicated per script ("keep in sync" comment); bf16 model, fp32
activation storage (fp16 for the large multi-position stores, which are gitignored).

**Recommended run order on Lambda** (one A100 session, items 8–15):
`logit_lens_f` (5 min) → `position_sweep_f` (~1 h) → `forced_prefix_f` (~30 min) →
`minimal_pairs_f` (needs `OPENAI_API_KEY`; ~$1 + ~30 min) → `patchscopes_f` (~30 min)
→ `introspection_f` (~15 min) → then with SGLang up: `alpha_sweep_f` (~15 min) →
finally `steering_f` (~2–4 h + ~$3, can run overnight).

---

## 3. Established findings (as of session 06)

### 3.1 Behavior benchmark (bf16)
Tier 3: 55.9% leaked / 13.3% refused / 30.7% appropriate (n=270; L2 explains the
+17.4pp vs Wang et al.'s 38.5%). Tiers 1/2: rating tasks, 100% appropriate, excluded
from behavioral analysis. Tier 4: n=20, exploratory only (L3).

### 3.2 The decodability–verbalizability–privilege triad (key numbers)

5-fold CV AUC, tier 3, layer 20, last prompt token. Input = TF-IDF on scenario text
(the "privileged information" baseline, following [arXiv:2509.13316](https://arxiv.org/abs/2509.13316));
Acts = linear probe on activations; Desc = TF-IDF on NLA descriptions.

| Attribute | Input | Acts | Desc | Privileged (Acts−Input) |
|---|---|---|---|---|
| tier (t3 vs t1/2) | 1.00 | 1.00 | 1.00 | 0 |
| topic cluster (t3)* | 1.00* | 0.85 | 0.61 | —* |
| scenario length | 0.63 | 0.92 | 0.60 | +0.29 |
| **leaked vs not** | **0.58** | **0.68** | **0.61** | **+0.10** |
| leaked vs appropriate (no refused) | 0.55 | 0.65 | 0.60 | +0.10 |
| **refused vs appropriate** | **0.62** | **0.92** | **0.76** | **+0.30** |
| response length | 0.59 | 0.72 | 0.58 | +0.13 |

*topic-cluster input baseline is circular (cluster defined on input TF-IDF).

Readings:
1. **Activations carry privileged information about upcoming behavior** — most
   dramatically for deflection (+0.30), modestly for leaking (+0.10).
2. **Verbalization collapses the leak signal back to the input floor** (0.61 ≈ 0.58):
   the NLA transmits what the text already shows and destroys the privileged part.
   Deflection partially survives (0.76 ≫ 0.62) — privileged content CAN pass through
   the channel when the underlying signal is strong.
3. **The leak/deflection dissociation**: at the moment of action the model has largely
   decided *whether to deflect* (0.92) but only weakly *whether its non-deflecting
   reply will constitute a leak* (0.65). Working hypothesis: **leakage is less a
   decision than a default** — the model fails to engage a protective policy rather
   than choosing disclosure. Direct analogue of the harmfulness-vs-refusal dissociation
   (Zhao et al. 2025, distinct encodings at distinct positions). The position sweep
   tests when (if ever) the leak outcome crystallizes.
4. NLA descriptions mention privacy terms in 21/496 cases, concentrated in tiers 1/2
   where the *prompt text* is explicitly about privacy (tier-3 hits: 2/3/2 across
   appropriate/leaked/refused — no signal). **The NLA says "privacy" when the input
   says privacy, never when the privacy reasoning is latent.**

### 3.3 Geometry and mechanism (L8/L9)
- cos(leaked_mean, not_leaked_mean) = 0.9990; ‖diff‖ = 4.0 vs within-class spread 18.5;
  diff is real (permutation p < 0.002) but ~⅓ noise energy.
- The NLA renormalizes every injected vector to L2 = 150 → **direction-only**. The
  α=2 counterfactual rotated the mean ~5°; unchanged output was guaranteed. Raw diff
  fails because its *direction* is off-manifold, not because its norm is small.
- All prior NLA calls sampled at temperature 1.0 (L9.3) — adds decode noise;
  contrastive runs must use temperature 0.

---

## 4. Limitations — current and anticipated

Current (detail in `LIMITATIONS.md`): L1 judge confidence useless; L2 leak rate
+17pp vs Wang; L3 tier-4 small/noisy; L4 NF4 deprecated; L5 `refused` =
CI-preserving deflection (collapse for binary analysis); L6 descriptions
format-dominated; L7 raw-diff gibberish (mechanism revised by L9); L8 leak weakly
decodable at extraction point; L9 direction-only injection, temp-1.0 decode noise.

Anticipated for the new experiments — write these into any paper draft:

1. **Judge noise ceiling.** ~10% label error puts an effective ceiling (~0.9) on any
   behavior-probe AUC. Affects every AUC in §3.2 equally — comparisons stand, absolute
   values are floors.
2. **Textual-divergence confound in the position sweep** (k≥1 mixes decision state
   with transcript-so-far). The forced-prefix control is the answer; report both.
3. **Minimal-pair rewrite quality.** If GPT-4o-mini also softens topic/tone,
   v_privacy is contaminated. Mitigations: similarity-ratio gate, manual spot-check of
   ~10 pairs (mandatory before trusting stage 3), and the per-pair cosine-consistency
   histogram (a multi-cluster histogram = the "direction" is actually a bundle).
4. **2×2 circularity is solved only if v_privacy never sees leak labels** — guaranteed
   by construction in `minimal_pairs_f.py` stage 3; keep it that way.
5. **Steering ≠ mediation.** A leak-rate shift shows causal *influence* of the
   direction, not that the model *uses* it in ordinary operation; incoherent-response
   shifts don't count (coherence flag in the judge). Wang et al. already show
   norm-direction steering works; our marginal claim is specifically about
   behavior-derived and minimal-pair directions in the acting frame.
6. **Patchscopes/introspection readouts are themselves model behavior** — subject to
   answer-style priors (e.g., yes-bias). The placebo question and the AUC-not-accuracy
   readout mitigate but don't eliminate.
7. **Single model, single layer-20 AV checkpoint, single benchmark.** Every claim is
   "Qwen2.5-7B layer 20 on ConfAIde tier 3" until shown otherwise. Cross-layer probes
   (position_sweep_f captures layers 10/15/20/24/28) partially address the layer part.
8. **Multiple comparisons.** We now compute many AUCs; pre-register the headline
   contrasts (leak triad, refusal triad, crystallization curve shape) and treat the
   rest as exploratory.

---

## 5. Planned experiments — design, hypotheses, decision rules

**E1. Position sweep** (`position_sweep_f.py`). Teacher-force stored responses, probe
(layer × position). H: leak AUC climbs steeply within ~10 tokens; refusal AUC starts
high (0.92) and stays. Decision rule: if leak AUC at k=10 (forced-prefix-controlled)
exceeds ~0.85, "the decision crystallizes early but after the prompt" — extraction
point was the limiting factor, and NLA verbalization should be retried at argmax-k.
If it only climbs in the uncontrolled sweep, the climb is transcript, not decision.

**E2. Forced-prefix control** (`forced_prefix_f.py`). Same probe with identical forced
text across scenarios. The (E1 − E2) gap at each k isolates transcript information
from internal-state evolution.

**E3. Minimal pairs** (`minimal_pairs_f.py`). The keystone. Produces: secret-vs-public
probe AUC at the acting-frame extraction point (H: ≥0.95 — norm encoding is strong
even when behavior encoding is weak; if NOT, the acting frame suppresses norm encoding
itself — bigger finding); per-pair consistency; cos(v_privacy, v_leak); projection→leak
AUC (H: ~0.55–0.65 — encoding strength only weakly predicts behavior, the per-scenario
awareness gap); the non-circular 2×2.

**E4. Steering** (`steering_f.py`). H (from Wang et al.): −v_privacy steering increases
leaking, +v_privacy reduces it, monotone in α within the coherent regime; behavioral
diff direction may do nothing coherent (it's mostly noise + deflection style). Either
way it pairs with the verbalization result: "causally potent but unverbalizable" or
"neither causal nor verbalizable — the leak direction is epiphenomenal."

**E5. Patchscopes** (`patchscopes_f.py`). Question-conditioned zero-training readout of
the SAME activations the NLA described. H: secret-present question beats NLA's 0.61 on
leak and approaches the probe on refusal; placebo stays ~0.5. If Patchscopes also
fails, the privileged signal is real (probe) but not *linguistically* extractable by
any current zero-shot method — strengthens the bottleneck claim beyond NLA-specific
criticism.

**E6. Introspection** (`introspection_f.py`). Plain-language self-report baseline.
Completes the ladder: probe / patchscope / NLA / self-report / input text, all on one
dataset. Also yields the three-level dissociation table (knows norm / leaked anyway /
blind to own leak).

**E7. Denoised α-sweep** (`alpha_sweep_f.py`). NLA's best shot under correct geometry:
rotations up to ~65°, PCA/bootstrap denoising, v_privacy direction, temp 0. Decision
rule: any privacy-language phase before gibberish = positive verbalization result
(report the readable window); none = the clean negative with the geometry now
unimpeachable.

**E8. Logit lens** (`logit_lens_f.py`). Free vocabulary-space read of every direction;
run first, takes 5 minutes, and the refusal direction acts as a positive control
(expect hedging/deflection tokens).

---

## 6. Ideas worth trying, not yet scripted

- **AR round-trip loss (NLA-native faithfulness metric).** Use the *reconstructor*
  (`kitft/nla-qwen2.5-7b-L20-ar`): reconstruct activation from each description, then
  measure projection of (original − reconstruction) onto v_privacy / v_leak vs random
  directions. Quantifies *which* subspaces the text channel drops — turns "the NLA
  lost it" into a measurement. Probably the single best addition to the paper's
  methods; moderate effort (AR serving mirrors AV serving).
- **NLA on the judge** (session-02 idea): swap GPT-4o-mini for an open judge (Qwen
  itself), extract judge activations at the verdict token, verbalize. The judgment
  frame should verbalize richly (norm encoding is strong there) — a positive control
  for the whole NLA pipeline that also tests the frame hypothesis.
- **SAE decomposition of v_privacy / v_leak**: express the direction in a Qwen2.5-7B
  L20 SAE basis (cf. [SAE-it Across Models](https://www.lesswrong.com/posts/AtbZQuAn2iY2jCup2/sae-it-across-models-explaining-features-with-foreign-nla)),
  verbalize top features individually with the NLA — feature directions are
  in-distribution for the AV where raw diffs are not.
- **Attention from the secret span**: at the last prompt token, measure attention mass
  to the tokens stating the secret/confidentiality clause; do leak scenarios attend
  less? Cheap addition to position_sweep_f (heads already computed).
- **Cross-position NLA**: once E1 finds argmax-k, re-run `run_nla.py` at that position
  (the AV was trained on layer-20 states generally, not only last-prompt-token — check
  the NLA paper's position distribution before claiming OOD).
- **Tier-2 norm probes**: ConfAIde tiers 1/2 ship human sensitivity/acceptability
  ratings we currently ignore; probing those at the acting vs judgment frame replicates
  Wang et al. within our exact setup.

---

## 7. Paper plan

**Working title:** "Decodable but not verbalizable: what natural-language readouts of
LLM activations miss about privacy behavior."

**Skeleton:** (1) triad figure + table (done); (2) crystallization curves E1/E2;
(3) minimal-pair 2×2 E3; (4) readout ladder E5/E6 (+E7 outcome either way);
(5) steering E4 as the causal punchline. Negative results are load-bearing throughout —
each is paired with a positive control (tier/refusal for probes, refusal direction for
logit lens, judgment frame for introspection) so "we couldn't read it" is never "our
pipeline is broken."

**Venue:** NeurIPS interpretability workshop (primary); privacy/safety workshop
(secondary — the leakage-is-a-default finding lands there). Workshop page limits fit
the triad figure + 2 experiment figures + ladder table.

**Differentiation one-liners:** vs Wang et al. — they probe/steer norm *attributes* in
a *judgment* frame; we measure *behavior* in the *acting* frame and test whether any
language channel can read it out. vs [arXiv:2509.13316](https://arxiv.org/abs/2509.13316) —
they test privileged information in verbalizations for QA/retrieval; we test it for
*behavioral self-knowledge* and add NLAs (untested there) plus causal steering.

---

## 8. Fundamentally different approaches (beyond NLA)

*Kept separate per request: if NLA had never existed, how else would you answer "do
LLMs internally represent CI violations in a human-readable way?" — each with its
application here and tradeoffs vs NLA.*

**Patchscopes** ([Ghandeharioun et al. 2024](https://arxiv.org/abs/2401.06102)) /
**SelfIE** ([Chen et al. 2024](https://arxiv.org/abs/2403.10949)). Patch the activation
into an interpretation prompt; the model itself decodes it — open-endedly ("what does
this state mean?") or as targeted QA. *Application:* E5 as scripted; also free-form
variant for qualitative reads. *Tradeoffs:* no training, any layer/position/model,
question-conditioned (finds weak attributes NLA's open-ended channel buries); but no
fidelity guarantee (the decoder model confabulates), sensitive to prompt/target-layer
choice, and the readout is itself behavior with its own biases. NLA's RL-trained
reconstruction objective is exactly an attempt to buy the fidelity Patchscopes lacks —
running both quantifies what that objective buys.

**LatentQA** (Pan et al.): fine-tune a decoder to answer questions about activations —
supervised Patchscopes. Higher fidelity, but needs training data of (activation,
QA) pairs; for CI you'd have to construct them, at which point you've nearly built a
labeled probe with a text interface.

**Sparse autoencoders + auto-interp.** Decompose layer-20 residuals into sparse
features; auto-label features; ask which fire differentially on leak vs not, or
decompose v_privacy in the feature basis. *Tradeoffs:* feature-level granularity,
in-distribution directions to verbalize, steerable handles (feature clamping); but
needs a trained SAE for this exact model/layer (training one: days of GPU + data),
feature splitting/absorption obscures single concepts, and auto-interp labels have the
same faithfulness problem one level down. If a public Qwen2.5-7B-L20 SAE exists (check
the SAE-it post's assets), §6's decomposition is cheap and worth it.

**Logit / tuned lens.** Project directions or states through (tuned) unembedding.
*Tradeoffs:* free, instant, vocabulary-grounded; but token-level not propositional,
and untuned lens at 71% depth is rough. Use as sanity layer (E8), never as the claim.

**Activation patching / causal tracing.** Swap activations between minimal-pair runs
(secret↔public twins from E3 at chosen positions/layers) and measure behavior flip;
attribution patching (gradient approximation) scales it to all positions at once.
*Application:* "which positions' states carry the secrecy information that changes the
reply?" — a causal crystallization map complementing E1's correlational one. *Tradeoffs:*
gold standard for "where is it used", no language output at all — pairs naturally with
a verbalizer pointed at the implicated sites. The minimal-pair infrastructure makes
this nearly free to add; strongest candidate for a follow-up paper.

**Probing variants.** Mass-mean / LDA probes (Marks & Tegmark: mass-mean directions
transfer better causally than logistic weights); concept erasure (LEACE) — erase
v_privacy from the stream and measure leak-rate change (erasure is the cleaner causal
test than addition: removing information vs injecting an off-manifold push); cone /
multi-direction analyses (Geometry of Refusal, [arXiv:2502.17420](https://arxiv.org/abs/2502.17420))
to test whether "privacy" is one direction or a cone. All cheap given existing data.

**Representation engineering (RepE)** (Zou et al. 2023): reading vectors from
contrastive *instruction* pairs ("think about privacy" vs neutral) rather than
stimulus pairs. Complementary direction source; worth a 30-minute comparison against
v_privacy (cosine + steering efficacy) to test whether instructed-concept and
stimulus-derived geometry agree.

**Crosscoders / transcoders + attribution graphs** (circuit tracing): full circuit
story of where secrecy information flows into the response decision. Highest insight
ceiling, highest cost; out of scope for a workshop paper, right scope for a thesis.

**Plain introspection / self-reports** (E6): the zero-interpretability baseline every
reviewer will ask for. If self-report matches the probe, interpretability adds nothing
here; expected outcome (from CoT-faithfulness literature) is that it won't — that gap
*is* the motivation for activation-level methods.

---

## 9. Related-work notes (borrow / avoid)

- **Wang et al. 2026** ([arXiv:2604.00209](https://arxiv.org/abs/2604.00209)): probes at
  last token of judgment templates; PCA directions per CI parameter at 75th-percentile
  layer; steering top-5 layers, α∈{0.5,1,2,4}; AUROC 0.53→0.90+ across depth; never
  probes actual leak behavior. *Borrow:* layer-depth sweep framing, α grid, their
  steering metrics (leakage rate, NCR, PPI) for comparability. *Avoid:* judgment-frame
  probing when claiming things about behavior.
- **Refusal direction** ([Arditi et al. 2024](https://arxiv.org/abs/2406.11717)):
  diff-in-means from contrastive *datasets*, direction selected over (layer, position)
  by validation, validated by ablation AND addition, 13 models. *Borrow:* select
  v_privacy's (layer, position) on a validation split rather than fixing layer 20 a
  priori; validate by erasure not just addition. *Follow-ups:* harmfulness vs refusal
  encoded distinctly ([Zhao et al. 2025](https://arxiv.org/abs/2507.11878)-adjacent),
  multiple refusal directions (Pan et al. 2025), refusal cones
  ([arXiv:2502.17420](https://arxiv.org/abs/2502.17420)) — expect the same richness for
  privacy; don't claim "the" privacy direction without a rank check (per-pair cosine
  histogram + PCA of paired diffs in E3 stage 3).
- **Privileged information in verbalizations** ([arXiv:2509.13316](https://arxiv.org/abs/2509.13316)):
  tests Patchscopes/LatentQA/SelfIE on QA/retrieval; finds limited privilege. *Borrow:*
  input-only baseline design (adopted in §3.2). *Gap we fill:* NLAs, behavioral
  decisions, causal layer.
- **NLA paper** ([transformer-circuits.pub/2026/nla](https://transformer-circuits.pub/2026/nla/)):
  documents posterior-collapse-style failure (plausible but information-poor text) and
  the headline use case (surfacing unverbalized evaluation awareness in the Opus 4.6
  audit). *Borrow:* their failure taxonomy when characterizing ours; check their
  training-position distribution before any "position is OOD for the AV" claim.
- **Mid-generation probing precedent**: hallucination detection from generation-time
  activations (e.g., cross-layer attention probing, [arXiv:2509.09700](https://arxiv.org/abs/2509.09700));
  self-explanations predicting behavior ([arXiv:2602.02639](https://arxiv.org/abs/2602.02639)).
  No decision-crystallization curve for a *normative* behavior found in the searches —
  E1/E2 appear novel in framing, not in machinery. Claim accordingly: the machinery is
  standard, the question and the text-matched control are the contribution.
