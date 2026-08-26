# References

Every external work this project cites, in one place, grouped by the role it
plays in the paper. Previously these were scattered across `REPORT.md`,
`docs/METHODOLOGY_f.md`, and `docs/LIMITATIONS.md`; this file is the index.

**Written 2026-08-26.** Add new entries here, with the role they play — not just
the citation.

---

## 1. The work we position against

**Wang et al. (2026), "Do LLMs Know What Is Private Internally?"**
[arXiv:2604.00209](https://arxiv.org/abs/2604.00209)
The closest prior work and the paper we must differentiate from. Probes CI
*attributes* (information type, recipient, transmission principle) as linearly
separable directions in Qwen2.5-7B, in a **judgment frame**; reports
near-perfect concept probes coexisting with ~38.5% leak rates, and steers along
CI-parametric directions to reduce violations. **Their abstract already claims
the population-level headline** ("privacy failures arise from misalignment
between representation and behavior rather than missing awareness").

*What they do not have, and what we lead with:* behaviour-outcome probes at the
moment of action in an **acting frame**, positional/crystallization dynamics,
any verbalization channel, and the deflection/leak dissociation.

**Nissenbaum (2004), Contextual Integrity.** The normative framework. Quoted
directly inside the judge system prompt (`scripts/benchmark.py`
`JUDGE_SYSTEM`), so it is load-bearing for the labels, not only for framing.

---

## 2. The NLA method itself

**Natural Language Autoencoders** (Anthropic, transformer-circuits 2026)
[transformer-circuits.pub/2026/nla](https://transformer-circuits.pub/2026/nla/)
The method. An activation verbalizer (AV) maps an activation to text; an
activation reconstructor (AR) maps text back to an activation; the pair is
jointly trained with RL to reconstruct residual-stream activations. Our
checkpoints are `kitft/nla-qwen2.5-7b-L20-av` and `-ar`. Their showcase result
is surfacing unverbalized **evaluation awareness** during a Claude Opus 4.6
pre-deployment audit — which is why our E-EVAL control exists.

**"Train the Model, Not the Reader: Decodability Supervision for Verifiable
Activation Explanations" (RECAP)**
[arXiv:2607.20379](https://arxiv.org/html/2607.20379v1) — **added 2026-08-26**
The most important methodological citation for our limitations. Two findings
bear directly on us:
1. **Reconstruction measures sufficiency, not faithfulness.** *"It does not
   penalize false additions, because the objective has no preference among claim
   values that induce the same reconstruction."* Empirically, on released
   systems, explanations reached r̃ = 0.84 reconstruction while only **~2% of
   specific claims were reconstruction-dependent.**
2. **Co-trained verbalizer/reconstructor pairs develop "private codes"** — false
   wording the reconstructor depends on — in **5 of 5 controlled runs**,
   detectable only by swapping in an independent reconstructor.

Their positive proposal (probe-based monitoring; *"the probe, not the free-form
explanation, is the reliable readout"*) matches the logic of our design, but
RECAP itself **cannot be applied to us**: *"RECAP must be co-trained in and
cannot be retrofitted onto a frozen model."* See **L17**.

**"Do Activation Verbalization Methods Convey Privileged Information?"**
[arXiv:2509.13316](https://arxiv.org/abs/2509.13316)
Tests Patchscopes / LatentQA / SelfIE on QA and retrieval. Documents the
**confabulation** failure mode: the verbalizer paraphrases the prompt or the
chain-of-thought instead of reading internals — worthless as a safety signal,
because you already had both. Also raises that the verbalizer is itself an LLM
with world knowledge, so a plausible description may reflect the *verbalizer's*
priors rather than the *target's* state. Motivates the `informed_p1`
(topic-only) control in `scripts/blinded_reader_f.py`. See **L17**.

**Universal Activation Verbalizer**
[arXiv:2605.25903](https://arxiv.org/pdf/2605.25903)
Explains activations from heterogeneous donor models with a shared decoder.
Relevant to the cross-family concern behind our reader-model choice.

---

## 3. Adjacent readout methods (the ladder we position within)

**Patchscopes** — Ghandeharioun et al. 2024,
[arXiv:2401.06102](https://arxiv.org/abs/2401.06102). Patch a stored activation
into an interpretation prompt of your choosing and read out an answer.
Scaffolded here as `scripts/patchscopes_f.py`.

**SelfIE** — Chen et al. 2024,
[arXiv:2403.10949](https://arxiv.org/abs/2403.10949). Self-interpretation of
embeddings; same family as Patchscopes.

**Activation Oracles** — [arXiv:2512.15674](https://arxiv.org/abs/2512.15674).
LatentQA-style general activation QA; matches white-box baselines.

**Predictive Concept Decoders** —
[arXiv:2512.15712](https://arxiv.org/abs/2512.15712).

**Cross-layer attention probing** —
[arXiv:2509.09700](https://arxiv.org/abs/2509.09700).

**Self-explanations predicting behaviour** —
[arXiv:2602.02639](https://arxiv.org/abs/2602.02639). Directly adjacent to our
verbalization-survival triad.

**Tuned lens** — Belrose et al. 2023. The principled version of the logit lens,
which failed for us at layer 20 of 28 (**L10**) because of the rotation gap. Not
pursued: too much machinery for a side figure.

---

## 4. Refusal / deflection representation (our positive control's neighbourhood)

**Arditi et al. 2024, refusal direction** —
[arXiv:2406.11717](https://arxiv.org/abs/2406.11717). The canonical
single-direction refusal result.

**Zhao et al. 2025, harmfulness ≠ refusal** —
[arXiv:2507.11878](https://arxiv.org/abs/2507.11878). Distinct encodings at
distinct positions. The direct analogue of our deflection/leak dissociation, and
the closest conceptual precedent for our headline.

**Refusal cones / multiple directions** —
[arXiv:2502.17420](https://arxiv.org/abs/2502.17420),
[arXiv:2602.02132](https://arxiv.org/abs/2602.02132). Refusal is not
one-dimensional. Relevant to why our E-EVAL erasure tests a rank-3 subspace and
not only a single direction.

**"Refusal Falls off a Cliff"** —
[arXiv:2510.06036](https://arxiv.org/abs/2510.06036). Token-position
refusal-score tracking during generation in reasoning models. **Closest existing
machinery to our position sweep**, asking a different question — cite it when
describing the sweep so a reviewer does not think we missed it.

*Our wedge against all of the above:* CI-preserving **deflection** (in-character
topic-steering with **zero** hard refusals, **L5**) is a distinct, socially
natural behaviour, not the refusal these papers study.

---

## 5. Probe methodology

**LEACE** — [arXiv:2306.03819](https://arxiv.org/abs/2306.03819). Concept
erasure. Preferred over naive direction-subtraction if erasure claims ever
become load-bearing.

**"Linear probes rely on textual evidence"** —
[arXiv:2509.21344](https://arxiv.org/abs/2509.21344). Behaviour-probe AUROC
drops 10–30 points when textual evidence is filtered (sandbagging 0.94 → 0.57).
**Independent motivation for our input-text baseline in the triad** — cite it as
the control this line of work demands, not as a threat.

**Hanley & McNeil (1982).** AUC standard errors. Used throughout
(`results/stats_hardening_hanley_f.csv`).

---

## 6. Eval awareness (the E-EVAL confound literature)

**Probing and steering evaluation awareness** —
[arXiv:2507.01786](https://arxiv.org/abs/2507.01786). Eval awareness is linearly
represented and steerable — the reason E-EVAL had to exist.

**Lindsey 2025** (concept-injection introspection); **Needham et al. 2025**.
Framing literature for "models contain decision-relevant state they do not
verbalize."

*2026 critiques worth citing beside our E-EVAL null:* content-agnostic injection
detection, and format-sensitivity of eval-awareness probes. Our manipulation is
**explicit**, so our null bounds only the explicit variety of the confound.
