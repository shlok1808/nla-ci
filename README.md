# nla-ci
can LLMs internally represent contextual integrity (CI)
violations using Natural Language Autoencoders (NLAs).

## background

### contextual integrity (CI)
Contextual integrity, developed by Helen Nissenbaum, reframes privacy around appropriate information flow rather than data secrecy. Privacy is violated not when information is shared, but when it flows in ways that breach contextual norms. CI operates through five components: sender, recipient, subject, information type, and transmission principle. The same detail can be appropriate in one setting and a violation in another.

Niloofar Mireshghallah's work ([ConfAIde](https://github.com/skywalker023/confaide), [CIMemories](https://github.com/facebookresearch/CIMemories)) shows that LLMs frequently fail to respect these norms — acknowledging sensitivity yet sharing secrets anyway, or incorrectly assuming recipients already know protected details. Her blog post [*From Black and White to Gray: Redefining Privacy for Language*](https://mireshghallah.github.io/blog/contextual_integrity.html) is a good primer.

### Natural Language Autoencoders (NLAs)
NLAs are an unsupervised interpretability method that generate natural language explanations of LLM internal activations. An NLA pairs an **activation verbalizer** (maps activations → text descriptions) with an **activation reconstructor** (maps descriptions → activations back), trained jointly via RL to reconstruct residual stream activations. The resulting explanations surface what a model internally represents — including things it never says in its outputs.

Introduced in [*Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations*](https://transformer-circuits.pub/2026/nla/) (Kantamneni et al., 2026).

## research question
Wang et al. (2026) proved CI norms (information type, recipient, transmission principle) are linearly encoded in Qwen2.5-7B's activation space. We ask: what do those representations actually say? We apply Natural Language Autoencoders (NLAs) at layer 20 to produce the first human-readable characterization of unverbalized CI cognition, tested across all 4 ConfAIde tiers.

## novelty / contributions

1. **NLA verbalization.** First method to convert CI-relevant activations into human-readable descriptions. Wang et al. detect the signal exists but cannot read it. We can.

2. **All 4 ConfAIde tiers.** Wang et al. only used Tier 3. We test whether NLA descriptions of CI violations degrade with scenario complexity across all tiers.

3. **Privacy awareness gap characterization.** Wang et al. showed the gap exists numerically. We describe what the model's internal state says in scenarios where it knows but leaks anyway.

## related work

**Wang et al. (2026) — "Do LLMs Know What Is Private Internally?"** is the most directly related prior work. They show that CI norms — information type, recipient, and transmission principle — are linearly encoded as independent directions in Qwen2.5-7B's residual stream, recoverable with near-perfect probe accuracy. They also document a *privacy awareness gap*: despite encoding CI norms internally, the model leaks in 38.5% of ConfAIde Tier 3 scenarios. Their methodology is linear probes and PCA; outputs are always scalar scores or geometric directions, never natural language. They do not verbalize what the representations contain, and they test only Tier 3. We take their finding as prior validation that the signal exists and extend it by asking what the signal *says*, using NLAs to produce the first sentence-level characterization of those representations, across all four tiers.

## Research Design

**Core Question**
Wang et al. (2026) proved CI norms (information type, recipient, transmission principle) are linearly encoded in Qwen2.5-7B's activation space. We ask: what do those representations actually say? We apply NLAs at layer 20 to produce the first human-readable characterization of unverbalized CI cognition, tested across all 4 ConfAIde tiers.

**Setup**
- Run ConfAIde scenarios through Qwen2.5-7B
- Extract layer 20 residual stream activations at last token position
- Feed to NLA AV → get NL descriptions
- Analyze descriptions for CI-relevant themes


**Key Design Choices**
- Don't trust individual descriptions, trust patterns across many
- Analysis is statistical: do violating scenarios produce privacy-themed descriptions significantly more than appropriate scenarios?

**Analysis Dimensions**

| Dimension | Values | Notes |
|-----------|--------|-------|
| ConfAIde Result | Passed / Failed | Model's output judgment on the scenario |
| NLA CI Signal | Present / Absent | Privacy-themed language in NLA description |
| ConfAIde Tier | 1 / 2 / 3 / 4 | Complexity level of scenario |

**Main Matrix (2x2)**

| ConfAIde Result | NLA CI Signal | Interpretation |
|----------------|---------------|----------------|
| Passed | Absent | Genuine CI understanding |
| Passed | Present | Model knows and behaves correctly |
| Failed | Absent | Genuine CI blindspot |
| Failed | Present | Privacy awareness gap — knows but leaks anyway |

**Extended Analysis (2x2x4)**
- Add ConfAIde tier dimension (1–4)
- Hypothesis: NLA CI signal strength stays consistent across tiers even as scenario complexity increases
- If true: internal CI representations are robust to surface complexity

**Methodology**

*Positioning relative to Wang et al.*
Wang et al. (2026) established that CI norm representations exist in Qwen2.5-7B's activation space and are linearly decodable. We treat that as prior validation and do not re-litigate it. Our contribution begins where theirs ends: given that the signal exists, we use NLAs to ask what it contains. All NLA analysis in this project presupposes Wang et al.'s geometric finding as a foundation.

*Measuring CI-Relevant Themes*
- Embed each NLA description using `all-MiniLM-L6-v2`
- Embed reference CI concept phrases ("privacy violation", "inappropriate information sharing", "sensitive context", etc.)
- Include adversarial non-CI privacy phrases ("data encryption", "password security") as negative controls — ensures the score discriminates CI specifically, not just privacy-adjacent language broadly
- Compute cosine similarity → average into a single CI-relevance score per scenario
- Compare score distributions: violating vs appropriate scenarios


*Layer 20 Justification*
- Layer 20 is ~71% through Qwen2.5-7B (28 layers total)
- Hypothesis: CI violation as a semantic concept is encoded in middle-to-late layers, not output layers — but this is a hypothesis, not established
- Cheap multi-layer comparison: extract activations at layers 10, 20, 28 and feed into same NLA checkpoint (layers 10/28 are out-of-distribution for the NLA — treat as exploratory, not a test set)

*Corroborating Method: Linear Probe*
- Train logistic regression on raw layer 20 activation vectors to predict CI violation (yes/no)
- If NLA descriptions AND linear probe both distinguish violating from appropriate scenarios, that's converging evidence
- Important distinction: probe shows CI violation is *linearly decodable* from activations, not that the model *uses* that representation in its output — two different claims
- **Note:** Wang et al. (2026) already validated probe accuracy on Qwen2.5-7B (near-perfect AUROC, 38.5% leakage on Tier 3). Our probe result is a consistency check against their baseline, not a novel finding. The primary contribution is NLA verbalization.

**Expected Outcomes**

All three outcomes are publishable:

| Outcome | Description | Implication |
|---------|-------------|-------------|
| No CI signal | No internal representation of CI violations | CI failures are a knowledge/training problem — fix requires retraining |
| Signal exists, model fails | Represents the violation but doesn't act on it | Unverbalized cognition — model "knows" but doesn't "do" |
| Signal exists, model passes | Internal representation predicts correct output | CI reasoning is happening internally and surfacing correctly — but correlation only; signal and correct output may both be downstream of general semantic understanding, not one causing the other |

> We are not claiming the model causally uses these representations. We are asking whether the information is linearly decodable from activations. Even Outcome 3 does not imply causal use — only correlation. Causal intervention (activation patching) is future work.

**Limitations**
- Layer 20 only — the publicly released NLA checkpoint; not a design choice
- Can't fully solve unobservability problem
- Mitigation: show signal is systematic and predictable across scenario types

## Licenses
- ConfAIde benchmark: MIT
- Qwen2.5-7B-Instruct: Apache 2.0
- NLA checkpoints ([kitft/nla-qwen2.5-7b-L20-av/ar](https://huggingface.co/kitft)): Apache 2.0
