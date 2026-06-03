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
Do models internally represent CI violations without verbalizing them in
outputs (unverbalized cognition), or do they simply lack the representation
entirely?

## Research Design

**Core Question**
Do LLMs internally represent CI violations without verbalizing them (unverbalized cognition), or do they simply lack the representation entirely?

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
| Eval Awareness | Aware / Not Aware | From NLA description themes |
| ConfAIde Tier | 1 / 2 / 3 / 4 | Complexity level of scenario |
| NLA CI Signal | Present / Absent | Privacy-themed language in NLA description |

**Main Matrix (2x2)**

| ConfAIde Result | Eval Aware | Interpretation |
|----------------|------------|----------------|
| Passed | No | Genuine CI understanding (best case) |
| Passed | Yes | Deceptive alignment concern |
| Failed | No | Genuine CI blindspot |
| Failed | Yes | Can't perform even under pressure (worst case) |

**Extended Analysis (2x2x4)**
- Add ConfAIde tier dimension (1–4)
- Hypothesis: eval awareness decreases with tier complexity, CI signals stay consistent
- If true: model's internal CI representations are independent of eval detection

**Secondary Finding**
- Eval awareness itself across tiers is a publishable observation
- Tier 1 likely highest eval awareness (simple, textbook-looking)
- Tier 4 likely lowest (complex multi-party, doesn't pattern-match to benchmark)

**Methodology**

*Measuring CI-Relevant Themes*
- Embed each NLA description using `all-MiniLM-L6-v2`
- Embed reference CI concept phrases ("privacy violation", "inappropriate information sharing", "sensitive context", etc.)
- Include adversarial non-CI privacy phrases ("data encryption", "password security") as negative controls — ensures the score discriminates CI specifically, not just privacy-adjacent language broadly
- Compute cosine similarity → average into a single CI-relevance score per scenario
- Compare score distributions: violating vs appropriate scenarios

*Measuring Eval Awareness*
- Define coding guide before labeling (e.g., what in an NLA description counts as eval aware — explicit mentions of "benchmark/test/evaluation", hedging language, etc.)
- Manually label 20–30 scenarios; re-label same set a week later to check self-consistency
- Use labels to build a simple classifier or keyword list for remaining scenarios
- No immediate automation

*Layer 20 Justification*
- Layer 20 is ~71% through Qwen2.5-7B (28 layers total)
- Hypothesis: CI violation as a semantic concept is encoded in middle-to-late layers, not output layers — but this is a hypothesis, not established
- Cheap multi-layer comparison: extract activations at layers 10, 20, 28 and feed into same NLA checkpoint (layers 10/28 are out-of-distribution for the NLA — treat as exploratory, not a test set)

*Corroborating Method: Linear Probe*
- Train logistic regression on raw layer 20 activation vectors to predict CI violation (yes/no)
- If NLA descriptions AND linear probe both distinguish violating from appropriate scenarios, that's converging evidence
- Important distinction: probe shows CI violation is *linearly decodable* from activations, not that the model *uses* that representation in its output — two different claims

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
