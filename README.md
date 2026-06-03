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

**Main Matrix (2x2)**
- Passed/Failed ConfAIde × Eval Aware/Not Aware
- Cleanest signal: passed + not eval aware = genuine CI understanding
- Scariest: failed + eval aware = can't even perform under pressure
- Most interesting: passed + eval aware = deceptive alignment concern

**Extended Analysis (2x2x4)**
- Add ConfAIde tier dimension (1–4)
- Hypothesis: eval awareness decreases with tier complexity, CI signals stay consistent
- If true: model's internal CI representations are independent of eval detection

**Secondary Finding**
- Eval awareness itself across tiers is a publishable observation
- Tier 1 likely highest eval awareness (simple, textbook-looking)
- Tier 4 likely lowest (complex multi-party, doesn't pattern-match to benchmark)

**Limitations**
- Layer 20 only, no choice given compute
- Can't fully solve unobservability problem
- Mitigation: show signal is systematic and predictable across scenario types
