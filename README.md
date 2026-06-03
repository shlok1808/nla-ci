# nla-ci
can LLMs internally represent contextual integrity (CI)
violations using Natural Language Autoencoders (NLAs).

## Background

### Contextual Integrity (CI)
Contextual integrity, developed by Helen Nissenbaum, reframes privacy around appropriate information flow rather than data secrecy. Privacy is violated not when information is shared, but when it flows in ways that breach contextual norms. CI operates through five components: sender, recipient, subject, information type, and transmission principle. The same detail can be appropriate in one setting and a violation in another.

Niloofar Mireshghallah's work ([ConfAIde](https://github.com/skywalker023/confaide), [CIMemories](https://github.com/facebookresearch/CIMemories)) shows that LLMs frequently fail to respect these norms — acknowledging sensitivity yet sharing secrets anyway, or incorrectly assuming recipients already know protected details. Her blog post [*From Black and White to Gray: Redefining Privacy for Language*](https://mireshghallah.github.io/blog/contextual_integrity.html) is a good primer.

### Natural Language Autoencoders (NLAs)
NLAs are an unsupervised interpretability method that generate natural language explanations of LLM internal activations. An NLA pairs an **activation verbalizer** (maps activations → text descriptions) with an **activation reconstructor** (maps descriptions → activations back), trained jointly via RL to reconstruct residual stream activations. The resulting explanations surface what a model internally represents — including things it never says in its outputs.

Introduced in [*Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations*](https://transformer-circuits.pub/2026/nla/) (Kantamneni et al., 2026).

## research question
Do models internally represent CI violations without verbalizing them in
outputs (unverbalized cognition), or do they simply lack the representation
entirely?
