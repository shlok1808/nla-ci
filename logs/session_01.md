# Session 01 — 2026-06-03

## What we did
- Created the project repo (`nla-ci`) with folder structure: `data/`, `notebooks/`, `src/`, `results/`, `logs/`
- Wrote the README with background on CI and NLAs, research question, full research design, methodology, expected outcomes, and licenses
- Added MIT license under Shlok's name
- Built `notebooks/setup.ipynb` — installs packages, downloads all ConfAIde benchmark files from GitHub into `data/`, parses all tiers into a DataFrame, prints one sample per tier

## Key decisions
- **Model:** Qwen2.5-7B — only model with a public NLA checkpoint
- **Layer:** Layer 20 (71% through the model) — only publicly released NLA checkpoint, framed as a feature not a limitation
- **CI theme measurement:** Embed NLA descriptions with `all-MiniLM-L6-v2`, cosine similarity to reference CI phrases; include non-CI privacy phrases as negative controls
- **Eval awareness:** Manual labeling of 20–30 scenarios first, define coding guide before labeling, then build a simple classifier
- **Corroboration:** Linear probe (logistic regression on raw activations) as a second independent method
- **Scope:** Nail the 2x2 matrix first (ConfAIde result × eval awareness), tier analysis (2x2x4) is secondary

## Key ideas to remember
- "Linearly decodable" ≠ "causally used" — don't overclaim; activation patching is future work
- Even Outcome 3 (signal + passing) doesn't imply causal use — could be downstream of same semantic understanding
- Running ConfAIde on Qwen2.5-7B is itself a novel contribution — no prior eval exists
- Eval awareness across tiers is a publishable secondary finding on its own
- Tier 1 = highest expected eval awareness (simple, textbook-looking); Tier 4 = lowest (complex multi-party)

## Next steps

**Before writing any more code:**
- Run `notebooks/setup.ipynb` and verify all tiers load correctly
- Read the ConfAIde paper/blog to understand the tier structure and what "passing" means before labeling anything
- Pull NLA weights from HuggingFace: `kitft/nla-qwen2.5-7b-L20-av` and `kitft/nla-qwen2.5-7b-L20-ar` (separate from Qwen — don't forget this)

**Notebook sequence:**
- `notebook_02_benchmark.ipynb` — run Qwen2.5-7B on all ConfAIde scenarios, collect pass/fail labels per scenario
- `notebook_03_activations.ipynb` — extract layer 20 residual stream activations at last token position for each scenario
- `notebook_04_nla.ipynb` — run NLA activation verbalizer on those activations, collect NL descriptions
- Manual step: read 20–30 NLA output descriptions, define eval awareness coding guide before labeling anything
- `notebook_05_scoring.ipynb` — embed NLA descriptions with `all-MiniLM-L6-v2`, compute CI-relevance scores via cosine similarity to reference phrases
- `notebook_06_probe.ipynb` — train linear probe (logistic regression) on raw layer 20 activations to predict CI violation as corroborating method
- `notebook_07_analysis.ipynb` — build the 2x2 matrix, visualize distributions, run statistical tests
