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
- Run `notebooks/setup.ipynb` and verify data loads correctly
- Pull NLA code from transformer-circuits release
- Start manual labeling of eval awareness on 20–30 scenarios
