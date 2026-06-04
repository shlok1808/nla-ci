# Session 02 — 2026-06-04

## What happened

Found Wang et al. (2026) "Do LLMs Know What Is Private Internally?" right as the project was getting started. It answers ~75% of the original research question directly.

**What they proved:**
- CI norms (information type, recipient, transmission principle) are linearly encoded as independent directions in Qwen2.5-7B's activation space
- Near-perfect probe AUROC for recovering CI norms from residual stream
- Privacy awareness gap: model leaks in 38.5% of ConfAIde Tier 3 scenarios despite internally encoding the norms correctly
- Method: linear probes + PCA. Outputs are always scalar scores or geometric directions, never natural language
- Evaluated Tier 3 only

**What this killed:**
- Original core question ("does the representation exist?") — answered
- Linear probe as a primary contribution — Wang et al. already validated this on the exact same model/layer
- The claim that running ConfAIde on Qwen2.5-7B is novel — they did Tier 3

**What's still ours:**
- NLA verbalization: first method to convert CI-relevant activations into human-readable descriptions. They can detect the signal, we can read it.
- All 4 tiers: they only used Tier 3. We test whether NLA descriptions degrade with scenario complexity.
- Privacy awareness gap characterization: they showed the gap exists numerically. We describe what the model's internal state actually says in the leak scenarios.

## Reframing

Original question: do LLMs internally represent CI violations?
New question: Wang et al. proved they do — what do those representations actually say?

The pivot is clean. We take their finding as prior validation and extend it. NLA verbalization is the only thing their method structurally couldn't do.

## What we did

- Updated README: new research question, contributions section, related work section, demoted linear probe
- Rewrote 2x2 matrix to use NLA CI Signal (Present/Absent) instead of Eval Awareness — removed eval awareness as a primary dimension entirely
- Removed stale "eval awareness" methodology and secondary finding sections
- Renamed notebooks: `setup.ipynb` → `01_setup.ipynb`, `notebook_02_benchmark.ipynb` → `02_benchmark.ipynb`
- Built `02_benchmark.ipynb`: runs Qwen2.5-7B-Instruct (4-bit) on all ConfAIde scenarios, GPT-4o-mini judge (leaked/refused/appropriate), saves to `results/benchmark_results.csv` with crash-resume and incremental saves every 10 scenarios

## Notebook sequence (updated)

- `01_setup.ipynb` — install packages, download ConfAIde data, parse all tiers ✓
- `02_benchmark.ipynb` — Qwen inference + GPT-4o-mini judge → `results/benchmark_results.csv` ✓
- `03_activations.ipynb` — extract layer 20 residual stream activations at last token position → save per scenario
- `04_nla.ipynb` — load NLA AV (`kitft/nla-qwen2.5-7b-L20-av`), feed activations → NL descriptions
- Manual step: read 20–30 descriptions, define CI-relevance coding guide
- `05_scoring.ipynb` — embed descriptions with `all-MiniLM-L6-v2`, cosine similarity to CI reference phrases + negative controls
- `06_probe.ipynb` — linear probe (logistic regression on raw activations), corroborating method only
- `07_analysis.ipynb` — 2x2 matrix (ConfAIde result × NLA CI Signal), tier breakdown (2x2x4), stats
