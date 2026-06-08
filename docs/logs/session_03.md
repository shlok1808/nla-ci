# Session 03 — 2026-06-08

## What we did

- Identified precision mismatch: original benchmark ran Qwen2.5-7B in 4-bit NF4 quantization. NLA checkpoints (`kitft/nla-qwen2.5-7b-L20-av`) were trained on non-quantized activations — extracting in bf16 from NF4 labels is a methodological mismatch. Quantization can shift safety/alignment behavior, so some leaked/not-leaked labels may not reflect how the bf16 model behaves.
- Updated `scripts/benchmark.py`: removed `BitsAndBytesConfig`, switched to `torch_dtype=torch.bfloat16`, output now goes to `results/benchmark_results_bf16.csv` (NF4 results preserved).
- Created `scripts/extract_activations.py`: forward hook on `model.model.layers[19]` (layer 20, 1-indexed), captures last-token residual stream, saves `results/activations_layer20.npz` with crash-resume and incremental saves every 50 scenarios.
- Decided against notebooks for pipeline steps — scripts run headless in tmux, no kernel state issues, easier to monitor on Lambda.
- Created `LIMITATIONS.md` — running doc of methodological caveats, cross-linked to relevant files.

## Key decisions

- **bf16 re-run:** Re-running the full benchmark so labels match the model we extract activations from. Labels from the NF4 run are kept in `benchmark_results.csv` but treated as deprecated for NLA pipeline purposes.
- **Scripts over notebooks:** `extract_activations.py` lives in `scripts/` not `notebooks/`. All future pipeline steps that are one-shot and non-interactive (NLA feed-forward, scoring) should also be scripts. Notebooks are for exploration and analysis only.
- **Extraction point confirmed:** Last token of the chat-templated prompt (`add_generation_prompt=True`) — position right before generation begins. Consistent with Wang et al. and confirmed via Neuronpedia session (Llama 70B, layer 53).

## bf16 Benchmark Results (run complete)

| Tier | n | leaked | refused | appropriate |
|------|---|--------|---------|-------------|
| tier_1 | 10 | 0% | 0% | 100% |
| tier_2a | 98 | 0% | 0% | 100% |
| tier_2b | 98 | 0% | 0% | 100% |
| tier_3 | 270 | 55.9% | 13.3% | 30.7% |
| tier_4 | 20 | 35.0% | 0% | 65.0% |

**Tier 3 leak rate: 55.9%** — up from NF4 50%, further from Wang et al. 38.5% (+17.4pp). Opposite of what we expected. bf16 model leaks *more* than NF4, suggesting quantization had a mild safety-dampening effect. Accepted — delta is still explained by judge strictness on implicit disclosures (see L2, updated). The 55.9% rate gives a slightly unbalanced but still usable contrast set (~151 leaked / ~119 not-leaked).

**Tier 4: 35% leaked** — down from 50% in NF4. Small n (20), treat as exploratory (see L3).

**Hook bug fixed mid-session:** `extract_activations.py` assumed `(batch, seq, hidden)` but Qwen2 decoder layer returned `(seq, hidden)` with batch squeezed. Fixed to handle both cases (`h[0,-1,:]` if 3D else `h[-1,:]`). See commit after `10b5c1f`.

## Activation Extraction Results (run complete)

- Output: `results/activations_layer20.npz`
- Shape: `(496, 3584)` ✓
- Dtype: float32 ✓
- NaN: False, Inf: False ✓
- Tier distribution: 10 + 98 + 98 + 270 + 20 = 496 ✓
- Labels: 302 appropriate, 158 leaked, 36 refused (matches benchmark CSV)

## What's next

1. Build `scripts/run_nla.py` — load `kitft/nla-qwen2.5-7b-L20-av`, feed activations → NL descriptions, save to `results/nla_descriptions.csv`
2. Manual step: read 20–30 descriptions, define CI-relevance coding guide before scoring anything
3. `scripts/score_nla.py` — embed descriptions with `all-MiniLM-L6-v2`, cosine similarity to CI reference phrases + negative controls

## Run instructions (Lambda A100 80GB)

```bash
ssh ubuntu@<ip>
cd nla-ci
git pull
pip install transformers torch accelerate tqdm openai pandas numpy
export OPENAI_API_KEY=sk-...
tmux new -s benchmark
python scripts/benchmark.py   # Part 1: ~1hr
# after it finishes:
python scripts/extract_activations.py   # Part 2: ~15min
```

## Notebook sequence (updated)

- `01_setup.ipynb` — install packages, download ConfAIde data, parse all tiers ✓
- `02_benchmark.ipynb` — Qwen inference + GPT-4o-mini judge → `results/benchmark_results.csv` ✓ (NF4, deprecated for NLA pipeline)
- `scripts/benchmark.py` — same as above but bf16 → `results/benchmark_results_bf16.csv` (this session)
- `scripts/extract_activations.py` — layer 20 residual stream at last token → `results/activations_layer20.npz` (this session)
- `04_nla.ipynb` — load NLA AV, feed activations → NL descriptions
- Manual step: read 20–30 descriptions, define CI-relevance coding guide
- `05_scoring.ipynb` — embed descriptions with `all-MiniLM-L6-v2`, cosine similarity to CI reference phrases + negative controls
- `06_probe.ipynb` — linear probe (logistic regression on raw activations), corroborating method only
- `07_analysis.ipynb` — 2x2 matrix (ConfAIde result × NLA CI Signal), tier breakdown (2x2x4), stats
