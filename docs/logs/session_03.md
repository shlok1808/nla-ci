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

## What's next

After bf16 benchmark completes:
1. Run `python scripts/extract_activations.py` to get `results/activations_layer20.npz`
2. Verify: shape `(496, 3584)`, no NaN/Inf, tier distribution `10 + 98 + 98 + 270 + 20`, all labels present
3. Compare tier 3 leak rate between NF4 (~50%) and bf16 — expect bf16 closer to Wang et al. 38.5%. >15pp divergence warrants investigation.
4. Build `scripts/run_nla.py` (or `04_nla.ipynb`) — feed activations into `kitft/nla-qwen2.5-7b-L20-av`, collect NL descriptions

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
