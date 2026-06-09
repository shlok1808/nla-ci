# Session 04 — 2026-06-09

## What we did

- Ran `scripts/run_nla.py` on Lambda A100: fed all 496 layer 20 activations through `kitft/nla-qwen2.5-7b-L20-av` via SGLang
- Fixed three blocking issues mid-run:
  1. `kernels` package conflict with transformers — fixed with `pip uninstall kernels`
  2. Injection token ID mismatch: `nla_meta.yaml` had `149705` but current tokenizer maps `㈎` to `149785` — fixed with `sed` in `actor_hf/nla_meta.yaml`
  3. Symlink needed for `actor_hf` path resolution (`ln -s ~/actor_hf ~/nla-ci/actor_hf`)
- Output: `results/nla_descriptions.csv` — 496 rows, 0 missing, 0 empty

## Key finding: NLA descriptions are structure-focused, not CI-focused

Reading the descriptions, **all 496 look essentially the same** regardless of label:

> *"Structured conversation format with chat reactions around a topic... Final token 'response\n' opens a direct quote block, immediately requiring a spoken reply..."*

Both leaked and appropriate scenarios produce descriptions about:
- The chat/dialogue format
- Character names and surface topic
- What the next token should be ("Final token X ends mid-sentence, immediately requiring...")

**No systematic difference between leaked and appropriate groups.** The NLA is reading the model's "predict next token" state, not its CI reasoning state. The CI signal Wang et al. found via linear probes is present geometrically but swamped by the much louder "I am about to generate dialogue" signal in the raw per-scenario activations.

This is a meaningful negative result, not a pipeline failure.

## Why this happened (hypothesis)

The extraction point (last token of prompt, pre-generation) captures the model's state right before it decides what to say. At that position, layer 20 is dominated by syntax/format prediction — "what kind of text comes next" — rather than semantic reasoning about CI. The CI signal is a small perturbation on top of this dominant format signal.

## Related: Francesco Zaffino's LessWrong post

While debugging, found [SAE-it Across Models (Francesco Zaffino)](https://www.lesswrong.com/posts/AtbZQuAn2iY2jCup2/sae-it-across-models-explaining-features-with-foreign-nla) — posted same day, uses Qwen2.5-7B layer 20 + NLA AVs. Key relevant finding: they feed SAE **feature directions** (decoder vectors) to the NLA, not raw per-token activations. They also use **background washout** — subtracting a mean background vector before feeding to NLA to help it focus on the signal. This is methodological validation for Idea 1 below.

## Ideas for next steps

**Idea 1 — Difference of means (doing next):**
- Compute `leaked_mean - appropriate_mean` across all tier 3 scenarios
- This cancels out common noise (format, character names, dialogue structure) and leaves only the systematic signal separating leaked from appropriate
- Feed that single difference vector to the NLA verbalizer
- If CI signal is real and human-readable, should come back as privacy/disclosure language
- Low cost: one numpy operation + one NLA call, no new model runs needed
- Methodologically supported by Francesco's approach (feeding directions, not raw activations)

**Idea 2 — Mid-generation extraction (deferred):**
- Extract activations during generation (first few output tokens) rather than at last prompt token
- Catches the model "in the act" of deciding to disclose/deflect
- More expensive: requires re-running all 496 scenarios with output token hooks
- Posted in mech interp Slack to get community input before deciding

## Fable notes (for review)

Anthropic released Claude Fable 5 (first public Mythos-class model) today — planning to use it to review methodology. Things to ask Fable:

1. **Is the extraction point wrong?** Should we extract at a different layer or token position to capture CI reasoning rather than next-token prediction?
2. **Is difference-of-means the right operation?** Or should we use something like the top principal component of the leaked-appropriate separation?
3. **Is the negative NLA result itself publishable?** "Per-scenario NLA descriptions at last prompt token don't distinguish CI violations" — is that a meaningful finding or just a null result?
4. **Is the 2×2 matrix still viable?** If per-scenario NLA descriptions are uninformative, what's the right way to construct the CI signal present/absent axis?
5. **Background washout:** Should we subtract a baseline activation vector before feeding to the NLA, similar to Francesco's approach?

## diff_of_means.py results (run complete)

Full output in `results/diff_means_output.txt`.

**Mean vectors → sensible but structure-focused** (same pattern as per-scenario):
- `leaked_mean`: *"Structured fiction format... friend's feelings about something negative... supportive reply around friendship"*
- `appropriate_mean`: *"Narrative format with a conversation loop... one friend expressing worry... work stress"*
- `not_leaked_mean`: *"Structured dialogue format... coworker's recent breakup... compassionate or supportive words"*

No CI language in any mean vector description.

**Difference vectors → injection failure (Chinese/math gibberish)**
Both `diff_leaked_vs_appropriate` and `diff_leaked_vs_not_leaked` produced Chinese text + math competition content — the NLA injection failure mode documented in nla-inference: *"if injection fails, the actor verbalizes something Chinese from free-association."*

**Why:** The difference vector has norm ~4.3, far smaller than natural activations. Most of the 3584 dimensions cancel in the subtraction — the result is a sparse, tiny, out-of-distribution vector the NLA has never seen. Scaling it to injection_scale amplifies noise, not signal.

**Fix: counterfactual interpolation**
Instead of feeding the raw diff, feed `not_leaked_mean + alpha * diff` — starts from a natural activation and pushes it along the leaked direction. Keeps the vector in distribution while amplifying the signal:

```python
counterfactual_leaked = not_leaked_mean + 2 * diff_leaked_vs_not_leaked
```

## What's next

1. Fix `diff_of_means.py` — add counterfactual interpolation vectors, verbalize those
2. Use Fable to review full methodology before building analysis
3. Based on Fable feedback: decide on scoring approach

## Run instructions (Lambda, if needed)

SGLang server needs to be running for the NLA call:
```bash
ssh -i ~/.ssh/lambda_final ubuntu@<ip>
cd nla-ci && git pull
tmux new -s sglang
python -m sglang.launch_server \
    --model-path ./actor_hf \
    --disable-radix-cache \
    --mem-fraction-static 0.85 \
    --trust-remote-code \
    --port 30000
# Ctrl+B D, then:
tmux new -s diff
python scripts/diff_of_means.py
```

## Pipeline status

| Step | File | Status |
|------|------|--------|
| 1 | `notebooks/01_setup.ipynb` | ✓ Done |
| 2 | `notebooks/02_benchmark.ipynb` | ✓ Done (NF4, deprecated) |
| 2b | `scripts/benchmark.py` | ✓ Done (bf16) |
| 3 | `scripts/extract_activations.py` | ✓ Done |
| 4 | `scripts/run_nla.py` | ✓ Done — negative result on per-scenario |
| 4b | `scripts/diff_of_means.py` | Next |
| 5 | `notebooks/05_scoring.ipynb` | Pending |
| 6 | `notebooks/06_probe.ipynb` | Pending |
| 7 | `notebooks/07_analysis.ipynb` | Pending |
