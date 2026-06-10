"""
logit_lens_f.py — Project every candidate direction through the unembedding.

The cheapest possible readout: apply the model's final RMSNorm + lm_head to a
direction vector and look at the top-promoted / top-suppressed vocabulary
tokens. One matmul per direction. If the behavioral leak direction or
v_privacy is privacy-flavored in vocabulary space, the top tokens say so
instantly; if they are format/topic junk, that is informative too.

Caveat (flag): logit lens at layer 20 of 28 is an approximation — directions
get rotated by the remaining 8 blocks, and RMSNorm is applied around a
direction, not a full residual state. Treat token lists as suggestive, not
dispositive (a tuned lens would be the rigorous version). The +/- asymmetry
is the more trustworthy signal: read both ends.

Directions:
  - diff_leaked_vs_not   (behavioral, tier 3)
  - refusal_dir          refused_mean - rest_mean (probes at 0.89 — strongest
                         behavioral direction we have; sanity-expect
                         deflection/hedging tokens)
  - v_privacy            if results/v_privacy_f.npz exists
  - top-3 PCs of the tier-3 cloud (context: what the DOMINANT directions say)

Usage (Lambda A100, ~5 min):
    python scripts/logit_lens_f.py

Output: results/logit_lens_f.txt
"""

import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID    = 'Qwen/Qwen2.5-7B-Instruct'
TOP_K       = 40
ACTIVATIONS = Path('results/activations_layer20.npz')
VPRIV_PATH  = Path('results/v_privacy_f.npz')
OUT_TXT     = Path('results/logit_lens_f.txt')


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    data = np.load(ACTIVATIONS, allow_pickle=True)
    t3 = data['tiers'] == 'tier_3'
    acts = data['activations'][t3]
    labels = data['labels'][t3]

    dirs = {
        'diff_leaked_vs_not': acts[labels == 'leaked'].mean(0) - acts[labels != 'leaked'].mean(0),
        'refusal_dir': acts[labels == 'refused'].mean(0) - acts[labels != 'refused'].mean(0),
    }
    if VPRIV_PATH.exists():
        dirs['v_privacy'] = np.load(VPRIV_PATH)['v_privacy']

    X = acts - acts.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    for i in range(3):
        dirs[f'pc{i + 1}'] = Vt[i]

    print(f'Loading {MODEL_ID} (bf16)...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto')
    model.eval()
    norm, head = model.model.norm, model.lm_head

    lines = []
    for name, v in dirs.items():
        vt = torch.tensor(v / np.linalg.norm(v), dtype=torch.bfloat16,
                          device=model.device)
        with torch.no_grad():
            logits = head(norm(vt.unsqueeze(0)))[0].float()
        top = torch.topk(logits, TOP_K)
        bot = torch.topk(-logits, TOP_K)
        fmt = lambda ids: ', '.join(repr(tokenizer.decode([t])) for t in ids.tolist())
        lines.append(f'\n{"=" * 70}\n[{name}]  (unit direction through RMSNorm + lm_head)')
        lines.append(f'  TOP {TOP_K} (+direction): {fmt(top.indices)}')
        lines.append(f'  TOP {TOP_K} (-direction): {fmt(bot.indices)}')
        print(lines[-3])
        print(lines[-2][:300] + ' ...')

    OUT_TXT.write_text('\n'.join(lines))
    print(f'\nSaved {OUT_TXT}')


if __name__ == '__main__':
    main()
