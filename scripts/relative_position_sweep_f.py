"""
relative_position_sweep_f.py — E2b. Close the L15 window hole.

position_sweep_f.py (E1) probes ABSOLUTE response positions k=0..64. But tier-3
responses have median 111 tokens (max 277) and 100% exceed 64, so E1 sees barely
half of a typical reply. The headline "leak decodability never exceeds 0.77" is
therefore only supported *within the first 64 response tokens* (L15) — and the
disclosure the judge actually cites can occur anywhere, including after the
window closes.

Fix: sample at RELATIVE positions — {10, 25, 50, 75, 90}% of each response, plus
the final response token — so coverage is end-to-end regardless of length. Every
scenario contributes at every relative position, so the probed population is
constant across the curve (unlike an absolute sweep extended to k=277, where
late positions would be a shrinking, length-biased subsample of long responses).

k=0 (last prompt token) is carried over unchanged as the pre-generation baseline
and is directly comparable to E1 and to activations_layer20.npz (L8's 0.68).

Pre-registered prediction + decision rules: docs/PREREGISTRATION.md §3.
  still <0.80 everywhere -> headline upgrades to "never, anywhere in the response"
                            and the L15 qualifier is retired
  crosses ~0.85 late     -> NEW POSITIVE FINDING: leak becomes readable only
                            after the disclosure is emitted (self-knowledge in
                            hindsight), not before it is committed

INHERITS E1's CONFOUND, and worse. At any k>=1 the response text already differs
between leaked and not-leaked scenarios, so a rising curve mixes decision state
with transcript-so-far. That is *more* severe here, since late relative positions
follow nearly the whole reply. Per scratch/03 (full-response text AUC 0.749 ~
the activation ceiling), late-position decodability is presumptively
transcript-reading and MUST be reported with that caveat unless forced_prefix_f.py
(E2) independently shows a text-controlled climb. This is an upper bound, not a
decision-crystallization measurement.

Usage (Lambda A100, ~40 min):
    tmux new -s relsweep
    python scripts/relative_position_sweep_f.py
    # Ctrl+B D to detach

Outputs:
    results/relative_position_acts_f.npz   (~70MB — gitignored, stays on Lambda)
    results/relative_position_aucs_f.csv   (small — commit this)
    results/relative_position_f.png
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse E1's data loading, prompt construction, and model loading verbatim so the
# two sweeps cannot drift apart. k=0 must mean the identical thing in both.
from position_sweep_f import (  # noqa: E402
    MODEL_ID, LAYERS, HIDDEN_DIM, PROBE_C,
    load_tier3, build_inputs, load_model,
)

# ── Config ────────────────────────────────────────────────────────────────────

FRACTIONS  = [0.10, 0.25, 0.50, 0.75, 0.90]   # of the response, in tokens
SAVE_EVERY = 25

# Column order in the stored array: k0, then the fractions, then the final token.
POS_NAMES = ['k0'] + [f'{int(f * 100)}%' for f in FRACTIONS] + ['last']
N_POS = len(POS_NAMES)

ACTS_PATH = Path('results/relative_position_acts_f.npz')
AUCS_PATH = Path('results/relative_position_aucs_f.csv')
PLOT_PATH = Path('results/relative_position_f.png')


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_relative(model, prompt_ids, resp_ids):
    """
    One forward over prompt+response; returns (n_layers, N_POS, dim) float16.

    Column 0 is the last prompt token (pre-generation, == E1's k=0). Columns
    1..len(FRACTIONS) are the response tokens at each fraction. The final column
    is the last response token. No NaN padding is needed: every response has at
    least one token, so every column is always defined.
    """
    import torch
    captured = {}

    def make_hook(i):
        def hook(module, inp, out):
            h = out[0]
            captured[i] = (h[0] if h.dim() == 3 else h)   # (seq, dim)
        return hook

    handles = [model.model.layers[l].register_forward_hook(make_hook(l))
               for l in LAYERS]
    ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0).to(model.device)
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()

    p_len, r_len = len(prompt_ids), len(resp_ids)
    # absolute sequence indices for each stored column
    idxs = [p_len - 1]                                     # k0
    for f in FRACTIONS:
        j = int(round(f * (r_len - 1)))                    # response-token index
        idxs.append(p_len + j)
    idxs.append(p_len + r_len - 1)                         # last

    out = np.empty((len(LAYERS), N_POS, HIDDEN_DIM), dtype=np.float16)
    for li, l in enumerate(LAYERS):
        seq = captured[l]
        out[li] = seq[idxs].float().cpu().numpy().astype(np.float16)
    return out, r_len


# ── Probing ───────────────────────────────────────────────────────────────────

def probe_curves(acts, labels):
    """AUC per (layer, relative position) for the three standard targets."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    targets = {
        'leaked_vs_not':    (np.ones(len(labels), bool), (labels == 'leaked').astype(int)),
        'leaked_vs_approp': (labels != 'refused',        (labels == 'leaked').astype(int)),
        'refused_vs_rest':  (np.ones(len(labels), bool), (labels == 'refused').astype(int)),
    }

    rows = []
    for li in range(acts.shape[1]):
        for pi in range(acts.shape[2]):
            X = acts[:, li, pi, :].astype(np.float32)
            for name, (mask, y) in targets.items():
                yy = y[mask]
                if len(np.unique(yy)) < 2 or min(np.bincount(yy)) < 10:
                    continue
                cv = StratifiedKFold(5, shuffle=True, random_state=0)
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(C=PROBE_C, max_iter=5000))
                s = cross_val_predict(clf, X[mask], yy, cv=cv,
                                      method='predict_proba')[:, 1]
                rows.append(dict(layer=LAYERS[li] + 1, pos_name=POS_NAMES[pi],
                                 pos_idx=pi, target=name, n=int(mask.sum()),
                                 auc=roc_auc_score(yy, s)))
        print(f'  probed layer {LAYERS[li] + 1}')
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_tier3()
    print(f'tier-3 scenarios: {len(df)}')

    if ACTS_PATH.exists():
        ex = np.load(ACTS_PATH, allow_pickle=True)
        all_ids = ex['scenario_ids'].tolist()
        all_acts = list(ex['acts'])
        all_rlen = ex['resp_len'].tolist()
        print(f'resuming: {len(all_ids)} already extracted')
    else:
        all_ids, all_acts, all_rlen = [], [], []

    pending = df[~df['scenario_id'].isin(set(all_ids))]
    if len(pending):
        model, tokenizer = load_model()
        new = 0
        for _, row in tqdm(pending.iterrows(), total=len(pending), desc='Extract'):
            prompt_ids, resp_ids = build_inputs(row, tokenizer)
            if len(resp_ids) < 2:
                continue
            a, rlen = extract_relative(model, prompt_ids, resp_ids)
            all_acts.append(a)
            all_ids.append(int(row['scenario_id']))
            all_rlen.append(int(rlen))
            new += 1
            if new % SAVE_EVERY == 0:
                np.savez(ACTS_PATH, scenario_ids=np.array(all_ids),
                         acts=np.array(all_acts), resp_len=np.array(all_rlen),
                         layers=np.array(LAYERS), pos_names=np.array(POS_NAMES))
        np.savez(ACTS_PATH, scenario_ids=np.array(all_ids),
                 acts=np.array(all_acts), resp_len=np.array(all_rlen),
                 layers=np.array(LAYERS), pos_names=np.array(POS_NAMES))
    print(f'activations: {np.array(all_acts).shape}')

    # align labels to extracted IDs
    bench = pd.read_csv('results/benchmark_results_bf16.csv').set_index('scenario_id')
    labels = bench.loc[all_ids, 'label'].values
    acts = np.array(all_acts)
    rlen = np.array(all_rlen)

    print(f'response lengths: min={rlen.min()} median={int(np.median(rlen))} '
          f'max={rlen.max()}; frac >64 tokens: {100 * (rlen > 64).mean():.1f}%')

    aucs = probe_curves(acts, labels)
    aucs.to_csv(AUCS_PATH, index=False)
    print(f'wrote {AUCS_PATH}')

    # ── Decision rule (prereg §3) ────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('E2b VERDICT (docs/PREREGISTRATION.md §3)')
    print('=' * 72)
    leak = aucs[aucs['target'].isin(['leaked_vs_not', 'leaked_vs_approp'])]
    for tgt, g in leak.groupby('target'):
        piv = g.pivot_table(index='layer', columns='pos_name', values='auc')
        piv = piv[[p for p in POS_NAMES if p in piv.columns]]
        print(f'\n{tgt}:')
        print(piv.round(3).to_string())
    mx = leak['auc'].max()
    arg = leak.loc[leak['auc'].idxmax()]
    print(f'\nmax leak AUC anywhere = {mx:.4f} '
          f'(layer {int(arg["layer"])}, {arg["pos_name"]}, {arg["target"]})')
    n_cells = len(leak)
    print(f'leak cells >= 0.80: {int((leak["auc"] >= 0.80).sum())}/{n_cells}')

    if mx < 0.80:
        print('\n=> STILL <0.80 END-TO-END. L15 qualifier retires; the headline')
        print('   upgrades to "leak decodability never exceeds ~0.8 anywhere in')
        print('   the response." Report alongside the transcript caveat.')
    elif mx >= 0.85 and arg['pos_name'] in ('75%', '90%', 'last'):
        print('\n=> LATE CROSSING (>=0.85 at a late position). This is the')
        print('   pre-registered NEW POSITIVE FINDING: leak becomes readable only')
        print('   after the disclosure is emitted. Report as a change of story,')
        print('   NOT as a patch — and flag it as presumptively transcript-borne')
        print('   (scratch/03: full-response text AUC 0.749) unless E2 shows a')
        print('   text-controlled climb.')
    else:
        print('\n=> INTERMEDIATE. Report the curve; claim neither upgrade nor')
        print('   late crystallization.')

    # deflection comparison — the asymmetry is the paper's headline
    defl = aucs[aucs['target'] == 'refused_vs_rest']
    if len(defl):
        print(f'\ndeflection (refused_vs_rest): max {defl["auc"].max():.4f}, '
              f'cells >=0.80: {int((defl["auc"] >= 0.80).sum())}/{len(defl)}')
        print('  (the leak-nowhere / deflection-everywhere contrast is the headline)')

    # ── Plot ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
        x = np.arange(N_POS)
        for ax, tgt in zip(axes, ['leaked_vs_not', 'leaked_vs_approp',
                                  'refused_vs_rest']):
            g = aucs[aucs['target'] == tgt]
            for layer, gg in g.groupby('layer'):
                gg = gg.sort_values('pos_idx')
                ax.plot(gg['pos_idx'], gg['auc'], marker='o', label=f'L{layer}')
            ax.axhline(0.8, ls='--', c='r', lw=1, alpha=.7)
            ax.axhline(0.5, ls=':', c='gray', lw=1)
            ax.set_xticks(x); ax.set_xticklabels(POS_NAMES)
            ax.set_title(tgt); ax.set_xlabel('relative position in response')
            ax.grid(alpha=.3)
        axes[0].set_ylabel('probe AUC'); axes[0].legend(fontsize=8)
        fig.suptitle('E2b — relative-position sweep (end-to-end coverage; closes L15)')
        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=150)
        print(f'\nwrote {PLOT_PATH}')
    except Exception as e:
        print(f'(plot skipped: {e})')


if __name__ == '__main__':
    main()
