"""
analyze_position_sweep_f.py — Post-hoc analysis of the position sweep.

Consumes results/position_sweep_aucs_f.csv (the committed AUC grid) and, where the
raw sweep activations are unavailable locally (position_sweep_acts_f.npz is
gitignored / Lambda-only), falls back to results/activations_layer20.npz for the
pos=0 / layer-20 checks (causal masking => the sweep's pos-0 last-prompt-token state
equals the npz extraction point).

Produces:
  results/position_sweep_heatmap_f.png       (task 1)
  results/position_sweep_trajectory_f.png    (task 2)
  results/position_sweep_pos0_depth_f.png    (task 5)
  results/position_sweep_threeway_f.png      (task 6, pos 0)
and prints a structured numeric report for tasks 3-7.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

RES = Path('results')
CSV = RES / 'position_sweep_aucs_f.csv'
NPZ = RES / 'activations_layer20.npz'
LAYERS = [10, 15, 20, 24, 28]
TARGETS = ['leaked_vs_not', 'leaked_vs_approp', 'refused_vs_rest']
TARGET_LABEL = {
    'leaked_vs_not':    'Leaked vs Not-leaked  (151 vs 119)',
    'leaked_vs_approp': 'Leaked vs Appropriate  (151 vs 83)',
    'refused_vs_rest':  'Refused/Deflection vs Rest  (36 vs 234)',
}

df = pd.read_csv(CSV)
POS = sorted(df['pos'].unique())          # 0..64
NPOS = len(POS)

def grid(target):
    """layer x pos AUC matrix, rows ordered by LAYERS, cols by POS."""
    sub = df[df.target == target]
    g = sub.pivot(index='layer', columns='pos', values='auc').reindex(LAYERS)
    return g[POS].values, g

def hdr(t):
    print('\n' + '=' * 78)
    print(t)
    print('=' * 78)

# ════════════════════════════════════════════════════════════════════════════
# TASK 1 — heatmaps (one per target, stacked), diverging cmap centered at 0.5
# ════════════════════════════════════════════════════════════════════════════
hdr('TASK 1 — heatmaps')
gmin = df.auc.min(); gmax = df.auc.max()
print(f'global AUC range across whole sweep: {gmin:.4f} .. {gmax:.4f}')
norm = TwoSlopeNorm(vcenter=0.5, vmin=min(gmin, 0.45), vmax=gmax)
fig, axes = plt.subplots(3, 1, figsize=(13, 7.2), constrained_layout=True)
ims = None
for ax, tgt in zip(axes, TARGETS):
    M, _ = grid(tgt)
    ims = ax.imshow(M, aspect='auto', cmap='RdBu_r', norm=norm,
                    extent=[POS[0]-0.5, POS[-1]+0.5, len(LAYERS)-0.5, -0.5])
    ax.set_yticks(range(len(LAYERS)))
    ax.set_yticklabels([f'L{l}' for l in LAYERS])
    ax.set_ylabel('layer')
    ax.set_title(TARGET_LABEL[tgt], fontsize=10, loc='left')
    ax.set_xticks(range(0, NPOS, 5))
    ax.axvline(42, color='k', lw=0.7, ls=':', alpha=0.6)
axes[-1].set_xlabel('response-token position k   (k=0 = last prompt token;  dotted line = k=42)')
cb = fig.colorbar(ims, ax=axes, shrink=0.85, pad=0.015)
cb.set_label('5-fold CV probe AUC  (white = 0.5 chance)')
fig.suptitle('Position sweep — behaviour decodability across layer × generation position', fontsize=12)
out = RES / 'position_sweep_heatmap_f.png'
fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
print(f'saved {out}')

# ════════════════════════════════════════════════════════════════════════════
# TASK 2 — trajectory plots (AUC vs pos, 5 layer lines, one panel per target)
# ════════════════════════════════════════════════════════════════════════════
hdr('TASK 2 — trajectories')
colors = plt.cm.viridis(np.linspace(0, 0.92, len(LAYERS)))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for ax, tgt in zip(axes, TARGETS):
    sub = df[df.target == tgt]
    for c, l in zip(colors, LAYERS):
        s = sub[sub.layer == l].sort_values('pos')
        ax.plot(s.pos, s.auc, color=c, lw=1.3, marker='.', ms=3, label=f'L{l}')
    ax.axhline(0.5, color='gray', lw=0.6, ls='--')
    ax.axvline(42, color='k', lw=0.6, ls=':', alpha=0.5)
    ax.set_title(TARGET_LABEL[tgt], fontsize=9.5)
    ax.set_xlabel('response-token position k')
    ax.grid(alpha=0.2)
axes[0].set_ylabel('5-fold CV probe AUC')
axes[0].legend(title='layer', fontsize=8, ncol=2)
fig.suptitle('Decision crystallization curves — AUC vs generation position', fontsize=12)
fig.tight_layout()
out = RES / 'position_sweep_trajectory_f.png'
fig.savefig(out, dpi=200); plt.close(fig)
print(f'saved {out}')

# ════════════════════════════════════════════════════════════════════════════
# TASK 3 — verify the "pos 42 peak" claim for leaked_vs_approp
# ════════════════════════════════════════════════════════════════════════════
hdr('TASK 3 — pos-42 peak claim (leaked_vs_approp)')
sub = df[df.target == 'leaked_vs_approp']
for l in LAYERS:
    s = sub[sub.layer == l].set_index('pos')['auc']
    argmax_pos = int(s.idxmax()); peak = s.max()
    # neighbourhood
    nb = {p: (s[p] if p in s.index else np.nan) for p in [40, 41, 42, 43, 44]}
    # plateau stats in window 38..46
    win = s.loc[38:46]
    print(f'\nL{l:>2}: argmax k={argmax_pos} AUC={peak:.4f}   | '
          f'pos42={s.get(42, np.nan):.4f}')
    print(f'      neighbours  40:{nb[40]:.4f} 41:{nb[41]:.4f} 42:{nb[42]:.4f} '
          f'43:{nb[43]:.4f} 44:{nb[44]:.4f}')
    print(f'      window[38..46]: mean={win.mean():.4f} std={win.std():.4f} '
          f'range={win.max()-win.min():.4f}  (42 - max(41,43)={s.get(42,np.nan)-max(nb[41],nb[43]):+.4f})')
# Is pos 42 a sharp peak or a plateau? quantify across the 4 layers where it "peaks"
print('\nPeak-sharpness summary (Δ = pos42 AUC minus mean of its two neighbours):')
for l in LAYERS:
    s = sub[sub.layer == l].set_index('pos')['auc']
    d = s.get(42, np.nan) - np.nanmean([s.get(41, np.nan), s.get(43, np.nan)])
    # how many positions in 30..55 are within 0.01 of the peak?
    win = s.loc[30:55]; near = (win >= win.max() - 0.01).sum()
    print(f'  L{l:>2}: Δ(42 vs nbrs)={d:+.4f}   #positions within 0.01 of peak in [30..55] = {near}')

# ════════════════════════════════════════════════════════════════════════════
# TASK 4 — deflection flatness (refused_vs_rest, layer 20)
# ════════════════════════════════════════════════════════════════════════════
hdr('TASK 4 — deflection flatness (refused_vs_rest, L20)')
s20 = df[(df.target == 'refused_vs_rest') & (df.layer == 20)].set_index('pos')['auc'].sort_index()
bins = {'early 0-9': s20.loc[0:9], 'mid 10-39': s20.loc[10:39], 'late 40-64': s20.loc[40:64]}
for name, b in bins.items():
    print(f'  {name:11s}: mean={b.mean():.4f}  std={b.std():.4f}  min={b.min():.4f}  max={b.max():.4f}  n={len(b)}')
print(f'  ALL 0-64    : mean={s20.mean():.4f}  std={s20.std():.4f}  '
      f'min={s20.min():.4f} (k={int(s20.idxmin())})  max={s20.max():.4f} (k={int(s20.idxmax())})  '
      f'range={s20.max()-s20.min():.4f}  CV={s20.std()/s20.mean():.4f}')
# slope test: linear fit AUC ~ pos
m, b0 = np.polyfit(s20.index.values, s20.values, 1)
print(f'  linear trend slope = {m:+.6f} AUC/token  (over 64 tokens: {m*64:+.4f})  intercept={b0:.4f}')
# compare to leak trend at L20 for contrast
sl = df[(df.target == 'leaked_vs_not') & (df.layer == 20)].set_index('pos')['auc'].sort_index()
ml, _ = np.polyfit(sl.index.values, sl.values, 1)
print(f'  [contrast] leaked_vs_not L20 slope = {ml:+.6f}/token (over 64: {ml*64:+.4f}), '
      f'pos0={sl.loc[0]:.4f} -> late mean={sl.loc[40:64].mean():.4f}')

# ════════════════════════════════════════════════════════════════════════════
# TASK 5 — layer-10 vs layer-20 gap for deflection at pos 0 (all targets)
# ════════════════════════════════════════════════════════════════════════════
hdr('TASK 5 — pos-0 AUC across depth (all targets)')
pos0 = df[df.pos == 0].pivot(index='layer', columns='target', values='auc').reindex(LAYERS)[TARGETS]
print(pos0.round(4).to_string())
print('\nconsecutive-layer deltas (is the jump gradual or sudden?):')
for tgt in TARGETS:
    vals = pos0[tgt].values
    deltas = np.diff(vals)
    seg = '  '.join(f'L{LAYERS[i]}->L{LAYERS[i+1]}:{deltas[i]:+.4f}' for i in range(len(deltas)))
    print(f'  {tgt:17s} {seg}')
fig, ax = plt.subplots(figsize=(7.2, 5))
markers = {'leaked_vs_not': 'o', 'leaked_vs_approp': 's', 'refused_vs_rest': '^'}
for tgt in TARGETS:
    ax.plot(LAYERS, pos0[tgt].values, marker=markers[tgt], lw=1.8, ms=8, label=TARGET_LABEL[tgt])
ax.axhline(0.5, color='gray', ls='--', lw=0.7)
ax.set_xlabel('layer'); ax.set_ylabel('5-fold CV probe AUC at k=0 (last prompt token)')
ax.set_xticks(LAYERS); ax.set_title('Where each behavioural signal forms across depth (pos 0)')
ax.legend(fontsize=8); ax.grid(alpha=0.25)
out = RES / 'position_sweep_pos0_depth_f.png'
fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
print(f'saved {out}')

# ════════════════════════════════════════════════════════════════════════════
# TASK 6 + 7 — recompute from npz (pos 0 == sweep's pos 0 by causal masking)
# ════════════════════════════════════════════════════════════════════════════
hdr('TASK 7 — reproduce sweep pos-0/L20 AUCs from activations_layer20.npz')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

d = np.load(NPZ, allow_pickle=True)
m3 = d['tiers'] == 'tier_3'
X3 = d['activations'][m3].astype(np.float32)          # (270, 3584) fp32
lab3 = d['labels'][m3]
print(f'tier-3 npz: {X3.shape}  labels={pd.Series(lab3).value_counts().to_dict()}')
# magnitude context for the 0.75 delta question
absX = np.abs(X3)
print(f'activation magnitude: max|x|={absX.max():.2f}  p99.9={np.percentile(absX,99.9):.2f}  '
      f'p99={np.percentile(absX,99):.2f}  median|x|={np.median(absX):.3f}  '
      f'mean L2 norm={np.linalg.norm(X3,axis=1).mean():.1f}')

def probe(X, y):
    """Exact replica of position_sweep_f.probe_curves config."""
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1e-3, max_iter=5000))
    s = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
    return roc_auc_score(y, s)

def Xf16(X):  # mimic sweep's float16 storage round-trip
    return X.astype(np.float16).astype(np.float32)

targets_pos0 = {
    'leaked_vs_not':    (np.ones(len(lab3), bool),  (lab3 == 'leaked').astype(int)),
    'leaked_vs_approp': (lab3 != 'refused',          (lab3 == 'leaked').astype(int)),
    'refused_vs_rest':  (np.ones(len(lab3), bool),  (lab3 == 'refused').astype(int)),
}
csv_pos0 = df[(df.pos == 0) & (df.layer == 20)].set_index('target')['auc']
print('\n  target            sweep_CSV   npz_fp32   npz_fp16   |Δ(fp32)|')
for name, (mask, y) in targets_pos0.items():
    a32 = probe(X3[mask], y[mask]); a16 = probe(Xf16(X3)[mask], y[mask])
    cv = csv_pos0[name]
    print(f'  {name:17s} {cv:.4f}     {a32:.4f}     {a16:.4f}     {abs(cv-a32):.4f}')

hdr('TASK 6 — three-way pairwise separation at L20')
groups = {'leaked': lab3 == 'leaked', 'appropriate': lab3 == 'appropriate', 'refused': lab3 == 'refused'}
print(f'group sizes: ' + ', '.join(f'{k}={v.sum()}' for k, v in groups.items()))
pairs = [('leaked', 'appropriate'), ('leaked', 'refused'), ('refused', 'appropriate')]
print('\nPOS 0  (computed from npz; positive class = first group):')
pair_aucs_pos0 = {}
for a, b in pairs:
    mask = groups[a] | groups[b]
    y = groups[a][mask].astype(int)
    auc = probe(X3[mask], y)
    pair_aucs_pos0[(a, b)] = auc
    print(f'  {a:11s} vs {b:11s}: AUC={auc:.4f}   (n={mask.sum()})')
print('\nPOS 42 (raw sweep acts are Lambda-only; only leaked-vs-appropriate is in the CSV):')
p42 = df[(df.pos == 42) & (df.layer == 20)].set_index('target')['auc']
print(f'  leaked      vs appropriate: AUC={p42["leaked_vs_approp"]:.4f}   (from CSV leaked_vs_approp)')
print(f'  leaked      vs refused    : NOT COMPUTABLE locally (needs position_sweep_acts_f.npz)')
print(f'  refused     vs appropriate: NOT COMPUTABLE locally (needs position_sweep_acts_f.npz)')

# three-way bar chart at pos 0
fig, ax = plt.subplots(figsize=(6.4, 4.4))
names = [f'{a}\nvs {b}' for a, b in pairs]
vals = [pair_aucs_pos0[p] for p in pairs]
bars = ax.bar(names, vals, color=['#c44', '#48a', '#4a4'])
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f'{v:.3f}', ha='center', fontsize=10)
ax.axhline(0.5, color='gray', ls='--', lw=0.8)
ax.set_ylim(0.45, 1.0); ax.set_ylabel('5-fold CV probe AUC')
ax.set_title('Three-way pairwise separability at layer 20, pos 0')
out = RES / 'position_sweep_threeway_f.png'
fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
print(f'saved {out}')

hdr('DONE — all figures + numeric checks complete')
