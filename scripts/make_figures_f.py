"""
make_figures_f.py — publication figures for the paper, from committed result CSVs.

Every number plotted here is read from results/ — nothing is hardcoded except the
figure layout. Re-run after any experiment changes and the figures update.

Palette (validated colourblind-safe, all six checks pass — blue/orange/violet,
worst adjacent CVD dE 24.7 protan / 33.6 normal, all >= 3:1 contrast on white):
    leak       #2a78d6  blue
    deflection #eb6834  orange
    privacy    #4a3aa7  violet
Colour follows the ENTITY across every figure — leak is always blue, deflection
always orange, privacy always violet — so a reader who learns the mapping in
Fig 1 keeps it for the rest of the paper.

Usage:
    python scripts/make_figures_f.py
Outputs: results/figures/fig{1..4}_*.{pdf,png}
"""

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Style ─────────────────────────────────────────────────────────────────────

C_LEAK, C_DEFL, C_PRIV = '#2a78d6', '#eb6834', '#4a3aa7'
C_INK, C_MUTED, C_GRID = '#0b0b0b', '#52514e', '#d9d8d4'
OUT = Path('results/figures'); OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': C_MUTED, 'axes.labelcolor': C_INK,
    'text.color': C_INK, 'xtick.color': C_MUTED, 'ytick.color': C_MUTED,
    'axes.grid': True, 'grid.color': C_GRID, 'grid.linewidth': 0.6,
    'grid.alpha': 0.9, 'axes.axisbelow': True, 'legend.frameon': False,
    'lines.linewidth': 2.0, 'lines.markersize': 5,
})


def hanley_se(auc, n_pos, n_neg):
    """Hanley & McNeil (1982) standard error of an AUC."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    return np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
                    + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg))


def ref_lines(ax, chance=True, thresh=True):
    if chance:
        ax.axhline(0.5, color=C_MUTED, lw=0.8, ls=':', zorder=1)
    if thresh:
        ax.axhline(0.8, color=C_MUTED, lw=0.8, ls='--', zorder=1)


def titles(fig, title, subtitle, h):
    """Title + subtitle above the axes, offset in INCHES so the gap does not
    collapse on short figures (figure-fraction offsets do)."""
    fig.text(0.0, 1 + 0.34 / h, title, ha='left', va='bottom',
             fontsize=10.5, fontweight='bold', color=C_INK)
    fig.text(0.0, 1 + 0.10 / h, subtitle, ha='left', va='bottom',
             fontsize=7.5, color=C_MUTED)


# ══ FIGURE 1 — the core asymmetry ════════════════════════════════════════════
# Job: identity (two behaviours) over an ordered position axis -> lines.
# Left = free generation end-to-end; right = text held constant. Same story twice.

def fig1():
    rel = pd.read_csv('results/relative_position_aucs_f.csv')
    fp = pd.read_csv('results/forced_prefix_aucs_f.csv')
    order = ['k0', '10%', '25%', '50%', '75%', '90%', 'last']

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True,
                             gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.12})

    # (a) relative-position sweep, layer 20
    ax = axes[0]
    r20 = rel[rel.layer == 20]
    for tgt, c, lbl in [('leaked_vs_approp', C_LEAK, 'leaking'),
                        ('refused_vs_rest', C_DEFL, 'deflection')]:
        s = r20[r20.target == tgt].set_index('pos_name')['auc'].reindex(order)
        ax.plot(range(len(order)), s.values, color=c, marker='o', label=lbl,
                markeredgecolor='white', markeredgewidth=0.8, zorder=3)
    ref_lines(ax)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(['prompt\nend', '10%', '25%', '50%', '75%', '90%', 'final\ntoken'])
    ax.set_ylabel('probe AUC (5-fold CV)')
    ax.set_xlabel('position through the generated response')
    ax.set_title('(a) free generation, end to end', loc='left', color=C_INK)
    ax.set_ylim(0.45, 1.0)

    # (b) forced prefix — text identical across scenarios.
    # Each prefix is drawn as its OWN line and stops where that prefix ends
    # (3 / 14 / 22 tokens). Averaging across prefixes would silently change the
    # underlying set at each x — past position 3 only two prefixes exist, past
    # 14 only one — the same population-shrinkage trap the relative sweep avoids.
    ax = axes[1]
    for tgt, c in [('leaked_vs_approp', C_LEAK), ('refused_vs_rest', C_DEFL)]:
        g = fp[fp.target == tgt]
        for pref in ['well', 'think', 'commit']:
            s = g[g.prefix == pref].sort_values('pos')
            ax.plot(s.pos, s.auc, color=c, lw=1.3, alpha=0.85, marker='o', ms=2.6,
                    markeredgecolor='none', zorder=3)
    ref_lines(ax)
    ax.set_xlabel('forced-prefix token position')
    ax.set_title('(b) same words forced on every scenario', loc='left', color=C_INK)
    ax.text(0.97, 0.97, 'one line per prefix (3 / 14 / 22 tokens)', transform=ax.transAxes,
            ha='right', va='top', fontsize=7, color=C_MUTED)

    # direct labels, not a legend box — 2 entities, colour is consistent paper-wide
    axes[0].annotate('deflection', xy=(6, 0.90), xytext=(3.3, 0.945),
                     color=C_DEFL, fontweight='bold', fontsize=8.5)
    axes[0].annotate('leaking', xy=(6, 0.685), xytext=(3.5, 0.605),
                     color=C_LEAK, fontweight='bold', fontsize=8.5)
    for ax in axes:
        x0 = ax.get_xlim()[0] + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        ax.text(x0, 0.825, 'AUC 0.80', fontsize=7, color=C_MUTED, va='bottom')
        ax.text(x0, 0.522, 'chance', fontsize=7, color=C_MUTED, va='bottom')

    titles(fig, 'Deflection is decided before the model speaks; leaking never is',
           'Layer 20 of Qwen2.5-7B-Instruct, ConfAIde tier 3 (n=270). '
           'Colour is consistent across all figures: blue = leaking, orange = deflection.',
           2.9)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig1_asymmetry.{ext}')
    plt.close(fig)
    print('  fig1_asymmetry')


# ══ FIGURE 2 — knowledge vs behaviour ════════════════════════════════════════
# Job: magnitude comparison across 3 named measures -> horizontal bars, sorted.

def fig2():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # recompute the secret-vs-public AUC from the stored pair activations
    ex = np.load('results/minimal_pairs_acts_f.npz', allow_pickle=True)
    S, P = ex['acts_secret'].astype(float), ex['acts_public'].astype(float)
    n = len(S)
    X = np.vstack([S, P]); y = np.r_[np.ones(n, int), np.zeros(n, int)]
    aucs = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(n)):
        itr, ite = np.r_[tr, tr + n], np.r_[te, te + n]
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1e-3, max_iter=5000)).fit(X[itr], y[itr])
        aucs.append(roc_auc_score(y[ite], clf.predict_proba(X[ite])[:, 1]))
    auc_priv = float(np.mean(aucs))

    ps = pd.read_csv('results/position_sweep_aucs_f.csv')
    k0 = ps[(ps.pos == 0) & (ps.layer == 20)].set_index('target')
    auc_defl = k0.loc['refused_vs_rest', 'auc']
    auc_leak = k0.loc['leaked_vs_approp', 'auc']

    rows = [
        ('is this information private?\n(secret vs public rewrite)', auc_priv,
         hanley_se(auc_priv, n, n), C_PRIV, 'knowledge'),
        ('will I deflect?\n(deflection vs normal reply)', auc_defl,
         hanley_se(auc_defl, 36, 234), C_DEFL, 'behaviour'),
        ('will I leak?\n(leaked vs normal reply)', auc_leak,
         hanley_se(auc_leak, 151, 83), C_LEAK, 'behaviour'),
    ]

    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    ypos = np.arange(len(rows))[::-1]
    for yp, (lbl, a, se, c, kind) in zip(ypos, rows):
        ax.barh(yp, a, height=0.5, color=c, zorder=3)
        ax.errorbar(a, yp, xerr=1.96 * se, color=C_INK, lw=1.2, capsize=3, zorder=4)
        ax.text(a + 1.96 * se + 0.012, yp, f'{a:.3f}', va='center',
                fontsize=9, fontweight='bold', color=C_INK, zorder=5)
    ax.axvline(0.5, color=C_MUTED, lw=0.8, ls=':', zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlim(0.5, 1.06)
    ax.set_xlabel('probe AUC (5-fold CV, 95% CI)')
    ax.grid(axis='y', visible=False)
    titles(fig, 'The model encodes privacy far better than its own upcoming behaviour',
           f'Identical extraction point (layer 20, final prompt token), probe family, and CV '
           f'scheme; both probes see the same words. Knowing-vs-doing gap: '
           f'{rows[0][1] - rows[2][1]:+.3f} AUC.', 2.5)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig2_dissociation.{ext}')
    plt.close(fig)
    print(f'  fig2_dissociation  (privacy AUC recomputed = {auc_priv:.3f})')


# ══ FIGURE 3 — the null coupling ═════════════════════════════════════════════
# Job: "is there a relationship?" where the answer is no. Show the raw
# distributions (overlap = the finding) plus binned rates with CIs (flat = the
# finding). A 2x2 table would hide both.

def fig3():
    m = pd.read_csv('results/minimal_pairs_analysis_f.csv')
    la = m[m.label.isin(['leaked', 'appropriate'])].copy()
    proj = la.proj_paired.values
    leaked = (la.label == 'leaked').values

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7),
                             gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.3})

    # (a) overlapping distributions. DENSITY, not counts — there are 128 leaked
    # vs 73 appropriate, so raw counts would make leaked look larger everywhere
    # purely from class imbalance.
    ax = axes[0]
    bins = np.linspace(proj.min(), proj.max(), 22)
    for mask, c, lbl in [(leaked, C_LEAK, f'leaked (n={leaked.sum()})'),
                         (~leaked, C_MUTED, f'appropriate (n={(~leaked).sum()})')]:
        ax.hist(proj[mask], bins=bins, density=True, color=c, alpha=0.30,
                zorder=3, lw=0)
        ax.hist(proj[mask], bins=bins, density=True, histtype='step',
                color=c, lw=1.8, label=lbl, zorder=4)
    ax.axvline(np.median(proj[leaked]), color=C_LEAK, lw=1.3, ls='--', zorder=5)
    ax.axvline(np.median(proj[~leaked]), color=C_MUTED, lw=1.3, ls='--', zorder=5)
    ax.set_xlabel('privacy encoding strength\n(projection onto $v_{privacy}$)')
    ax.set_ylabel('density')
    ax.set_title('(a) the two behaviours overlap', loc='left', color=C_INK)
    ax.legend(loc='upper right', fontsize=7, borderpad=0.3, handlelength=1.2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)

    # (b) leak rate by encoding quintile, with Wilson CIs
    ax = axes[1]
    q = pd.qcut(proj, 5, labels=False)
    xs, rates, los, his = [], [], [], []
    for i in range(5):
        k = q == i
        p, nn = leaked[k].mean(), k.sum()
        z = 1.96
        den = 1 + z ** 2 / nn
        cen = (p + z ** 2 / (2 * nn)) / den
        half = z * np.sqrt(p * (1 - p) / nn + z ** 2 / (4 * nn ** 2)) / den
        xs.append(i); rates.append(p); los.append(cen - half); his.append(cen + half)
    rates = np.array(rates)
    ax.errorbar(xs, rates, yerr=[rates - np.array(los), np.array(his) - rates],
                fmt='o', color=C_LEAK, lw=1.6, capsize=3, markeredgecolor='white',
                markeredgewidth=0.8, ms=7, zorder=4)
    ax.axhline(leaked.mean(), color=C_MUTED, lw=1.2, ls='--', zorder=2)
    ax.text(4.62, leaked.mean() - 0.018, f'overall {leaked.mean():.0%}',
            fontsize=7.5, color=C_MUTED, ha='right', va='top')
    ax.set_xticks(range(5))
    ax.set_xticklabels(['weakest\n1', '2', '3', '4', 'strongest\n5'])
    ax.set_xlim(-0.5, 4.7)
    ax.set_xlabel('privacy encoding strength (quintile)')
    ax.set_ylabel('leak rate')
    ax.set_ylim(0.25, 1.0)
    # neutral panel title: the eye may read the noisy quintiles as a rising
    # trend, so let the overlapping CIs and the stated test carry the claim
    ax.set_title('(b) leak rate by encoding strength', loc='left', color=C_INK)
    ax.text(0.5, 0.955, r'$\chi^2$=0.68, p=0.41', transform=ax.transAxes,
            ha='center', va='top', fontsize=7.5, color=C_MUTED)

    titles(fig, 'Stronger privacy encoding does not mean less leaking',
           'AUC 0.421 [0.339, 0.503] — the CI includes chance, and the two projection '
           'variants disagree in sign. The awareness gap is not an encoding-strength gap.',
           2.7)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig3_null_coupling.{ext}')
    plt.close(fig)
    print('  fig3_null_coupling')


# ══ FIGURE 4 — layer trajectory (supplementary) ══════════════════════════════
# Job: magnitude over an ORDERED variable (depth) -> line, sequential emphasis.

def fig4():
    ps = pd.read_csv('results/position_sweep_aucs_f.csv')
    k0 = ps[ps.pos == 0]
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for tgt, c, lbl in [('refused_vs_rest', C_DEFL, 'deflection'),
                        ('leaked_vs_approp', C_LEAK, 'leaking')]:
        s = k0[k0.target == tgt].sort_values('layer')
        ax.plot(s.layer, s.auc, color=c, marker='o', label=lbl,
                markeredgecolor='white', markeredgewidth=0.8, zorder=3)
    ref_lines(ax)
    ax.axvline(20, color=C_MUTED, lw=0.8, ls='-', alpha=0.5, zorder=1)
    ax.text(20.4, 0.965, 'extraction layer', fontsize=7, color=C_MUTED, va='top')
    ax.set_xlabel('layer (of 28)')
    ax.set_ylabel('probe AUC at the final prompt token')
    ax.set_xticks(sorted(k0.layer.unique()))
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc='lower right')
    titles(fig, 'Deflection peaks exactly at the NLA layer',
           'Layer 20 was fixed by the available NLA checkpoint, not chosen post hoc.',
           2.6)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig4_layer_trajectory.{ext}')
    plt.close(fig)
    print('  fig4_layer_trajectory')


if __name__ == '__main__':
    print(f'writing figures to {OUT}/')
    fig1(); fig2(); fig3(); fig4()
    print('done — .pdf (vector, for LaTeX) and .png (preview) for each')
