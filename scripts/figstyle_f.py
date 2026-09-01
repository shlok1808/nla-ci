"""figstyle_f.py — shared figure style for paper artifacts.

The palette, rcParams, hanley_se, ref_lines and titles below are copied VERBATIM
from scripts/make_figures_f.py (session 13) so that new figures match the four
already-committed ones. make_figures_f.py is deliberately NOT refactored to
import this module: editing it would risk perturbing committed output for no
benefit. scripts/validate_paper_step01_f.py AST-compares the two files and fails
if the palette or rcParams drift apart.

Palette (validated colourblind-safe, all six checks pass — blue/orange/violet,
worst adjacent CVD dE 24.7 protan / 33.6 normal, all >= 3:1 contrast on white):
    leak       #2a78d6  blue
    deflection #eb6834  orange
    privacy    #4a3aa7  violet
Colour follows the ENTITY across every figure.

In the step-01 package the same three colours carry CONSTRUCT FAMILY, never
significance or verdict:
    C_LEAK  blue   -> disclosure presence  (substantive_leak, broad_breach,
                                            leak_vs_appropriate)
    C_DEFL  orange -> response strategy    (limiting_vs_direct,
                                            limiting_among_disclosers)
    C_PRIV  violet -> disclosure degree    (degree_boundary_broadonly_vs_leaked)
"""

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Style (verbatim from make_figures_f.py) ───────────────────────────────────

C_LEAK, C_DEFL, C_PRIV = '#2a78d6', '#eb6834', '#4a3aa7'
C_INK, C_MUTED, C_GRID = '#0b0b0b', '#52514e', '#d9d8d4'

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

# Deterministic vector output: no CreationDate/Producer drift between builds, and
# real text (Type 42) rather than outlines so the PDF stays editable/searchable.
plt.rcParams.update({
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'pdf.compression': 6, 'svg.hashsalt': 'nla-ci-step01',
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


# ── Step-01 additions ─────────────────────────────────────────────────────────

FAMILY_COLOR = {
    'disclosure_presence': C_LEAK,
    'response_strategy': C_DEFL,
    'disclosure_degree': C_PRIV,
}


def save_all(fig, stem):
    """Write pdf + svg + png for one figure stem (a pathlib.Path without suffix).

    Metadata is emptied so repeated builds hash identically.
    """
    meta_pdf = {'Creator': None, 'Producer': None, 'CreationDate': None}
    fig.savefig(f'{stem}.pdf', metadata=meta_pdf)
    fig.savefig(f'{stem}.svg', metadata={'Date': None})
    fig.savefig(f'{stem}.png', dpi=300)
    plt.close(fig)
