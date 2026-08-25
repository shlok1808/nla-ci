# scratch/ — independent audit analyses (2026-07-02)

Local, CPU-only re-derivations written during the independent research audit
(`REPORT.md`). They read only committed artifacts (`results/*.csv`, `*.npz`,
`data/minimal_pairs_f.csv`) and **modify nothing** — safe to re-run at any time.

Captured stdout lives in `results/audit/<name>.txt`, re-run and verified
**2026-08-25** (all six exit 0; every headline number reproduces).

| Script | Establishes | Feeds |
|---|---|---|
| `01_verify_headlines.py` | Independent re-derivation of every headline AUC in the repo; CV-leakage check; per-fold spread; Hanley SEs; balanced-subsample control | `REPORT.md` §1 |
| `02_patterns.py` | (A–B) nonlinear probes do **not** beat linear on leak — null; PCA rank analysis. (C) leak/deflection anti-correlation `cos=−0.52`, deflection-erasure 0.684→0.628. (E–F) leave-one-cluster-out / leave-one-info-type-out transfer | **L12**, `REPORT.md` §2.1–2.2, §2.7 |
| `03_text_baseline_curve.py` | Text-prefix baseline curve vs activation curve (free E2 preview); full-response text ceiling 0.749; response-length distribution (median 111, 100% >64 tokens) | **L15**, `REPORT.md` §2.3–2.5 |
| `04_infotype_slices.py` | Per-info-type leak rates and AUCs; between-type variance ≈ Hanley SE → type-level AUCs are **not** reportable findings | `REPORT.md` §2.7 |
| `05_nla_descriptions.py` | Formal quantification of L6: 100% of descriptions carry format vocabulary, 21/496 privacy terms; k-means clusters track response *register*, not privacy content | L6, `REPORT.md` §2.7 |
| `06_minimal_pairs_audit.py` | Minimal-pairs confound quantification pre-GPU: length-clean (0.522), edit-minimal (0.500), but text AUC **0.956** / 0.824 marker-ablated | **L13**, **E3 claim language**, `REPORT.md` §2.6 |

## Re-running

```bash
mkdir -p results/audit
for f in scratch/0*.py; do
  python3 "$f" > "results/audit/$(basename "$f" .py).txt" 2>&1
done
```

Needs `pandas`, `numpy`, `scikit-learn`, and `transformers` (tokenizer only — no
PyTorch, no GPU). `06` downloads the Qwen2.5-7B-Instruct tokenizer on first run.

## Status

These are **audit artifacts, not pipeline scripts**. Their findings are folded into
`docs/LIMITATIONS.md` (L11–L15) and `REPORT.md`. The forward-looking analysis
notebook (`notebooks/07_analysis.ipynb`, per the `CLAUDE.local.md` pipeline table)
is still pending and belongs after `v_privacy` exists — it is the home of the 2×2,
not of these re-derivations.
