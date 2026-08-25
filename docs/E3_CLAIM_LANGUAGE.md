# E3 (minimal pairs) — permitted claim language

**Written 2026-08-25, before the GATE-2 validate run. Locked.**
**Read this before running `minimal_pairs_f.py --stage validate`, and again before writing the E3 paragraph.**

The purpose of this file is to fix the interpretation of E3 *in advance*, because
the confound that constrains it has already been measured (L13, `scratch/06`) and
its GATE-2 verdict is therefore known before the experiment runs. Deciding the
wording afterwards would be choosing a claim to fit a result.

---

## 1. What is already known, pre-GPU

From `scratch/06_minimal_pairs_audit.py` on the 233 valid pairs
(`results/audit/06_minimal_pairs_audit.txt`, re-verified 2026-08-25):

| Property | Value | Status |
|---|---|---|
| Qwen-token length Δ (secret − public) | +1.57 mean, +1 median | clean |
| Length-only probe AUC | 0.522 | **passes** (`LEN_AUC_FLAG` = 0.60) |
| Text AUC after removing all per-pair edit vocabulary | 0.500 | **no drift outside the edits** |
| Pair-grouped TF-IDF text AUC (secret vs public) | **0.956** | irreducible confound |
| …after ablating a 30+-term secrecy/publicity lexicon | **0.824** | survives marker removal |
| Function-word-only (markers pre-removed) | 0.758 | reaches down to prepositions |

The pairs are simultaneously **well-constructed** (length-clean, minimal, no
drift) and **irreducibly lexically confounded**. These are not in tension: the
rewrite did exactly what it was asked to do, and what it was asked to do
necessarily changes the words.

**Therefore `stage_validate` check [4] will report a privileged delta of ≈0 and
raise the CONFOUND FLAG.** That is the expected outcome. It is **not** a GATE-2
failure and must **not** trigger pair regeneration, prompt retuning, or a re-run.

---

## 2. Claim language

### ❌ Forbidden — permanently foreclosed by the design

- "The model encodes privacy **beyond** its lexical expression."
- "`v_privacy` captures privacy **semantics** rather than vocabulary."
- "Activations separate secret from public **better than** the text does."
- Any phrasing in which the *level* of the secret-vs-public activation AUC is
  itself the evidence of conceptual encoding.

No result this experiment can produce licenses these. A high activation AUC is
fully explained by a text probe at 0.956; a privileged delta near zero is the
prediction, not a disappointment.

### ✅ Permitted — the load-bearing claim

> Under an identical extraction point, probe family, and cross-validation
> scheme, the model's representation of **whether information is private**
> separates at ≈0.95, while its representation of **what it is about to do about
> it** separates at 0.65. Both probes see the same words. The finding is the
> **contrast**, not either level.

This survives the confound completely, because the confound applies to the
privacy side of the comparison and would, if anything, only *inflate* the side
that is already high — it cannot manufacture the *gap*.

### ✅ Also permitted

- "Privacy status **and its lexical expression** are jointly encoded here; this
  design cannot separate them." (L13's original wording — keep it.)
- The **non-circular 2×2**, without qualification on this axis: `v_privacy` is
  derived from secret/public rewrites and never sees a leak label, so the 2×2 is
  non-circular *regardless* of whether the axis is semantic or lexical.
- "The rewrite is minimal and length-clean (0.522 length-only AUC; 0.500 after
  edit-vocabulary removal), so the pair difference is attributable to the edited
  clauses and not to drift elsewhere in the story."

### ⚠️ Requires the §3 test before use

- "`v_privacy` generalizes across surface realizations of secrecy."
- Any claim that the direction is about privacy *as a relation* rather than
  about a particular phrasing.

---

## 3. The marker-family split (the one available partial rescue)

**Logic.** If a direction derived from one surface family of secrecy language
transfers to *held-out pairs phrased a different way*, it is not a detector for
those particular words. This is the strongest semantics-flavored evidence
obtainable inside this design. It does not defeat the confound — the families
share the general secrecy register — but it discriminates "direction for the
token *confided*" from "direction for secrecy however phrased."

**Families present in the 233 valid pairs** (regex over `story_secret`, measured
2026-08-25):

| Family | Any-match | Exactly-one-match |
|---|---|---|
| `confided` (confide/confided/confidant) | 103 | 34 |
| `secret_word` (secret/private/privately) | 80 | 24 |
| `kept_quiet` (kept it/between them/to herself) | 59 | 18 |
| `without_any` (without anyone/anyone else) | 46 | 17 |
| `only_knew` (the only one/only X knew) | 41 | 13 |
| `not_tell` (not to tell/never mention) | 7 | 0 |
| no family matched | 26 | — |

Overlap is substantial: 106 pairs match exactly one family, 76 match two, 25
match three or more.

**Pre-specified protocol — run both variants, report both:**

1. **Primary (powered, contaminated).** Assign each pair its highest-priority
   matching family (priority order: `only_knew`, `confided`, `kept_quiet`,
   `secret_word`, `without_any`, `not_tell`); n = 207 across five families of
   size 18–90. Leave-one-family-out: derive `v_privacy` on the remainder,
   report held-out secret-vs-public separation on the excluded family.
2. **Confirmatory (clean, underpowered).** Restrict to the 106 exactly-one-family
   pairs and repeat. Families are 13–34 pairs, so Hanley SE at AUC ≈ 0.9 is
   ≈0.06–0.09.

**Pre-specified reading — fixed now, not after seeing the numbers:**

- Held-out AUC **within ~0.05 of the pooled** value on the primary variant →
  report as "the direction transfers across surface families of secrecy
  language," with the confirmatory variant as support.
- **Substantial drop** (held-out near chance) on the primary variant → report as
  "`v_privacy` is family-specific," and drop every §2 ⚠️ claim.
- A null on the **confirmatory** variant alone, given n = 13–34, is
  **uninformative** — it is not evidence against transfer and must not be
  reported as such.
- The primary variant's family overlap means a positive result there is
  **suggestive, not decisive**; say so in the text.

Implement as an addition to `stage_validate` (local, free, no new GPU pass — it
reuses `results/minimal_pairs_acts_f.npz`).

---

## 4. GATE-2 decision rule

Proceed to `--stage project` (the 2×2) when:

- [ ] Length-only AUC < 0.60 — **already known to pass** (0.522)
- [ ] Held-out secret-vs-public activation AUC ≥ 0.90
- [ ] Per-pair cosine consistency ≥ 0.80 (one direction, not a topic bundle)
- [ ] Extraction cross-check: re-extracted secret activations match
      `results/activations_layer20.npz`
- [ ] CONFOUND FLAG fired **and** was expected — confirm against §1, do not re-run

The confound flag alone never blocks `project`. Only a failure of per-pair
cosine consistency (the direction is a bundle, not a direction) or of the
extraction cross-check (the pipeline is wrong) blocks it.

---

**References:** `docs/LIMITATIONS.md` L13, `REPORT.md` §2.6 and §3 T4,
`scratch/06_minimal_pairs_audit.py`, `results/audit/06_minimal_pairs_audit.txt`,
`scripts/minimal_pairs_f.py` (`stage_validate` check [4]),
`docs/PREREGISTRATION.md` §E3.
