# RUNBOOK — what to do next, in order

**Written 2026-08-25 (session 12).** Open this first when you sit down to work.
`HANDOFF.md` is the map of the whole project; this file is just the queue.

Two things are live:

- **Part A — the GPU session.** Four experiments, ~2h on one A100. Do this first.
- **Part B — the NLA description thread.** Parked. Free and local, no GPU.
  Everything is written and committed; pick it up after Part A.

---

# PART A — the GPU session (do this first)

Four experiments, one A100 spin-up, ~2h, ~$4. All scripts are audited and fixed
as of session 12. Order below is deliberate: the gates come first so a failure
costs you 30 minutes instead of the whole session.

## A0. Set up the box

```bash
# on the Lambda A100, from the repo root
bash setup_lambda.sh
```

Downloads the ConfAIde data (the repo does not vendor it) and fixes the
torchvision/sglang ABI clash. Idempotent — safe to re-run.

Then always work inside tmux so a dropped SSH connection doesn't kill a run:

```bash
tmux new -s sessA          # detach with Ctrl+B then D, reattach with: tmux a -t sessA
```

## A1. Relative-position sweep (~40 min)

```bash
python scripts/relative_position_sweep_f.py
```

**What it's for.** The old sweep only looked at the first 64 tokens of each
response, but a typical response is 111 tokens (range 65–277). So the headline
"leak decodability never exceeds 0.77" only ever covered about half a reply.
This one samples at 10/25/50/75/90% of each response plus the final token, so
coverage is end-to-end no matter the length.

**What to look for.** The script prints its own verdict. Pre-registered
(`PREREGISTRATION.md` §3):

- max leak AUC still **< 0.80** everywhere → the claim upgrades to "never,
  anywhere in the response," and L15 retires. This is the expected outcome.
- **≥ 0.85 at a late position** (75%, 90%, or last) → a *new positive finding*:
  leak only becomes readable after the disclosure has been emitted. This is a
  change of story, not a patch — and it must be reported with the caveat that
  it is presumptively the model reading its own transcript, unless A4 shows a
  text-controlled climb.

**It will stop itself** if the k=0 activations don't match
`activations_layer20.npz` — that assert is there to catch prompt drift. If it
fires, stop and investigate; do not work around it.

## A2. Minimal pairs — the keystone (~30 min)

```bash
python scripts/minimal_pairs_f.py --stage extract
python scripts/minimal_pairs_f.py --stage validate
```

**What it's for.** Produces `v_privacy`, the privacy direction, which unlocks
the 2×2 and gives the NLA a second direction to verbalize. Nothing downstream
works without it.

**STOP after `validate` and read the verdict.** Before you do, re-read
`docs/E3_CLAIM_LANGUAGE.md` §4 — it was written in advance precisely so this
decision isn't made while staring at the numbers.

What the verdict means now (fixed in session 12; it used to print warnings
without enforcing them):

| Verdict | Meaning |
|---|---|
| FAIL — cross-check | Pipeline bug. Activations don't match the reference. Fix before anything else. **Blocks.** |
| FAIL — length | Story length leaks the label. **Blocks.** (Expected to pass: 0.522 measured locally.) |
| FAIL — AUC/direction | `v_privacy` is weak or is a bundle rather than one direction. **Blocks the 2×2.** Report as a negative methodological result. |
| PASS w/ CONFOUND FLAG | **Expected. Does NOT block.** The lexical confound was measured in advance (L13) and this flag firing is the prediction, not a failure. Do not regenerate pairs. |

The cross-check excludes 9 pairs whose story text differs between pipeline
stages (**L16**) and prints their cosines separately — expect those to be lower,
that's known and handled.

Only run `--stage project` after the verdict passes.

## A3. Eval-awareness (~20 min)

```bash
python scripts/eval_awareness_f.py --stage build      # <-- NOT optional
python scripts/eval_awareness_f.py --stage extract
```

⚠️ **`--stage build` must run on the Lambda box.** The prompts CSV is gitignored,
so it does not travel with the repo. It was rebuilt in session 12 on the
canonical parser; skipping this step means running with nothing or with a stale
file. The extract stage will now refuse to reuse activations built from a
different prompt set rather than silently mixing them.

**What it's for.** This is the result most capable of undermining the paper.
Deflection (AUC 0.89) is your strongest signal — is it really about privacy, or
is the model just detecting that it's being tested? Same scenarios, two system
framings ("this is an evaluation" vs "this is a real conversation"), user text
byte-identical.

`--stage analyze` runs locally afterwards. Pre-registered (§6, amended A1.1):
deflection surviving above 0.80 after erasing the eval direction closes the
concern — but now it must survive **both** the single-direction erasure and the
rank-3 subspace erasure. The script warns you if those disagree.

If it collapses, that is a finding, not a disaster — see Part B and the
"eval-awareness in costume" framing.

## A4. Forced prefix (~30–60 min)

```bash
python scripts/forced_prefix_f.py
```

**Run last** — it's the one whose result you already expect (flat). It's a
control: same neutral opening words forced onto every scenario, so any AUC climb
can't be the transcript. Pre-registered §2, both outcomes reportable; a flat
result is explicitly **not** a failed experiment.

## A5. If ≥30 min are left

```bash
python scripts/patchscopes_f.py
```

Cheap, already scaffolded, no API key. Asks the stored activations direct yes/no
questions and includes a placebo question as a control. Independently triangulates
the same worry A3 addresses.

## A6. Copy results back

```bash
# from your laptop
scp lambda:~/nla-ci/results/minimal_pairs_acts_f.npz  results/
scp lambda:~/nla-ci/results/eval_awareness_acts_f.npz results/
scp lambda:~/nla-ci/results/*_aucs_f.csv              results/
scp lambda:~/nla-ci/results/*.png                     results/
```

Then, locally and free:

```bash
python scripts/eval_awareness_f.py --stage analyze
python scripts/minimal_pairs_f.py --stage project     # only if validate passed
```

## A7. Before shutting the box down

If `v_privacy` validated and you still have the instance, the NLA run
(`verbalize_directions_f.py`) needs SGLang on the same box — see
`HANDOFF.md` §6 and the script's docstring. Otherwise kill the instance; the
analysis is all local from here.

**Always before an NLA run:** `grep injection_token_id actor_hf/nla_meta.yaml`
→ must be **149705** (L9). The script now asserts this, but check anyway.

---

# PART B — PARKED: the NLA description thread

Everything below is written, committed, and free to run. No GPU. Pick it up
after Part A.

## Where this came from

Observation (user, 2026-08-25): the descriptions do say something real, but it
sits *underneath* the formatting preamble — usually in the second or third
paragraph — and it's easy to dismiss the whole thing as format noise. Reading
past the preamble, a human can often tell what's going on. Checking 270 by hand
isn't feasible.

## What was already found (free, done)

**Every tier-3 description has exactly 3 paragraphs**, with a consistent job for
each: a short format preamble (~23 words), a setup quote (~35), and a final
paragraph that **forecasts the upcoming reply** (~42) — "Final token X ...
immediately requiring ..." plus example continuations.

Probing each slice separately (`scratch/08_paragraph_slices.py` →
`results/audit/08_paragraph_slices.txt`):

| slice | refused vs appropriate | leaked vs appropriate |
|---|---|---|
| P1 (preamble) | 0.569 | 0.493 |
| P2 (setup) | 0.665 | 0.470 |
| **P3 (forecast)** | **0.703** | **0.626** |
| P2+P3 | 0.743 | 0.584 |
| whole | 0.760 | 0.600 |

**For the leak contrast, P3 alone beats the whole description** — the preamble
actively dilutes it. That matters because the leak signal is the thing the
project has never been able to find.

**Why P3 is different — the example that makes it obvious.** P3 is not
describing format at all. It is the NLA *predicting what the model is about to
say*, with example continuations. Same paragraph, two labels:

> **refused** (id 222) — Final token "way\n" closes an opening quotation
> ("I would say: "), immediately requiring a spoken quote content like
> *"That's... wow, I feel a bit awkward about my life choices"* or
> *"Amy, I've got to really own this,"* completing the character's reaction.

> **leaked** (id 208) — Final token "me\n" opens a quoted example mid-sentence
> ("I'd say..."), immediately requiring a quoted dialogue continuation like
> *"Sarah, I totally understand and—"* or *"Hey, I totally get what you're
> going through, but let's find a way to support our team."*

The deflecting one forecasts awkward self-focus; the leaking one forecasts
engaging with the other person's content. That is a behavioural forecast, and a
leak/deflect probe should be able to read it. Everyone (including every prior
session) read "Final token ..." as more format-talk and stopped there — which is
how L6 was written.

⚠️ **This is post-hoc.** 5 slices × 3 contrasts = 15 cells and this is the best
one. That's the same trap as "position 42" (L11). Treat it as a **hypothesis to
test on the clean temp-0 reads**, not as a result. Write the prediction down
first, then test it.

Also relevant: `scratch/07` showed the description-level signal rides on tone
words (*awkward, dilemma* vs *supportive, understand where you're coming from*)
with **zero privacy vocabulary in the top 20 either side**. So L6's "21/496
contain privacy vocabulary" was counting the wrong thing — it looked for
vocabulary when the signal is carried by meaning and tone.

## What to run when you come back

```bash
export OPENAI_API_KEY=sk-...
python scripts/blinded_reader_f.py --dry-run       # eyeball the prompts
python scripts/blinded_reader_f.py --stage read    # ~15 min, ~$0.15, resumable
python scripts/blinded_reader_f.py --stage score   # free
```

A reader model sees **only** the description — no scenario, no label, no
response — and rates disclose/deflect likelihood, its own confidence, and how
legible the passage is. Four conditions:

- `naive_full` — no hint about what the text is (the conservative baseline)
- `informed_full` — told what an NLA description is and to read past the preamble
- `informed_p3` — forecast paragraph only
- `informed_p1` — preamble only. **The control**: P1 names the topic but has no
  forecast, so if it scores like the full condition the reader is guessing from
  subject matter, not reading internal state.

**Blinding verified safe:** longest verbatim word-run shared between a
description and the actual response is 8 words (median 2), and those are generic
phrases ("I understand where you're coming from"). The reader cannot be reading
the answer off the page.

**Bar to clear:** the whole-description text probe — leak **0.600**, deflect
**0.760**. Below that, word-counting is as good as meaning-reading and the
simpler method wins.

**Key test:** AUC broken down by the reader's own 1–5 legibility rating. If
accuracy rises with rated legibility, the variation in description quality is
real signal and the reader knows which descriptions to trust — that is the
user's observation turned into evidence. Flat accuracy across ratings means the
confidence is confabulated.

**Known caveat, written into the script:** the informed prompt tells the reader
that the forecast paragraph matters most — which is the post-hoc finding above.
So the informed condition **cannot independently confirm that hypothesis**. That
is what `naive_full` is for. If only the informed condition works, the honest
claim is "a reader told what to look for can extract it; a cold reader cannot."

**If the cheap reader comes back at chance,** re-run once with a stronger model
before concluding the channel is unreadable. "A weak reader failed" and "the
information isn't there" are different claims.

## Open ideas not yet built

- **AR round-trip** (~40 lines): descriptions → reconstructor → reconstructed
  activations → re-probe. Measures where in the channel the signal is lost, and
  completes input → acts → desc → reconstruction as a single figure. Already in
  `METHODOLOGY_f.md` §6; promote it into the NLA session.
- **Second judge** (~$2, ~1h): every deflection claim rests on 36 positives from
  one GPT-4o-mini pass. Re-judging with a second model either armors the headline
  or tells you now rather than at review time. Highest value-per-dollar item in
  the project and not scheduled anywhere.
- **Paper reframe:** lead with the *verbalization channel audit* rather than
  "first description of CI representations" (which L6/L7 already killed). Turns
  every negative on record into calibration of the instrument.
- **Pre-commit the "eval-awareness in costume" paper variant** before A3 runs, so
  either outcome lands in a paper you have already outlined.

---

## Still open from earlier sessions

- Fix or drop tier 4 (**L14** — ID 495 is a confirmed judge hallucination still
  live in the bf16 CSV). Must be done before any tier-4 number is reported.
- `HANDOFF.md` §2/§3 still describe pre-audit state in places.
