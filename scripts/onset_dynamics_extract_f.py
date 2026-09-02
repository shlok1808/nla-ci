#!/usr/bin/env python3
"""Step 3 onset-aligned multi-layer extraction for Qwen2.5-7B-Instruct.

One teacher-forced forward pass per canonical scenario captures five decoder
block outputs at prompt-final, onset-relative, and response-final positions.
The output contains IDs and alignment metadata but deliberately contains no
behaviour labels; analysis always joins the canonical label table by ID.

Storage is exact: registered cells are float32 (a lossless widening of the
model's bf16 residual stream) and the optional full-sequence dump keeps the raw
bf16 bit pattern. Next-token logit summaries are computed by applying the
model's own final norm and unembedding only at the registered positions,
which is numerically identical to slicing the full-vocabulary logits.

Cross-check: `prompt_final` at layer 20 is compared per scenario with the
Step 1/2 store `results/activations_layer20.npz` (prompt-only forward). Under
causal attention the appended response cannot change prompt states, so the
cosine must be ~1; a mismatch means a template, tokenizer, hook, or layer
error and the analysis refuses the bundle.

GPU usage:
    python scripts/onset_dynamics_extract_f.py

Smoke test (writes scratch only, skips the review gate):
    python scripts/onset_dynamics_extract_f.py --smoke 3
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

import model_registry_f as _registry
from onset_dynamics_common_f import (
    EXPECTED_ANALYSIS_N,
    EXPECTED_CANONICAL_N,
    SPEC,
    ALIGNMENT,
    ALIGNMENT_META,
    ALL_OFFSETS,
    BENCHMARK,
    BLOCK_INDICES,
    CANONICAL,
    EXPECTED_BLOCKS,
    EXPECTED_HIDDEN,
    MODEL_ID,
    POSITION_NAMES,
    POSITION_SEMANTICS,
    HAS_STEP2_CROSSCHECK,
    PRIMARY_BLOCK,
    REVIEW_EXCLUDED_IDS,
    REPORTED_LAYERS,
    STEP2_ACTS,
    TIER3,
    bf16_bits_from_tensor,
    build_tokenized_example,
    load_step3_rows,
    position_indices,
    sha256_file,
)

_P = _registry.paths(SPEC)

THIS_SCRIPT = Path(__file__)
COMMON_SCRIPT = Path(__file__).with_name("onset_dynamics_common_f.py")
CUE_REVIEW = _P["cue_review"]
CUE_CANDIDATES = Path("results/" + SPEC.suffix("onset_cue_audit_candidates", "csv"))
CUE_SHEET = _P["cue_sheet"]

CELL_KEYS = (
    "scenario_ids", "activations", "absolute_indices", "response_indices", "valid",
    "prompt_lengths", "response_lengths", "next_token_entropy", "next_token_top1",
    "next_token_top5_mass", "crosscheck_cos",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=_P["acts"])
    p.add_argument("--manifest", type=Path, default=_P["manifest"])
    p.add_argument("--fullseq", type=Path, default=_P["fullseq"],
                   help="exploratory bf16-exact dump of every response position at all five layers")
    p.add_argument("--no-fullseq", action="store_true")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--revision", default="main")
    p.add_argument("--attn", default="sdpa", help="attention implementation passed to from_pretrained")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--smoke", type=int, default=0, metavar="N")
    p.add_argument(
        "--allow-unreviewed-cues",
        action="store_true",
        help="development only; production extraction requires a GO review record bound to the current cue audit",
    )
    return p.parse_args()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    ).stdout.strip()


def _atomic_npz(path: Path, compress: bool, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    (np.savez_compressed if compress else np.savez)(tmp, **arrays)
    os.replace(tmp, path)


def _empty_state():
    return {k: [] for k in CELL_KEYS}


def load_checkpoint(path: Path, config_key: str):
    if not path.exists():
        return _empty_state()
    z = np.load(path, allow_pickle=False)
    if str(z["config_key"].item()) != config_key:
        raise RuntimeError("checkpoint configuration differs from this extraction")
    return {k: list(z[k]) for k in CELL_KEYS}


def save_checkpoint(path: Path, state, config_key: str, compress: bool) -> None:
    acts = np.asarray(state["activations"], dtype=np.float32)
    if acts.size:
        valid = np.asarray(state["valid"], dtype=bool)          # (n, n_pos)
        finite_cells = np.isfinite(acts).all(axis=(1, 3))       # (n, n_pos)
        if not finite_cells[valid].all():
            raise RuntimeError("non-finite activation in a valid cell; refusing to write")
    _atomic_npz(
        path, compress,
        config_key=np.asarray(config_key),
        scenario_ids=np.asarray(state["scenario_ids"], dtype=np.int32),
        activations=acts,
        absolute_indices=np.asarray(state["absolute_indices"], dtype=np.int32),
        response_indices=np.asarray(state["response_indices"], dtype=np.int32),
        valid=np.asarray(state["valid"], dtype=bool),
        prompt_lengths=np.asarray(state["prompt_lengths"], dtype=np.int32),
        response_lengths=np.asarray(state["response_lengths"], dtype=np.int32),
        next_token_entropy=np.asarray(state["next_token_entropy"], dtype=np.float32),
        next_token_top1=np.asarray(state["next_token_top1"], dtype=np.float32),
        next_token_top5_mass=np.asarray(state["next_token_top5_mass"], dtype=np.float32),
        crosscheck_cos=np.asarray(state["crosscheck_cos"], dtype=np.float32),
        block_indices=np.asarray(BLOCK_INDICES, dtype=np.int16),
        reported_layers=np.asarray(REPORTED_LAYERS, dtype=np.int16),
        position_names=np.asarray(POSITION_NAMES),
        offsets=np.asarray(ALL_OFFSETS, dtype=np.int16),
        storage_dtype=np.asarray("float32 (lossless widening of bf16)"),
    )


def load_fullseq(path: Path, config_key: str):
    if not path.exists():
        return {"scenario_ids": [], "segments": []}
    z = np.load(path, allow_pickle=False)
    if str(z["config_key"].item()) != config_key:
        raise RuntimeError("full-sequence checkpoint configuration differs from this extraction")
    offsets = z["segment_offsets"]
    bits = z["acts_bf16_bits"]
    return {
        "scenario_ids": list(z["scenario_ids"]),
        "segments": [bits[offsets[i]: offsets[i + 1]] for i in range(len(z["scenario_ids"]))],
    }


def save_fullseq(path: Path, fs, config_key: str, compress: bool) -> None:
    lengths = np.asarray([len(s) for s in fs["segments"]], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    bits = (np.concatenate(fs["segments"], axis=0) if fs["segments"]
            else np.zeros((0, len(BLOCK_INDICES), EXPECTED_HIDDEN), np.uint16))
    _atomic_npz(
        path, compress,
        config_key=np.asarray(config_key),
        scenario_ids=np.asarray(fs["scenario_ids"], dtype=np.int32),
        segment_offsets=offsets.astype(np.int64),
        acts_bf16_bits=bits,
        block_indices=np.asarray(BLOCK_INDICES, dtype=np.int16),
        reported_layers=np.asarray(REPORTED_LAYERS, dtype=np.int16),
        layout=np.asarray(
            "segment i = rows [prompt_final, response_0, ..., response_{L-1}] x (layer, hidden); "
            "decode with onset_dynamics_common_f.bf16_bits_to_float32; exploratory only"
        ),
    )


def cue_gate(smoke: bool, allow: bool) -> dict:
    if smoke or allow:
        return {"status": "skipped", "reason": "smoke" if smoke else "--allow-unreviewed-cues"}
    if not CUE_REVIEW.exists():
        raise SystemExit(
            "BLOCKED: run onset_cue_audit_f.py, review every sheet row, and record the review "
            "with --record-review. Use --allow-unreviewed-cues only for a non-paper development run."
        )
    review = json.loads(CUE_REVIEW.read_text())
    verdict = review.get("verdict")
    excluded = frozenset(int(x) for x in review.get("excluded_scenario_ids", []))
    allowed = verdict == "GO" and not excluded
    allowed = allowed or (verdict == "GO_WITH_EXCLUSIONS" and excluded == REVIEW_EXCLUDED_IDS)
    if not allowed:
        raise SystemExit(f"BLOCKED: cue-audit review verdict is {review.get('verdict')!r}")
    for key, path in (("candidates_sha256", CUE_CANDIDATES), ("sheet_sha256", CUE_SHEET)):
        if not path.exists() or sha256_file(path) != review.get(key):
            raise SystemExit(f"BLOCKED: {path} changed since the review was recorded; re-review")
    reviewed = Path(review.get("sheet_reviewed_path", ""))
    if not reviewed.exists() or sha256_file(reviewed) != review.get("sheet_reviewed_sha256"):
        raise SystemExit("BLOCKED: reviewed sheet missing or changed since the review was recorded")
    return {"status": verdict, **{k: review.get(k) for k in ("reviewer", "reviewed_at_utc", "n_rows", "dispositions", "excluded_scenario_ids")}}


def load_model(model_id: str, revision: str, attn: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    kwargs = dict(revision=revision, torch_dtype=torch.bfloat16, device_map="auto")
    if attn:
        kwargs["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    if len(model.model.layers) != EXPECTED_BLOCKS:
        raise RuntimeError(f"expected {EXPECTED_BLOCKS} decoder blocks, got {len(model.model.layers)}")
    if int(model.config.hidden_size) != EXPECTED_HIDDEN:
        raise RuntimeError(f"expected hidden size {EXPECTED_HIDDEN}, got {model.config.hidden_size}")
    if next(model.parameters()).dtype != torch.bfloat16:
        raise RuntimeError("model must run in bf16 so the full-sequence dump is lossless")
    return model, tokenizer


def extract_one(model, ids, absolute_indices, block_indices=BLOCK_INDICES, fullseq_from=None):
    """Teacher-forced pass over `ids` (1-D LongTensor: prompt + response).

    Returns registered-cell activations (n_layers, n_pos, hidden) float32 with
    NaN in invalid cells, the valid mask, next-token entropy/top1/top5 mass at
    each valid cell, and — when `fullseq_from` (an absolute index) is given —
    the exact bf16 bits of every position from that index on, shaped
    (n_seq, n_layers, hidden).
    """
    import torch

    captured = {}

    def make_hook(name):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[name] = h[0] if h.dim() == 3 else h
        return hook

    handles = [model.model.layers[b].register_forward_hook(make_hook(b)) for b in block_indices]
    handles.append(model.model.norm.register_forward_hook(make_hook("final_norm")))
    device = next(model.parameters()).device
    try:
        with torch.inference_mode():
            model.model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    hidden = int(captured[block_indices[0]].shape[-1])
    n_pos = len(absolute_indices)
    acts = np.full((len(block_indices), n_pos, hidden), np.nan, np.float32)
    entropy = np.full(n_pos, np.nan, np.float32)
    top1 = np.full(n_pos, np.nan, np.float32)
    top5 = np.full(n_pos, np.nan, np.float32)
    valid = np.asarray(absolute_indices) >= 0
    take = torch.as_tensor(np.asarray(absolute_indices)[valid], dtype=torch.long, device=device)
    for li, block in enumerate(block_indices):
        acts[li, valid] = captured[block].index_select(0, take).float().cpu().numpy()

    with torch.inference_mode():
        logits = model.lm_head(captured["final_norm"].index_select(0, take)).float()
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        entropy[valid] = (-(p * logp).sum(dim=-1)).cpu().numpy()
        values = torch.topk(p, k=min(5, p.shape[-1]), dim=-1).values
        top1[valid] = values[:, 0].cpu().numpy()
        top5[valid] = values.sum(dim=-1).cpu().numpy()

    fullseq = None
    if fullseq_from is not None:
        fullseq = np.stack(
            [bf16_bits_from_tensor(captured[b][fullseq_from:]) for b in block_indices], axis=1
        )
    return acts, valid, entropy, top1, top5, fullseq


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main() -> None:
    args = parse_args()
    smoke = args.smoke > 0
    if smoke:
        args.output = Path("scratch/onset_dynamics_smoke.npz")
        args.manifest = Path("scratch/onset_dynamics_smoke_manifest.json")
        args.fullseq = Path("scratch/onset_dynamics_smoke_fullseq.npz")
    want_fullseq = not args.no_fullseq
    gate = cue_gate(smoke, args.allow_unreviewed_cues)

    alignment_meta = json.loads(ALIGNMENT_META.read_text())
    if alignment_meta.get("verdict") != "GO" or alignment_meta.get("roundtrip_failures") != 0:
        raise SystemExit("BLOCKED: onset-alignment preflight is not GO with zero failures")

    rows = load_step3_rows()
    if smoke:
        rows = rows.head(args.smoke)

    source_hashes = {
        str(p): sha256_file(p)
        for p in (CANONICAL, ALIGNMENT, ALIGNMENT_META, BENCHMARK, TIER3, THIS_SCRIPT, COMMON_SCRIPT)
    }
    config = {
        "schema_version": "onset-dynamics-v2",
        "model_id": args.model,
        "revision_requested": args.revision,
        "attn_implementation": args.attn,
        "block_indices": list(BLOCK_INDICES),
        "reported_layers": list(REPORTED_LAYERS),
        "position_names": list(POSITION_NAMES),
        "offsets": list(ALL_OFFSETS),
        "position_semantics": POSITION_SEMANTICS,
        "source_hashes": source_hashes,
        "git_head": _git_head(),
        "smoke_n": args.smoke,
    }
    config_key = json.dumps(config, sort_keys=True, separators=(",", ":"))
    state = load_checkpoint(args.output, config_key)
    fs = load_fullseq(args.fullseq, config_key) if want_fullseq else None
    if fs is not None and list(map(int, fs["scenario_ids"])) != list(map(int, state["scenario_ids"])):
        raise RuntimeError("registered-cell and full-sequence checkpoints disagree; delete both and restart")
    done = set(int(x) for x in state["scenario_ids"])

    old = None
    if HAS_STEP2_CROSSCHECK and STEP2_ACTS.exists():
        z = np.load(STEP2_ACTS, allow_pickle=True)
        old = {int(s): z["activations"][i] for i, s in enumerate(z["scenario_ids"])}
    layer20 = list(BLOCK_INDICES).index(PRIMARY_BLOCK)

    model = tokenizer = None
    started = time.time()
    for row in rows.itertuples(index=False):
        sid = int(row.scenario_id)
        if sid in done:
            continue
        if model is None:
            model, tokenizer = load_model(args.model, args.revision, args.attn)
        import torch

        _, prompt_ids, _, response_ids = build_tokenized_example(row, tokenizer)
        absolute, response = position_indices(len(prompt_ids), len(response_ids), row.cutoff_tok)
        ids = torch.cat([prompt_ids, response_ids])
        acts, valid, entropy, top1, top5, fullseq = extract_one(
            model, ids, absolute, BLOCK_INDICES,
            fullseq_from=(len(prompt_ids) - 1) if want_fullseq else None,
        )
        cos = np.nan
        if old is not None and sid in old:
            cos = _cosine(acts[layer20, 0], old[sid])
        state["scenario_ids"].append(sid)
        state["activations"].append(acts)
        state["absolute_indices"].append(absolute)
        state["response_indices"].append(response)
        state["valid"].append(valid)
        state["prompt_lengths"].append(len(prompt_ids))
        state["response_lengths"].append(len(response_ids))
        state["next_token_entropy"].append(entropy)
        state["next_token_top1"].append(top1)
        state["next_token_top5_mass"].append(top5)
        state["crosscheck_cos"].append(cos)
        if fs is not None:
            fs["scenario_ids"].append(sid)
            fs["segments"].append(fullseq)
        done.add(sid)
        if len(done) % args.save_every == 0:
            save_checkpoint(args.output, state, config_key, compress=False)
            if fs is not None:
                save_fullseq(args.fullseq, fs, config_key, compress=False)
            print(f"checkpoint: {len(done)}/{len(rows)}  step2 cos={cos:.4f}  {time.time()-started:.0f}s", flush=True)

    if not smoke and len(state["scenario_ids"]) != EXPECTED_CANONICAL_N:
        raise RuntimeError(f"production extraction ended with {len(state['scenario_ids'])} scenarios, expected {EXPECTED_CANONICAL_N}")
    save_checkpoint(args.output, state, config_key, compress=True)
    if fs is not None:
        save_fullseq(args.fullseq, fs, config_key, compress=True)

    import torch
    import transformers

    cc = np.asarray(state["crosscheck_cos"], dtype=np.float64)
    cc = cc[np.isfinite(cc)]
    manifest = {
        **config,
        "status": "complete",
        "n_scenarios": len(state["scenario_ids"]),
        "n_analysis_expected": EXPECTED_ANALYSIS_N if not smoke else None,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "fullseq": str(args.fullseq) if fs is not None else None,
        "fullseq_sha256": sha256_file(args.fullseq) if fs is not None else None,
        "elapsed_seconds": time.time() - started,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_commit_hash": getattr(model.config, "_commit_hash", None) if model else None,
        "tokenizer_commit_hash": getattr(tokenizer, "_commit_hash", None) if tokenizer else None,
        "dtype": "float32 storage (lossless widening of bf16); bf16 forward",
        "hook_semantics": "decoder block output (residual stream after block b) == HF hidden_states[b+1]; before model final norm",
        "logit_semantics": "final norm + lm_head applied at registered positions only (identical to full logits sliced)",
        "cue_review_gate": gate,
        "step2_crosscheck": {
            "store": str(STEP2_ACTS),
            "n_compared": int(cc.size),
            "cos_min": float(cc.min()) if cc.size else None,
            "cos_median": float(np.median(cc)) if cc.size else None,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.output} and {args.manifest}")
    if cc.size:
        print(f"step2 cross-check cosine: min {cc.min():.4f} median {np.median(cc):.4f} (n={cc.size})")


if __name__ == "__main__":
    main()
