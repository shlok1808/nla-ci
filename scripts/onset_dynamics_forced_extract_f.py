#!/usr/bin/env python3
"""Layer-20 forced-prefix extraction for the Step 3 transcript control.

Every scenario receives the same fixed assistant prefixes. Capturing all
prefix positions tests whether behaviour decodability changes while the
generated words are held constant. This arm is a control for the registered
layer-20 primary analysis, so other layers are intentionally not extracted.

Prefix design notes:
  * no trailing whitespace — Qwen tokenises "Well, " as `Well , Ġ`, which would
    make the final probed position a lone-space token the model never emits;
  * the three fixed prefixes are neutral about candour (no "I'll be honest");
  * `natural` is an in-distribution TEMPLATE, "Oh, {questioner}," — 58% of the
    model's own responses open with "Oh" and 39% with "Oh, {name}," at equal
    rates in both classes. Its words vary by scenario only through the
    prompt-given name, so it is reported as a secondary, not as the registered
    fixed-word control statistic.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

import model_registry_f as _registry
from onset_dynamics_common_f import (
    SPEC,
    BENCHMARK,
    CANONICAL,
    EXPECTED_BLOCKS,
    EXPECTED_HIDDEN,
    MODEL_ID,
    PRIMARY_BLOCK,
    PRIMARY_REPORTED_LAYER,
    TIER3,
    build_tokenized_example,
    load_step3_rows,
    sha256_file,
)

_P = _registry.paths(SPEC)

FIXED_PREFIXES = {
    "well": "Well,",
    "think": "Hmm, let me think about how to put this.",
    "commit": "Okay, so here's what I'm going to say about that.",
}
TEMPLATE_PREFIXES = {
    "natural": "Oh, {questioner},",
}
PREFIXES = {**FIXED_PREFIXES, **TEMPLATE_PREFIXES}


def render_prefix(name: str, row) -> str:
    return PREFIXES[name].format(questioner=row.questioner)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=_P["forced_acts"])
    p.add_argument("--manifest", type=Path, default=_P["forced_manifest"])
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--revision", default="main")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--smoke", type=int, default=0, metavar="N")
    return p.parse_args()


def _git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()


def _atomic(path, compress, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    (np.savez_compressed if compress else np.savez)(tmp, **arrays)
    os.replace(tmp, path)


def load_model(model_id, revision, attn):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    kwargs = dict(revision=revision, torch_dtype=torch.bfloat16, device_map="auto")
    if attn:
        kwargs["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    if len(model.model.layers) != EXPECTED_BLOCKS or model.config.hidden_size != EXPECTED_HIDDEN:
        raise RuntimeError(
            f"architecture mismatch for {model_id}: {len(model.model.layers)} blocks / "
            f"hidden {model.config.hidden_size}; registry expects {EXPECTED_BLOCKS} / {EXPECTED_HIDDEN}"
        )
    return model, tok


def extract_prefix_positions(model, ids, n_positions, block=PRIMARY_BLOCK):
    """Block output at the last `n_positions` sequence positions, float32."""
    import torch

    captured = {}

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h[0] if h.dim() == 3 else h

    handle = model.model.layers[block].register_forward_hook(hook)
    device = next(model.parameters()).device
    try:
        with torch.inference_mode():
            model.model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
    finally:
        handle.remove()
    seq = captured["h"][-n_positions:].float().cpu().numpy()
    if not np.isfinite(seq).all():
        raise RuntimeError("non-finite forced-prefix activation")
    return seq


def save(path, state, config_key, max_positions, compress):
    n = len(state["scenario_ids"])
    acts = np.full((n, len(PREFIXES), max_positions, EXPECTED_HIDDEN), np.nan, np.float32)
    valid = np.zeros((n, len(PREFIXES), max_positions), bool)
    for i, per_scenario in enumerate(state["activations"]):
        for j, seq in enumerate(per_scenario):
            acts[i, j, : len(seq)] = seq
            valid[i, j, : len(seq)] = True
    _atomic(
        path, compress,
        config_key=np.asarray(config_key),
        scenario_ids=np.asarray(state["scenario_ids"], dtype=np.int32),
        activations=acts,
        valid=valid,
        prefix_names=np.asarray(list(PREFIXES)),
        prefix_texts=np.asarray(list(PREFIXES.values())),
        prefix_is_template=np.asarray([k in TEMPLATE_PREFIXES for k in PREFIXES]),
        prefix_token_lengths=np.asarray(state["prefix_token_lengths"], dtype=np.int16),
        reported_layer=np.asarray(PRIMARY_REPORTED_LAYER, dtype=np.int16),
        block_index=np.asarray(PRIMARY_BLOCK, dtype=np.int16),
        layout=np.asarray("position 0 = prompt_final; position k = state after forced prefix token k"),
    )


def main():
    args = parse_args()
    if args.smoke:
        args.output = Path("scratch/onset_dynamics_forced_smoke.npz")
        args.manifest = Path("scratch/onset_dynamics_forced_smoke_manifest.json")
    rows = load_step3_rows()
    if args.smoke:
        rows = rows.head(args.smoke)

    script = Path(__file__)
    common = script.with_name("onset_dynamics_common_f.py")
    source_hashes = {str(p): sha256_file(p) for p in (CANONICAL, BENCHMARK, TIER3, script, common)}
    config = {
        "schema_version": "onset-dynamics-forced-v2",
        "model_id": args.model,
        "revision_requested": args.revision,
        "attn_implementation": args.attn,
        "reported_layer": PRIMARY_REPORTED_LAYER,
        "block_index": PRIMARY_BLOCK,
        "prefixes": PREFIXES,
        "template_prefixes": list(TEMPLATE_PREFIXES),
        "source_hashes": source_hashes,
        "git_head": _git_head(),
        "smoke_n": args.smoke,
    }
    config_key = json.dumps(config, sort_keys=True, separators=(",", ":"))

    if args.output.exists():
        z = np.load(args.output, allow_pickle=False)
        if str(z["config_key"].item()) != config_key:
            raise RuntimeError("forced-prefix checkpoint configuration drift")
        state = {
            "scenario_ids": list(z["scenario_ids"]),
            "activations": [
                [z["activations"][i, j, z["valid"][i, j]] for j in range(len(PREFIXES))]
                for i in range(len(z["scenario_ids"]))
            ],
            "prefix_token_lengths": list(z["prefix_token_lengths"]),
        }
    else:
        state = {"scenario_ids": [], "activations": [], "prefix_token_lengths": []}

    done = set(int(x) for x in state["scenario_ids"])
    model = tok = None
    started = time.time()
    for row in rows.itertuples(index=False):
        if int(row.scenario_id) in done:
            continue
        if model is None:
            model, tok = load_model(args.model, args.revision, args.attn)
        import torch

        prompt_text, prompt_ids, _, _ = build_tokenized_example(row, tok)
        seqs, lengths = [], []
        for name in PREFIXES:
            text = render_prefix(name, row)
            if text != text.rstrip():
                raise ValueError(f"prefix {name!r} has trailing whitespace")
            prefix_ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            full_ids = tok(prompt_text + text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            joined = torch.cat([prompt_ids, prefix_ids])
            if not torch.equal(joined, full_ids):
                raise ValueError(f"prompt/forced-prefix token boundary drift for scenario {row.scenario_id}: {text!r}")
            seqs.append(extract_prefix_positions(model, joined, len(prefix_ids) + 1))
            lengths.append(len(prefix_ids))
        state["scenario_ids"].append(int(row.scenario_id))
        state["activations"].append(seqs)
        state["prefix_token_lengths"].append(lengths)
        done.add(int(row.scenario_id))
        if len(done) % args.save_every == 0:
            max_positions = int(np.max(np.asarray(state["prefix_token_lengths"]))) + 1
            save(args.output, state, config_key, max_positions, compress=False)
            print(f"checkpoint: {len(done)}/{len(rows)}  {time.time()-started:.0f}s", flush=True)
    if not args.smoke and len(state["scenario_ids"]) != 258:
        raise RuntimeError(f"production extraction ended with {len(state['scenario_ids'])} scenarios, expected 258")
    max_positions = int(np.max(np.asarray(state["prefix_token_lengths"]))) + 1
    save(args.output, state, config_key, max_positions, compress=True)

    import torch
    import transformers

    manifest = {
        **config,
        "status": "complete",
        "n_scenarios": len(state["scenario_ids"]),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "elapsed_seconds": time.time() - started,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_commit_hash": getattr(model.config, "_commit_hash", None) if model else None,
        "tokenizer_commit_hash": getattr(tok, "_commit_hash", None) if tok else None,
        "dtype": "float32 storage (lossless widening of bf16); bf16 forward",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.output} and {args.manifest}")


if __name__ == "__main__":
    main()
