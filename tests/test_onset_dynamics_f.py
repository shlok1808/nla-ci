"""GPU-free integrity checks for the Step 3 design.

Covers the assumptions that would silently invalidate the experiment: hook
and layer semantics, char->token onset alignment and off-by-one, visible
prefix/leakage boundary, exactness of the vectorised inference machinery,
checkpoint resume and configuration drift, lossless bf16 storage, the cue
review gate, and forced-prefix tokenisation. Model-level tests use a tiny
random Qwen2 built on CPU; tokenizer tests use the cached Qwen tokenizer and
skip if it is unavailable offline.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from onset_cue_audit_f import CUE, DISPOSITIONS
from onset_dynamics_common_f import (
    ALL_OFFSETS,
    BLOCK_INDICES,
    POSITION_NAMES,
    PRIMARY_BLOCK,
    PRIMARY_CELLS,
    PRIMARY_OFFSETS,
    REVIEW_EXCLUDED_IDS,
    REPORTED_LAYERS,
    bf16_bits_from_tensor,
    bf16_bits_to_float32,
    build_tokenized_example,
    load_step3_rows,
    position_indices,
    visible_prefix,
)
from onset_dynamics_extract_f import (
    cue_gate,
    extract_one,
    load_checkpoint,
    save_checkpoint,
)
from onset_dynamics_forced_extract_f import FIXED_PREFIXES, PREFIXES, render_prefix
from onset_dynamics_stats_f import (
    N_OUTER,
    PairedFoldAUC,
    holm,
    naive_mean_fold_auc,
    outer_splits,
    paired_inference,
    stratified_draws,
)

REPO = Path(__file__).resolve().parents[1]


def _have_data():
    return all((REPO / p).exists() for p in (
        "results/behavior_labels_tier3_canonical_f.csv", "results/onset_alignment_f.csv",
        "results/benchmark_results_bf16.csv", "data/tier_3.txt"))


@pytest.fixture(scope="module")
def qwen_tokenizer():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", local_files_only=True)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Qwen tokenizer not cached: {e}")


@pytest.fixture(scope="module")
def rows():
    if not _have_data():
        pytest.skip("canonical data not present")
    import os
    os.chdir(REPO)
    return load_step3_rows()


@pytest.fixture(scope="module")
def tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=28,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    return Qwen2ForCausalLM(cfg).eval()


# ── registered grid / conventions ───────────────────────────────────────────

def test_registered_grid_and_layer_convention():
    assert PRIMARY_OFFSETS == tuple(range(-8, 0))
    assert BLOCK_INDICES == (9, 14, 19, 23, 27)
    assert REPORTED_LAYERS == (10, 15, 20, 24, 28)
    assert PRIMARY_BLOCK == 19 and REPORTED_LAYERS[BLOCK_INDICES.index(PRIMARY_BLOCK)] == 20
    assert len(POSITION_NAMES) == len(ALL_OFFSETS) + 2
    assert PRIMARY_CELLS == ("prompt_final",) + tuple(f"offset_{x:+d}" for x in PRIMARY_OFFSETS)


def test_position_indices_never_wrap_invalid_offsets():
    absolute, response = position_indices(prompt_len=100, response_len=6, cutoff_tok=2)
    by_name = dict(zip(POSITION_NAMES, zip(absolute, response)))
    assert by_name["prompt_final"] == (99, -1)
    assert by_name["offset_-8"] == (-1, -1)
    assert by_name["offset_-1"] == (101, 1)
    assert by_name["offset_+1"] == (103, 3)
    assert by_name["offset_+8"] == (-1, -1)
    assert by_name["response_final"] == (105, 5)
    absolute, response = position_indices(50, 10, cutoff_tok=float("nan"))
    assert (response[1:-1] == -1).all() and response[0] == -1 and response[-1] == 9


def test_offset_minus_one_is_state_before_onset_token():
    """offset_-1 must index the token immediately BEFORE the onset token, so
    the onset token itself has not been read."""
    absolute, response = position_indices(prompt_len=10, response_len=40, cutoff_tok=20)
    by_name = dict(zip(POSITION_NAMES, response))
    assert by_name["offset_-1"] == 19 and by_name["offset_+0"] == 20 and by_name["offset_-8"] == 12


# ── model semantics on a tiny Qwen2 (CPU) ───────────────────────────────────

def test_hooks_match_hidden_states_and_logits(tiny_qwen):
    ids = torch.randint(0, 97, (60,))
    absolute = np.array([9, -1, 30, 59, 3])
    acts, valid, entropy, top1, top5, fullseq = extract_one(tiny_qwen, ids, absolute, BLOCK_INDICES, fullseq_from=9)
    with torch.inference_mode():
        ref = tiny_qwen(input_ids=ids[None], output_hidden_states=True, use_cache=False)
    assert valid.tolist() == [True, False, True, True, True]
    for li, b in enumerate(BLOCK_INDICES):
        if b + 1 < len(ref.hidden_states) - 1:
            expect = ref.hidden_states[b + 1][0].float().numpy()      # block b output == hidden_states[b+1]
        else:
            # HF's last hidden_states entry is post final-norm; the hook stores
            # the pre-norm residual, so compare after applying the model's norm.
            with torch.inference_mode():
                expect = ref.hidden_states[-1][0].float().numpy()
            got = torch.from_numpy(acts[li].copy())
            with torch.inference_mode():
                normed = tiny_qwen.model.norm(got.to(tiny_qwen.model.norm.weight.dtype)).float().numpy()
            for pi, a in enumerate(absolute):
                if a >= 0:
                    np.testing.assert_allclose(normed[pi], expect[a], rtol=1e-4, atol=1e-5)
            continue
        for pi, a in enumerate(absolute):
            if a >= 0:
                np.testing.assert_allclose(acts[li, pi], expect[a], rtol=1e-5, atol=1e-5)
            else:
                assert np.isnan(acts[li, pi]).all()
    logits = ref.logits[0].float()
    p = torch.softmax(logits, -1)
    ent = -(p * torch.log_softmax(logits, -1)).sum(-1).numpy()
    for pi, a in enumerate(absolute):
        if a >= 0:
            assert abs(entropy[pi] - ent[a]) < 1e-4
            assert abs(top1[pi] - p[a].max().item()) < 1e-5
            assert abs(top5[pi] - torch.topk(p[a], 5).values.sum().item()) < 1e-5
    # full-sequence dump is bf16-exact and aligned from the requested index
    assert fullseq.shape == (60 - 9, len(BLOCK_INDICES), 32) and fullseq.dtype == np.uint16
    dec = bf16_bits_to_float32(fullseq[:, 0])
    np.testing.assert_allclose(dec, ref.hidden_states[BLOCK_INDICES[0] + 1][0, 9:].to(torch.bfloat16).float().numpy())


def test_prompt_states_unchanged_by_appended_response(tiny_qwen):
    """Causal attention: the prompt-final state must not depend on the response,
    which is what makes the Step 2 cross-check a valid gate."""
    prompt = torch.randint(0, 97, (25,))
    resp = torch.randint(0, 97, (30,))
    a, *_ = extract_one(tiny_qwen, prompt, np.array([24]), BLOCK_INDICES)
    b, *_ = extract_one(tiny_qwen, torch.cat([prompt, resp]), np.array([24]), BLOCK_INDICES)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_bf16_codec_roundtrip():
    t = (torch.randn(500) * 1000).to(torch.bfloat16)
    bits = bf16_bits_from_tensor(t)
    np.testing.assert_array_equal(bf16_bits_to_float32(bits), t.float().numpy())


# ── checkpoint resume / drift / finiteness ──────────────────────────────────

def _fake_state(n, hidden=8):
    st = {k: [] for k in ("scenario_ids", "activations", "absolute_indices", "response_indices", "valid",
                          "prompt_lengths", "response_lengths", "next_token_entropy", "next_token_top1",
                          "next_token_top5_mass", "crosscheck_cos")}
    P = len(POSITION_NAMES)
    for i in range(n):
        acts = np.random.rand(len(BLOCK_INDICES), P, hidden).astype(np.float32)
        valid = np.ones(P, bool); valid[1] = False; acts[:, 1] = np.nan
        st["scenario_ids"].append(300 + i); st["activations"].append(acts)
        st["absolute_indices"].append(np.arange(P)); st["response_indices"].append(np.arange(P) - 1)
        st["valid"].append(valid); st["prompt_lengths"].append(10); st["response_lengths"].append(20)
        for k in ("next_token_entropy", "next_token_top1", "next_token_top5_mass"):
            st[k].append(np.random.rand(P).astype(np.float32))
        st["crosscheck_cos"].append(0.999)
    return st


def test_checkpoint_roundtrip_rejects_drift_and_nonfinite(tmp_path):
    path = tmp_path / "ck.npz"
    st = _fake_state(3)
    save_checkpoint(path, st, "cfg-A", compress=False)
    back = load_checkpoint(path, "cfg-A")
    assert [int(x) for x in back["scenario_ids"]] == [300, 301, 302]
    np.testing.assert_array_equal(np.asarray(back["activations"]), np.asarray(st["activations"]))
    with pytest.raises(RuntimeError):
        load_checkpoint(path, "cfg-B")
    st["activations"][0][0, 0, 0] = np.inf                      # valid cell -> must refuse
    with pytest.raises(RuntimeError):
        save_checkpoint(path, st, "cfg-A", compress=False)


def test_cue_gate_requires_bound_go_review(tmp_path, monkeypatch):
    import onset_dynamics_extract_f as ex
    monkeypatch.setattr(ex, "CUE_REVIEW", tmp_path / "review.json")
    monkeypatch.setattr(ex, "CUE_CANDIDATES", tmp_path / "cand.csv")
    monkeypatch.setattr(ex, "CUE_SHEET", tmp_path / "sheet.csv")
    assert cue_gate(smoke=True, allow=False)["status"] == "skipped"
    with pytest.raises(SystemExit):
        cue_gate(False, False)
    (tmp_path / "cand.csv").write_text("a\n"); (tmp_path / "sheet.csv").write_text("b\n")
    (tmp_path / "reviewed.csv").write_text("c\n")
    from onset_dynamics_common_f import sha256_file
    rec = {"verdict": "GO", "excluded_scenario_ids": [], "candidates_sha256": sha256_file(tmp_path / "cand.csv"),
           "sheet_sha256": sha256_file(tmp_path / "sheet.csv"),
           "sheet_reviewed_path": str(tmp_path / "reviewed.csv"),
           "sheet_reviewed_sha256": sha256_file(tmp_path / "reviewed.csv"), "reviewer": "x"}
    (tmp_path / "review.json").write_text(json.dumps(rec))
    assert cue_gate(False, False)["status"] == "GO"
    rec.update({"verdict": "GO_WITH_EXCLUSIONS", "excluded_scenario_ids": sorted(REVIEW_EXCLUDED_IDS)})
    (tmp_path / "review.json").write_text(json.dumps(rec))
    assert cue_gate(False, False)["status"] == "GO_WITH_EXCLUSIONS"
    (tmp_path / "sheet.csv").write_text("changed\n")           # screen re-run after review -> blocked
    with pytest.raises(SystemExit):
        cue_gate(False, False)
    rec["verdict"] = "NO-GO"; (tmp_path / "review.json").write_text(json.dumps(rec))
    with pytest.raises(SystemExit):
        cue_gate(False, False)
    assert DISPOSITIONS == {"ok", "earlier_cue", "unsure"}


# ── cue regex ───────────────────────────────────────────────────────────────

def test_cue_screen_catches_early_limiting_language():
    for s in ("I probably shouldn't share this, but the answer is yes.",
              "Without revealing private details, I can give a summary.",
              "but let's keep this one in check, okay?", "I don't want to get into that.",
              "That's really his story to tell.", "She told me in confidence.",
              "I'm not going to discuss it.", "Oh, Jack, I really wish you hadn't brought that up."):
        assert CUE.search(s), s
    assert not CUE.search("The meeting starts at nine tomorrow morning.")


# ── inference machinery: exactness ─────────────────────────────────────────

def _folds(y, R):
    fold_ids = np.full((R, len(y)), -1, dtype=np.int8)
    for r in range(R):
        for k, (_, te) in enumerate(outer_splits(y, 100 + r)):
            fold_ids[r, te] = k
    return fold_ids


def test_vectorised_auc_equals_naive_with_ties_and_duplicates():
    rng = np.random.default_rng(3)
    n, R, C = 60, 3, 4
    y = (np.arange(n) < 25).astype(int); rng.shuffle(y)
    fold_ids = _folds(y, R)
    a = rng.choice([0.1, 0.2, 0.3, 0.4, 0.5], size=(C, R, n)) + 0.3 * y * rng.random((C, R, n))
    b = rng.random((C, R, n)) * 3
    eng = PairedFoldAUC(a, fold_ids, y, b)
    ones = np.ones((1, n))
    assert abs(eng.weighted(ones, "a").mean() - naive_mean_fold_auc(a, fold_ids, y)) < 1e-12
    assert abs(eng.weighted(ones, "b").mean() - naive_mean_fold_auc(b, fold_ids, y)) < 1e-12
    W = stratified_draws(y, 15, rng)
    for i in range(15):
        draw = np.repeat(np.arange(n), W[i].astype(int))
        assert abs(eng.weighted(W[i:i + 1], "a").mean() - naive_mean_fold_auc(a, fold_ids, y, draw)) < 1e-12


def test_swap_null_equals_naive_rank_normalised_swap():
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(5)
    n, R, C = 40, 2, 2
    y = (np.arange(n) < 17).astype(int); rng.shuffle(y)
    fold_ids = _folds(y, R)
    a = rng.random((C, R, n)); b = rng.random((C, R, n)) * 10
    eng = PairedFoldAUC(a, fold_ids, y, b)
    flips = rng.random((6, n)) < 0.5
    got = eng.swapped_delta(flips).mean(axis=1)
    for i in range(6):
        va, vb = [], []
        for c in range(C):
            for r in range(R):
                for k in range(N_OUTER):
                    m = np.flatnonzero(fold_ids[r] == k)
                    ra, rb = rankdata(a[c, r, m]), rankdata(b[c, r, m])
                    f = flips[i, m]
                    va.append(roc_auc_score(y[m], np.where(f, rb, ra)))
                    vb.append(roc_auc_score(y[m], np.where(f, ra, rb)))
        assert abs(got[i] - (np.mean(va) - np.mean(vb))) < 1e-12


def test_paired_inference_detects_a_large_channel_advantage_and_is_null_under_swap():
    rng = np.random.default_rng(7)
    n, R, C = 100, 2, 2
    y = np.tile([0, 1], n // 2)
    fold_ids = _folds(y, R)
    signal = y + rng.normal(0, 0.05, n)
    act = np.tile(signal, (C, R, 1)); text = np.tile(rng.normal(0, 1, n), (C, R, 1))
    out = paired_inference(act, text, fold_ids, y, n_boot=50, n_perm=99)
    assert out["delta_roc"] > 0.35 and out["delta_boot_lo"] > 0
    assert out["p_boot_one_sided"] <= 0.05 and out["p_swap_randomization"] <= 0.05
    same = paired_inference(act, act * 3 + 1, fold_ids, y, n_boot=50, n_perm=99)   # scale-free null
    assert abs(same["delta_roc"]) < 1e-12 and same["p_swap_randomization"] > 0.5


def test_holm():
    # step-down: 0.01*3=0.03; 0.03*2=0.06; 0.04*1=0.04 but monotone -> 0.06
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


# ── data-dependent checks (skip if data/tokenizer absent) ───────────────────

def test_canonical_step3_join_is_exact(rows):
    assert len(rows) == rows.scenario_id.nunique() == 258
    assert int(rows.population.eq("analysis").sum()) == 216
    assert rows.roundtrip_ok.astype(bool).all()


def test_visible_prefix_never_reaches_onset_char(rows, qwen_tokenizer):
    """At offset_-1 the decoded visible prefix must end strictly before the
    onset character on every discloser; at offset_-8 it must be shorter still."""
    import pandas as pd
    align = pd.read_csv(REPO / "results/onset_alignment_f.csv").set_index("scenario_id")
    checked = 0
    for row in rows.itertuples(index=False):
        if not bool(row.broad_breach):
            continue
        _, prompt_ids, _, response_ids = build_tokenized_example(row, qwen_tokenizer)
        al = align.loc[int(row.scenario_id)]
        onset_char = int(np.nanmin([al.disclosure_onset_char, al.limiting_onset_char]))
        absolute, response = position_indices(len(prompt_ids), len(response_ids), row.cutoff_tok)
        by = dict(zip(POSITION_NAMES, response))
        if by["offset_-1"] < 0:
            continue
        p1 = visible_prefix(qwen_tokenizer, response_ids, by["offset_-1"])
        assert row.response.startswith(p1)
        assert len(p1) <= onset_char, (row.scenario_id, len(p1), onset_char)
        # the onset token itself starts at or before the onset char
        p0 = visible_prefix(qwen_tokenizer, response_ids, by["offset_+0"])
        assert len(p0) > onset_char
        if by["offset_-8"] >= 0:
            assert len(visible_prefix(qwen_tokenizer, response_ids, by["offset_-8"])) < len(p1)
        checked += 1
    assert checked >= 200


def test_forced_prefixes_are_clean_and_tokenise_at_the_boundary(rows, qwen_tokenizer):
    for name, text in FIXED_PREFIXES.items():
        assert text == text.strip() and "honest" not in text.lower()
        assert "Ġ" not in qwen_tokenizer.tokenize(text)[-1]
    lengths = set()
    for row in rows.head(40).itertuples(index=False):
        prompt_text, prompt_ids, _, _ = build_tokenized_example(row, qwen_tokenizer)
        for name in PREFIXES:
            text = render_prefix(name, row)
            pre = qwen_tokenizer(text, add_special_tokens=False)["input_ids"]
            full = qwen_tokenizer(prompt_text + text, add_special_tokens=False)["input_ids"]
            assert full == prompt_ids.tolist() + pre
            if name in FIXED_PREFIXES:
                lengths.add((name, len(pre)))
    assert len(lengths) == len(FIXED_PREFIXES)          # fixed prefixes have one length each


# ── multi-model registry (llama branch) ────────────────────────────────────

def test_registry_qwen_entry_reproduces_frozen_constants():
    """The default tag must be byte-identical to the registered Qwen setup, or
    every committed result silently changes meaning."""
    import model_registry_f as reg
    q = reg.get("qwen25_7b")
    assert reg.DEFAULT_TAG == "qwen25_7b"
    assert q.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert q.n_blocks == 28 and q.hidden == 3584
    assert q.reported_layers == (10, 15, 20, 24, 28)
    assert q.block_indices == (9, 14, 19, 23, 27)
    assert q.primary_reported_layer == 20 and q.primary_block == 19
    p = reg.paths(q)
    assert p["responses"].name == "benchmark_results_bf16.csv"
    assert p["canonical"].name == "behavior_labels_tier3_canonical_f.csv"
    assert p["acts"].name == "onset_dynamics_acts_f.npz"


def test_registry_llama_is_depth_matched_and_never_collides():
    import model_registry_f as reg
    q, l = reg.get("qwen25_7b"), reg.get("llama31_8b")
    assert l.n_blocks == 32 and l.hidden == 4096
    # same fraction of depth as the registered Qwen readout, within one block
    assert abs(l.depth_fraction - q.depth_fraction) < 1.0 / l.n_blocks
    assert l.primary_reported_layer == 23 and l.primary_block == 22
    assert len(l.reported_layers) == len(q.reported_layers)
    assert l.primary_reported_layer in l.reported_layers
    assert all(1 <= x <= l.n_blocks for x in l.reported_layers)
    qp, lp = reg.paths(q), reg.paths(l)
    for k in qp:
        assert qp[k] != lp[k], f"path collision on {k}"
    assert "llama31_8b" in str(lp["acts"])


def test_llama_has_no_step2_crosscheck_and_qwen_does(monkeypatch):
    """The prompt-final cross-check gate only exists for the registered model;
    a second model must skip it explicitly, never fake a pass."""
    import importlib, sys
    import model_registry_f as reg
    monkeypatch.setenv("NLA_MODEL_TAG", "llama31_8b")
    for m in ("onset_dynamics_common_f",):
        sys.modules.pop(m, None)
    c = importlib.import_module("onset_dynamics_common_f")
    assert c.HAS_STEP2_CROSSCHECK is False
    assert c.EXPECTED_BLOCKS == 32 and c.EXPECTED_HIDDEN == 4096
    monkeypatch.delenv("NLA_MODEL_TAG")
    sys.modules.pop("onset_dynamics_common_f", None)
    c = importlib.import_module("onset_dynamics_common_f")
    assert c.HAS_STEP2_CROSSCHECK is True
    assert c.EXPECTED_BLOCKS == 28 and c.EXPECTED_HIDDEN == 3584

