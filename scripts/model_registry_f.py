#!/usr/bin/env python3
"""Per-model architecture and path configuration for the multi-model pipeline.

Adding a second subject model must not perturb any committed Qwen result, so
the Qwen entry reproduces the frozen constants byte-for-byte and every path
helper returns the original unsuffixed filename for the default tag. A new
model gets its own suffixed artifacts and never touches the originals.

Layer choice across architectures. The registered Qwen readout is decoder block
19 of 28 (reported layer 20), i.e. 71.4% of depth -- chosen because it is the
only publicly released NLA checkpoint depth, not because 20 is special. For a
model with a different depth the comparable cell is the same *fraction* of
depth, so the primary block is round(0.714 * n_blocks) - 1 and the grid is the
depth-matched image of {10,15,20,24,28}/28. This is a registered choice, made
before any second-model activation is seen; the five-layer grid is reported
descriptively so the choice can be checked after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

QWEN_BLOCKS = 28
QWEN_REPORTED_LAYERS = (10, 15, 20, 24, 28)
QWEN_PRIMARY_REPORTED = 20
PRIMARY_DEPTH_FRACTION = QWEN_PRIMARY_REPORTED / QWEN_BLOCKS  # 0.714


def _depth_matched(reported_layers, src_blocks: int, dst_blocks: int) -> tuple[int, ...]:
    """Map reported layers from a source depth onto a destination depth."""
    out = []
    for lay in reported_layers:
        m = max(1, min(dst_blocks, round(lay / src_blocks * dst_blocks)))
        if m not in out:
            out.append(m)
    return tuple(out)


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    model_id: str
    n_blocks: int
    hidden: int
    reported_layers: tuple[int, ...]
    primary_reported_layer: int
    supports_system_role: bool = True
    notes: str = ""

    @property
    def block_indices(self) -> tuple[int, ...]:
        return tuple(l - 1 for l in self.reported_layers)

    @property
    def primary_block(self) -> int:
        return self.primary_reported_layer - 1

    @property
    def depth_fraction(self) -> float:
        return self.primary_reported_layer / self.n_blocks

    def suffix(self, stem: str, ext: str) -> str:
        """results/<stem>_f.csv for the default tag; <stem>_<tag>_f.csv otherwise."""
        return f"{stem}_f.{ext}" if self.tag == DEFAULT_TAG else f"{stem}_{self.tag}_f.{ext}"


def _llama31_8b() -> ModelSpec:
    n = 32
    return ModelSpec(
        tag="llama31_8b",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        n_blocks=n,
        hidden=4096,
        reported_layers=_depth_matched(QWEN_REPORTED_LAYERS, QWEN_BLOCKS, n),
        primary_reported_layer=max(1, round(PRIMARY_DEPTH_FRACTION * n)),
        notes="second subject model; depth-matched to the Qwen readout",
    )


REGISTRY: dict[str, ModelSpec] = {
    "qwen25_7b": ModelSpec(
        tag="qwen25_7b",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        n_blocks=QWEN_BLOCKS,
        hidden=3584,
        reported_layers=QWEN_REPORTED_LAYERS,
        primary_reported_layer=QWEN_PRIMARY_REPORTED,
        notes="registered primary model; frozen constants",
    ),
    "llama31_8b": _llama31_8b(),
}
DEFAULT_TAG = "qwen25_7b"


def get(tag: str | None = None) -> ModelSpec:
    tag = tag or DEFAULT_TAG
    if tag not in REGISTRY:
        raise SystemExit(f"unknown model tag {tag!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[tag]


def by_model_id(model_id: str) -> ModelSpec:
    for spec in REGISTRY.values():
        if spec.model_id == model_id:
            return spec
    raise SystemExit(f"no registry entry for model id {model_id!r}")


def responses_path(spec: ModelSpec) -> Path:
    """Generation output. The Qwen tag keeps the original frozen filename."""
    if spec.tag == DEFAULT_TAG:
        return Path("results/benchmark_results_bf16.csv")
    return Path(f"results/responses_{spec.tag}_f.csv")


def paths(spec: ModelSpec) -> dict[str, Path]:
    s = spec.suffix
    return {
        "responses": responses_path(spec),
        "annotations": Path("results/" + s("behavior_annotations_sol_tier3_canonical", "json")),
        "canonical": Path("results/" + s("behavior_labels_tier3_canonical", "csv")),
        "alignment_csv": Path("results/" + s("onset_alignment", "csv")),
        "alignment_json": Path("results/" + s("onset_alignment", "json")),
        "cue_sheet": Path("results/" + s("onset_cue_audit_sheet", "csv")),
        "cue_review": Path("results/" + s("onset_cue_audit_review", "json")),
        "acts": Path("results/" + s("onset_dynamics_acts", "npz")),
        "manifest": Path("results/" + s("onset_dynamics_manifest", "json")),
        "forced_acts": Path("results/" + s("onset_dynamics_forced_acts", "npz")),
        "forced_manifest": Path("results/" + s("onset_dynamics_forced_manifest", "json")),
        "fullseq": Path("results/" + s("onset_dynamics_fullseq", "npz")),
        "out_dir": Path("results/paper_pipeline/03_onset_dynamics")
        / ("" if spec.tag == DEFAULT_TAG else spec.tag),
    }


def summary() -> str:
    lines = []
    for tag, s in REGISTRY.items():
        lines.append(
            f"{tag:12s} {s.model_id:36s} blocks={s.n_blocks:3d} hidden={s.hidden:5d} "
            f"primary=L{s.primary_reported_layer} (block {s.primary_block}, "
            f"{100*s.depth_fraction:.1f}% depth) grid={list(s.reported_layers)}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
