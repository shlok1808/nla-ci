#!/usr/bin/env bash
# setup_lambda.sh — provision a fresh Lambda A100 image for the nla-ci pipeline.
#
# Idempotent: safe to re-run on every new instance. Handles the things that
# bite on a fresh spin-up:
#   1. A torchvision built against a different torch ABI than the one actually
#      installed -> `operator torchvision::nms does not exist` -> transformers
#      fails to import Qwen2ForCausalLM (image_utils imports torchvision.io).
#      Every experiment HERE is text-only, so it's tempting to just remove
#      torchvision — DO NOT. sglang's own srt/utils/common.py does an
#      unguarded `from torchvision.io import decode_jpeg` at import time (not
#      behind is_vision_available()), so a missing torchvision is an
#      import-time crash for sglang itself, not a clean skip.
#      (2026-08-26: removing it broke `python -m sglang.launch_server` with
#      `ModuleNotFoundError: No module named 'torchvision'`, discovered only
#      at server launch — after the model download — costing another chunk
#      of the session.)
#      So we INSTALL a torchvision matched to whatever torch sglang left
#      behind, rather than removing it. A stale apt egg-info can make pip
#      report "already satisfied" for a torchvision whose actual package
#      directory was hand-removed in an earlier session — that reads as
#      installed but is not importable, so we verify by *importing*, not by
#      trusting `pip show`.
#   2. The repo does not vendor the ConfAIde benchmark; scripts need data/*.txt.
#
# Do NOT assume a torch version here. Observed on Lambda images: the base image
# shipped torch 2.7.0+cu128, and `pip install sglang` silently upgraded it to
# 2.13.0+cu130 *after* this script's sanity print — so any hardcoded torchvision
# pin is matched to the wrong torch. (2026-08-25, cost ~20 min of a GPU session.)
#
# We never touch torch itself. Run from the repo root:
#   bash setup_lambda.sh

set -euo pipefail

PY="python"
PIP="$PY -m pip"

echo "==> torch sanity BEFORE (informational only — sglang may change this below)"
$PY -c "import torch; print('    torch', torch.__version__, '| cuda', torch.version.cuda)"

echo "==> 1/7 remove HF 'kernels' if present (breaks transformers kernel loading)"
if $PIP show kernels >/dev/null 2>&1; then
  $PIP uninstall -y kernels
else
  echo "    not installed, skipping"
fi

echo "==> 2/7 sglang (text inference backend for alpha_sweep / steering)"
$PIP install sglang --quiet

echo "==> 3/7 scientific stack"
$PIP install --upgrade numpy pandas scipy scikit-learn --quiet

echo "==> 4/7 accelerate (device_map='auto') + jinja2 (chat templates)"
$PIP install accelerate "jinja2>=3.1.0" --quiet

echo "==> 5/7 ConfAIde benchmark data (public; repo does not vendor it)"
mkdir -p data
BASE="https://raw.githubusercontent.com/skywalker023/confaide/main/benchmark"
for t in tier_1 tier_2a tier_2b tier_3 tier_4; do
  if [ -s "data/$t.txt" ]; then
    echo "    data/$t.txt present, skipping"
  else
    curl -fsSL -o "data/$t.txt" "$BASE/$t.txt"
    echo "    downloaded data/$t.txt"
  fi
done

echo "==> 6/7 install torchvision matched to whatever torch sglang left behind"
echo "    sglang hard-imports torchvision.io at startup; it must be importable,"
echo "    not absent (see header). We match it to the live torch build so the"
echo "    ABI is right, and verify by IMPORTING it, not by trusting pip show."
TORCH_VER=$($PY -c "import torch; print(torch.__version__)")
CUDA_TAG=$($PY -c "import torch; v=torch.version.cuda or ''; print('cu'+v.replace('.', '')[:3] if v else '')")
echo "    torch is actually $TORCH_VER (post-sglang), cuda tag: ${CUDA_TAG:-none}"

if $PY -c "import torchvision" >/dev/null 2>&1; then
  echo "    torchvision already importable, skipping install"
else
  # a stale dist-info/egg-info can make `pip show` claim "already satisfied"
  # for a torchvision whose package directory was removed by hand in an
  # earlier session (2026-08-26) — clear that metadata first or pip will
  # refuse to actually install anything.
  for meta in /usr/lib/python3/dist-packages/torchvision*.egg-info \
              /usr/lib/python3/dist-packages/torchvision*.dist-info; do
    if [ -e "$meta" ]; then
      sudo rm -rf "$meta"
      echo "    cleared stale metadata: $meta"
    fi
  done
  if [ -n "$CUDA_TAG" ]; then
    $PIP install --force-reinstall --no-deps torchvision \
      --index-url "https://download.pytorch.org/whl/$CUDA_TAG" --quiet \
      || $PIP install --force-reinstall --no-deps torchvision --quiet
  else
    $PIP install --force-reinstall --no-deps torchvision --quiet
  fi
fi
$PY -c "
import torchvision
print('    OK: torchvision', torchvision.__version__, 'importable')"

echo "==> 7/7 verify (Qwen2 + sglang import paths are clean)"
$PY - <<'EOF'
import torch, transformers
from transformers.models.qwen2 import Qwen2ForCausalLM   # the import that breaks
print("    OK: torch", torch.__version__, "| transformers", transformers.__version__)
print("    OK: cuda available:", torch.cuda.is_available())
print("    OK: Qwen2ForCausalLM importable")
import sglang   # exercises the torchvision.io.decode_jpeg import path
print("    OK: sglang", sglang.__version__, "importable")
EOF

echo
echo "==> setup complete. next (see docs/RUNBOOK.md for the full order):"
echo "    tmux new -s sessA   # detach with Ctrl+B D"
echo "    python scripts/relative_position_sweep_f.py 2>&1 | tee ~/e2b.log"
