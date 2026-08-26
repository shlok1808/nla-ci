#!/usr/bin/env bash
# setup_lambda.sh — provision a fresh Lambda A100 image for the nla-ci pipeline.
#
# Idempotent: safe to re-run on every new instance. Handles the two things that
# bite on a fresh spin-up:
#   1. A torchvision built against a different torch ABI than the one actually
#      installed -> `operator torchvision::nms does not exist` -> transformers
#      fails to import Qwen2ForCausalLM (image_utils imports torchvision.io).
#      We do not need torchvision at all: every experiment here is text-only.
#      So we REMOVE it instead of trying to match versions, and we remove BOTH
#      copies — the pip one and the apt one under /usr/lib/python3/dist-packages,
#      which shadows nothing and breaks everything if left behind. A missing
#      torchvision is skipped cleanly by is_vision_available(); a broken one is
#      an import-time crash.
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

echo "==> 6/7 remove torchvision entirely (LAST: sglang may have re-installed one)"
echo "    every experiment here is text-only; a torchvision built against a"
echo "    different torch ABI is an import-time crash, a missing one is skipped."
$PY -c "import torch; print('    torch is actually', torch.__version__, '(post-sglang)')"
# pip-installed copy (may live in ~/.local or site-packages)
while $PIP show torchvision >/dev/null 2>&1; do
  $PIP uninstall -y torchvision >/dev/null 2>&1 || break
  echo "    removed a pip torchvision"
done
# apt-installed copy — the one that bites, because uninstalling the pip copy
# just uncovers it and the traceback looks identical
for d in /usr/lib/python3/dist-packages/torchvision /usr/lib/python3*/dist-packages/torchvision; do
  if [ -d "$d" ]; then
    sudo mv "$d" "$d.disabled.$(date +%s)"
    echo "    disabled system torchvision at $d"
  fi
done
$PY -c "
import importlib.util as u
print('    torchvision importable:', u.find_spec('torchvision') is not None, '(want False)')"

echo "==> 7/7 verify (Qwen2 import path is clean)"
$PY - <<'EOF'
import torch, transformers
from transformers.models.qwen2 import Qwen2ForCausalLM   # the import that breaks
print("    OK: torch", torch.__version__, "| transformers", transformers.__version__)
print("    OK: cuda available:", torch.cuda.is_available())
print("    OK: Qwen2ForCausalLM importable")
EOF

echo
echo "==> setup complete. next (see docs/RUNBOOK.md for the full order):"
echo "    tmux new -s sessA   # detach with Ctrl+B D"
echo "    python scripts/relative_position_sweep_f.py 2>&1 | tee ~/e2b.log"
