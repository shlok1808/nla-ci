#!/usr/bin/env bash
# setup_lambda.sh — provision a fresh Lambda A100 image for the nla-ci pipeline.
#
# Idempotent: safe to re-run on every new instance. Handles the two things that
# bite on a fresh spin-up:
#   1. `pip install sglang` drags in a torchvision built for the wrong torch ABI
#      (image ships torch 2.11.0+cu130) -> `operator torchvision::nms does not
#      exist` -> transformers fails to import Qwen2ForCausalLM. We pin the matched
#      torchvision LAST so nothing re-clobbers it.
#   2. The repo does not vendor the ConfAIde benchmark; scripts need data/*.txt.
#
# We never touch torch itself. Run from the repo root:
#   bash setup_lambda.sh

set -euo pipefail

PY="python"
PIP="$PY -m pip"

echo "==> torch sanity (expect 2.11.0+cu130; we never reinstall torch)"
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

echo "==> 6/7 torchvision matched to torch 2.11 / cu130 (LAST: sglang installs a mismatched one)"
$PIP install --force-reinstall --no-deps "torchvision==0.26.0" \
  --index-url https://download.pytorch.org/whl/cu130

echo "==> 7/7 verify (torchvision::nms registers + Qwen2 import path is clean)"
$PY - <<'EOF'
import torch, torchvision
from torchvision.ops import nms          # triggers the C++ op that was missing
from transformers.models.qwen2 import Qwen2ForCausalLM
print("    OK: torch", torch.__version__, "| torchvision", torchvision.__version__)
print("    OK: torchvision::nms registered, Qwen2ForCausalLM importable")
EOF

echo
echo "==> setup complete. next:"
echo "    tmux new -s sweep   # detach with Ctrl+B D"
echo "    python scripts/position_sweep_f.py"
