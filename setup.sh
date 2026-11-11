#!/usr/bin/env bash
# One-time setup for Drum Queue (same approach as MIDI Queue: uv + Python 3.11, no Homebrew).
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (a fast Python installer)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "Creating Python 3.11 environment in .venv ..."
uv venv --python 3.11 --seed .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing audio-separator (PyTorch + ONNX), Flask and audio libraries ..."
uv pip install -r requirements.txt

python - <<'PY'
import torch
print("Apple GPU (MPS) available:", torch.backends.mps.is_available())
PY
echo
echo "Done. Start the app with:  ./run.sh"
echo "The first song downloads the Demucs model (~300 MB). DrumSep is reused from Xport Studio if it's there."
