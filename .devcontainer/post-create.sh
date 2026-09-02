#!/usr/bin/env bash
# Runs once, when the Codespace / dev container is created.
#
# Deliberately small: CPU-only wheels, no CUDA, no model downloads. On a 4-core
# Codespace this finishes in roughly two minutes.
set -euo pipefail

echo "==> installing Python dependencies (CPU-only torch)"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "==> building the datasets into data/"
python scripts/prepare_data.py

echo
echo "==> checking the pre-baked results are present"
if compgen -G "checkpoints/*.pt" > /dev/null; then
  ls -la checkpoints artifacts
else
  echo "    no checkpoints found - run 'python scripts/pretrain_all.py' (~6 minutes)"
fi

echo
echo "Ready. Open notebooks/00_setup_check.ipynb to begin."
