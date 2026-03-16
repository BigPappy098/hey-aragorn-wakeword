#!/bin/bash
set -e

echo "=== Bootstrapping Hey Aragorn Training ==="

# Require env vars
: "${GITHUB_TOKEN:?❌ Set GITHUB_TOKEN in RunPod environment variables}"
: "${GITHUB_REPO:?❌ Set GITHUB_REPO in RunPod environment variables}"

# Clone the repo
git config --global user.email "runpod@training.bot"
git config --global user.name "RunPod Training Bot"
git clone "https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git" /workspace/training
cd /workspace/training

# ── Install base dependencies that train.py needs at import time ─────────────
# On RunPod (Docker as root), pip works directly — no venv needed.
# If pip fails due to externally-managed-environment, fall back to a venv.
if pip install --upgrade pip -q 2>/dev/null && \
   pip install -q "tensorflow>=2.18.0" numpy pyyaml mmap-ninja huggingface-hub 2>/dev/null; then
    echo "[bootstrap] Base dependencies installed (system Python)."
else
    echo "[bootstrap] System pip blocked — creating venv..."
    python3 -m venv /workspace/training/venv
    source /workspace/training/venv/bin/activate
    pip install --upgrade pip -q
    pip install -q "tensorflow>=2.18.0" numpy pyyaml mmap-ninja huggingface-hub
    echo "[bootstrap] Base dependencies installed (venv)."
fi

# ── Run training (train.py self-installs remaining deps as needed) ───────────
python train.py 2>&1 | tee training.log
