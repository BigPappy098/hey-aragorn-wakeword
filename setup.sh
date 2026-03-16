#!/usr/bin/env bash
# setup.sh — RunPod manual setup (if not using bootstrap.sh)
# Creates a venv and installs base dependencies so train.py can run.
set -e

echo "=== microWakeWord RunPod Setup ==="

WORK_DIR="${WORK_DIR:-/workspace/training}"
cd "$WORK_DIR"

# ── Create venv (required on newer images with externally-managed Python) ────
if [ ! -d "$WORK_DIR/venv" ]; then
    echo "[1/2] Creating Python virtual environment..."
    python3 -m venv "$WORK_DIR/venv"
else
    echo "[1/2] Venv already exists, skipping creation."
fi

source "$WORK_DIR/venv/bin/activate"
pip install --upgrade pip -q

# ── Install base dependencies that train.py needs at import time ─────────────
echo "[2/2] Installing base Python packages..."
pip install -q "tensorflow>=2.18.0" numpy pyyaml mmap-ninja huggingface-hub

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  source $WORK_DIR/venv/bin/activate"
echo "  python train.py"
echo ""
