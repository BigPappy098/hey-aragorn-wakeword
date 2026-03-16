#!/usr/bin/env bash
# setup.sh — Install base dependencies for train.py
# Works on RunPod (direct pip) and local machines (auto-creates venv if needed).
set -e

echo "=== microWakeWord Setup ==="

WORK_DIR="${WORK_DIR:-/workspace/training}"
cd "$WORK_DIR"

# Try direct pip first (works on RunPod / Docker as root).
# Fall back to venv if pip is blocked (externally-managed Python).
if pip install --upgrade pip -q 2>/dev/null && \
   pip install -q "tensorflow>=2.18.0" numpy pyyaml mmap-ninja huggingface-hub 2>/dev/null; then
    echo "Base dependencies installed (system Python)."
else
    echo "System pip blocked — creating venv..."
    python3 -m venv "$WORK_DIR/venv"
    source "$WORK_DIR/venv/bin/activate"
    pip install --upgrade pip -q
    pip install -q "tensorflow>=2.18.0" numpy pyyaml mmap-ninja huggingface-hub
    echo "Base dependencies installed (venv)."
    echo ""
    echo "NOTE: Activate the venv before running train.py:"
    echo "  source $WORK_DIR/venv/bin/activate"
fi

echo ""
echo "Setup complete! Run: python train.py"
