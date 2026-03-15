#!/usr/bin/env bash
# setup_local.sh — Set up training environment on any Linux machine (GPU optional)
# Use this instead of RunPod. For RunPod, use setup.sh instead.
set -e

echo "=== microWakeWord Local Setup ==="
echo "This installs everything needed to run train.py outside of RunPod."
echo ""

# ── System packages ──────────────────────────────────────────────────────────
echo "[1/4] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv git wget unzip ffmpeg

# ── Working directory ────────────────────────────────────────────────────────
WORK_DIR="${1:-$HOME/wakeword-training}"
mkdir -p "$WORK_DIR"
echo "[2/4] Working directory: $WORK_DIR"

# ── Python venv ──────────────────────────────────────────────────────────────
echo "[3/4] Creating Python virtual environment..."
python3 -m venv "$WORK_DIR/venv"
source "$WORK_DIR/venv/bin/activate"

pip install --upgrade pip -q

# ── Python packages ──────────────────────────────────────────────────────────
echo "[4/4] Installing Python packages..."
pip install -q tensorflow==2.18.0
pip install -q numpy pyyaml mmap-ninja huggingface-hub soundfile librosa

# GPU users: install cuDNN
if python3 -c "import tensorflow as tf; exit(0 if tf.config.list_physical_devices('GPU') else 1)" 2>/dev/null; then
    echo ""
    echo "GPU detected — installing cuDNN..."
    pip install -q "nvidia-cudnn-cu12>=9.3,<10"
else
    echo ""
    echo "No GPU detected — CPU-only mode. Training will be slower but functional."
fi

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "To train:"
echo "  source $WORK_DIR/venv/bin/activate"
echo "  cd $REPO_DIR"
echo "  WORK_DIR=$WORK_DIR python3 train.py"
echo ""
