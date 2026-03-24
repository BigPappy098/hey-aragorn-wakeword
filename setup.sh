#!/usr/bin/env bash
# setup.sh — Install all dependencies for train.py into a persistent venv.
# On RunPod the venv lives on the volume (/workspace/) so packages survive pod restarts.
set -e

echo "=== microWakeWord Setup ==="

WORK_DIR="${WORK_DIR:-/workspace/training}"
cd "$WORK_DIR"

# ── GPU detection ────────────────────────────────────────────────────────────
# Some RunPod containers have nvidia-smi issues (exit 2) even with a working GPU,
# so also check for /dev/nvidia0 as a fallback.
HAS_GPU=false
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    HAS_GPU=true
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "GPU detected: ${GPU_NAME:-unknown}"
elif [ -e /dev/nvidia0 ]; then
    HAS_GPU=true
    echo "GPU detected: /dev/nvidia0 exists (nvidia-smi unavailable)"
fi

# ── System packages ──────────────────────────────────────────────────────────
echo "Installing system packages (ffmpeg, espeak-ng)..."
apt-get install -y -q ffmpeg espeak-ng libespeak-ng-dev 2>/dev/null || \
    echo "WARNING: Could not install system packages (apt-get failed). ffmpeg and espeak-ng must be available."

# ── Python venv (persistent on the volume) ───────────────────────────────────
# Always create the venv inside WORK_DIR so it lives on the RunPod volume,
# not the container disk. This means packages survive pod stop/restart.
VENV_DIR="$WORK_DIR/venv"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Existing venv found at $VENV_DIR — reusing."
    source "$VENV_DIR/bin/activate"
else
    echo "Creating Python venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip -q
fi

# ── Core dependencies ────────────────────────────────────────────────────────
echo "Installing core dependencies..."
pip install -q "tensorflow>=2.18.0" keras ai_edge_litert
pip install -q numpy pyyaml mmap-ninja huggingface-hub tensorboard

echo "Installing audio & ML dependencies..."
pip install -q soundfile librosa scipy scikit-learn "audiomentations>=0.35.0" webrtcvad "datasets<3" tqdm onnxruntime

# ── PyTorch (needed for piper-tts sample generation) ─────────────────────────
# Install the build matching the driver's CUDA version so GPU is available.
if [ "$HAS_GPU" = true ]; then
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' || echo "")
    if [ -n "$CUDA_VER" ]; then
        # Convert "12.4" → "cu124"
        CUDA_TAG="cu$(echo "$CUDA_VER" | tr -d '.')"
        # Pick best available PyTorch wheel (cu118, cu121, cu124, cu126)
        case "$CUDA_TAG" in
            cu118|cu119)              PT_INDEX="cu118" ;;
            cu120|cu121)              PT_INDEX="cu121" ;;
            cu122|cu123|cu124|cu125)  PT_INDEX="cu124" ;;
            *)                        PT_INDEX="cu126" ;;
        esac
        echo "GPU machine — CUDA $CUDA_VER → installing PyTorch ($PT_INDEX)..."
        pip install -q torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$PT_INDEX"
    else
        echo "GPU machine — installing PyTorch (default index)..."
        pip install -q torch torchvision torchaudio
    fi
else
    echo "No GPU — installing CPU-only PyTorch..."
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
pip install -q piper-tts

# ── cuDNN (TF 2.18 requirement) ─────────────────────────────────────────────
if [ "$HAS_GPU" = true ]; then
    echo "Ensuring cuDNN >= 9.3 for TensorFlow 2.18..."
    pip install -q 'nvidia-cudnn-cu12>=9.3,<10' 2>/dev/null || true
fi

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "  The venv is at: $VENV_DIR"
echo "  It lives on the volume, so it survives pod restarts."
echo ""
echo "  To train, run:"
echo "    source $VENV_DIR/bin/activate"
echo "    python -u train.py 2>&1 | tee training.log"
echo ""
