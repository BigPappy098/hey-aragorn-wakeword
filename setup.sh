#!/usr/bin/env bash
# setup.sh — Install base dependencies for train.py
# Works on RunPod (direct pip) and local machines (auto-creates venv if needed).
set -e

echo "=== microWakeWord Setup ==="

WORK_DIR="${WORK_DIR:-/workspace/training}"
cd "$WORK_DIR"

# Detect if we're on a GPU machine (RunPod, etc.)
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

# Try direct pip first (works on RunPod / Docker as root).
# Fall back to venv if pip is blocked (externally-managed Python).
if pip install --upgrade pip -q 2>/dev/null && \
   pip install -q "tensorflow>=2.18.0" numpy pyyaml mmap-ninja huggingface-hub tensorboard 2>/dev/null; then
    echo "Base dependencies installed (system Python)."
else
    echo "System pip blocked — creating venv..."
    python3 -m venv "$WORK_DIR/venv"
    source "$WORK_DIR/venv/bin/activate"
    pip install --upgrade pip -q
    pip install -q "tensorflow>=2.18.0" numpy pyyaml mmap-ninja huggingface-hub tensorboard
    echo "Base dependencies installed (venv)."
    echo ""
    echo "NOTE: Activate the venv before running train.py:"
    echo "  source $WORK_DIR/venv/bin/activate"
fi

# System packages needed for audio processing and piper TTS
echo "Installing system packages (ffmpeg, espeak-ng)..."
apt-get install -y -q ffmpeg espeak-ng libespeak-ng-dev 2>/dev/null || \
    echo "WARNING: Could not install system packages"

# Pre-install slow packages so train.py doesn't have to
echo "Installing additional dependencies..."
pip install -q soundfile librosa scipy "audiomentations>=0.35.0" webrtcvad "datasets<3" tqdm onnxruntime

# PyTorch: only needed for piper-tts (sample generation, not training).
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

# On GPU machines, ensure cuDNN >= 9.3 is available (TF 2.18 requirement)
if [ "$HAS_GPU" = true ]; then
    echo "Ensuring cuDNN >= 9.3 for TensorFlow 2.18..."
    pip install -q 'nvidia-cudnn-cu12>=9.3,<10' 2>/dev/null || true
fi

echo ""
echo "Setup complete! Run: python train.py"
