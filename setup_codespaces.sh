#!/bin/bash
# Setup script for Hey Aragorn training in Codespaces

echo "🐍 Setting up Python 3.10 environment..."

# Install Python 3.10
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip

# Create virtual environment
python3.10 -m venv ~/wakeword-venv
source ~/wakeword-venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --quiet tensorflow==2.16.1 numpy==1.26.4 pyyaml scipy datasets mmap-ninja tqdm audiomentations webrtcvad-wheels ipykernel

# Install micro-wake-word
echo "🔧 Installing micro-wake-word..."
git clone https://github.com/kahrendt/microWakeWord.git ~/microWakeWord 2>/dev/null || true
pip install --quiet -e ~/microWakeWord

# Clone piper-sample-generator
git clone https://github.com/rhasspy/piper-sample-generator.git ~/piper-sample-generator 2>/dev/null || true

# Install kernel for Jupyter
python3 -m ipykernel install --user --name=wakeword --display-name "Python 3.10 (Wakeword)"

echo "✅ Setup complete!"
echo ""
echo "📝 To use this environment:"
echo "   1. In Jupyter, click the kernel selector (top right)"
echo "   2. Select 'Python 3.10 (Wakeword)'"
echo ""
echo "Or run: source ~/wakeword-venv/bin/activate"
