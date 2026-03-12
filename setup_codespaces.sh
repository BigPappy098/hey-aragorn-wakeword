#!/bin/bash
# Setup script for Hey Aragorn training in Codespaces (works with Ubuntu 24.04)

echo "🐍 Setting up Python environment..."

# Use pyenv to install Python 3.10
echo "📥 Installing pyenv..."
curl -s https://pyenv.run | bash

# Add pyenv to PATH
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"

# Install Python 3.10
echo "⬇️ Installing Python 3.10 (this takes a few minutes)..."
pyenv install 3.10.14
pyenv global 3.10.14

# Verify
python3 --version

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv ~/wakeword-venv
source ~/wakeword-venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --quiet tensorflow==2.16.1 numpy==1.26.4 pyyaml scipy datasets mmap-ninja tqdm audiomentations webrtcvad-wheels ipykernel

# Install micro-wake-word
echo "🔧 Installing micro-wake-word..."
if [ ! -d "$HOME/microWakeWord" ]; then
    git clone https://github.com/kahrendt/microWakeWord.git ~/microWakeWord
fi
pip install --quiet -e ~/microWakeWord

# Clone piper-sample-generator
if [ ! -d "$HOME/piper-sample-generator" ]; then
    git clone https://github.com/rhasspy/piper-sample-generator.git ~/piper-sample-generator
fi

# Install kernel for Jupyter
python3 -m ipykernel install --user --name=wakeword --display-name "Python 3.10 (Wakeword)"

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 To use this environment:"
echo "   1. In Jupyter, click the kernel selector (top right)"
echo "   2. Select 'Python 3.10 (Wakeword)'"
echo ""
echo "Or run: source ~/wakeword-venv/bin/activate"
