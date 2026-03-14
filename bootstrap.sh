#!/bin/bash
set -e

echo "=== Bootstrapping Hey Aragorn Training ==="

# Require env vars
: "${GITHUB_TOKEN:?❌ Set GITHUB_TOKEN in RunPod environment variables}"
: "${GITHUB_REPO:?❌ Set GITHUB_REPO in RunPod environment variables}"

# Clone the repo
git config --global user.email "michaelpersson060@gmail.com"
git config --global user.name "Chungus"
git clone https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git /workspace/training
cd /workspace/training

# Run setup then training
bash setup.sh
python train.py 2>&1 | tee training.log
