#!/bin/bash
set -e
echo "=== Installing system deps ==="
apt-get update -qq && apt-get install -y -q espeak-ng libespeak-ng-dev wget unzip git

echo "=== Pinning numpy/scipy ==="
pip install --force-reinstall --no-cache-dir numpy==1.26.4 scipy==1.13.1

echo "=== Removing conflicting packages ==="
pip uninstall -y jax jaxlib tensorstore tensorflow-decision-forests tensorflow-text \
  opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true

echo "=== Installing Python packages ==="
pip install piper-tts
pip install tensorflow==2.18.0 protobuf==4.25.3 ml-dtypes==0.3.2
pip install onnxruntime pyyaml datasets mmap-ninja tqdm audiomentations \
  webrtcvad-wheels huggingface_hub

echo "=== Verifying GPU ==="
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'TF {tf.__version__} | GPUs: {gpus}')
assert gpus, '❌ No GPU found!'
print('✅ GPU confirmed')
"
echo "✅ Setup complete"
