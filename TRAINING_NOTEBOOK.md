# Hey Aragorn Wake Word - Full Training Pipeline

This notebook runs the complete micro-wake-word training pipeline in one shot.

## Setup

```bash
# Install dependencies
pip install tensorflow==2.16.1 numpy==1.26.4 pyyaml scipy datasets mmap-ninja tqdm audiomentations webrtcvad-wheels

# Clone repos
git clone https://github.com/kahrendt/microWakeWord.git
git clone https://github.com/rhasspy/piper-sample-generator.git

# Download Piper voice model
mkdir -p piper-sample-generator/models
wget -O piper-sample-generator/models/en_US-libritts_r-medium.pt \
  https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt

# Install micro-wake-word
cd microWakeWord
pip install -e .
cd ..
```

## Step 1: Generate Wake Word Samples

```python
import subprocess

TARGET_WORD = "hey_air_uh_gorn"
NUM_SAMPLES = 1000

subprocess.run([
    "python3", "piper-sample-generator/generate_samples.py",
    TARGET_WORD,
    "--max-samples", str(NUM_SAMPLES),
    "--batch-size", "100",
    "--model", "piper-sample-generator/models/en_US-libritts_r-medium.pt",
    "--output-dir", "generated_samples",
], check=True)

print(f"Generated {NUM_SAMPLES} samples for '{TARGET_WORD}'")
```

## Step 2: Download Augmentation Data

```python
import os

# MIT RIR
os.makedirs("mit_rirs", exist_ok=True)
subprocess.run([
    "wget", "-q", "https://mcdermottlab.mit.edu/Reverb/IR_MIT_Survey.zip",
    "-O", "/tmp/ir_mit.zip"
], check=True)
subprocess.run(["unzip", "-q", "/tmp/ir_mit.zip", "-d", "mit_rirs"], check=True)

# AudioSet (using a subset)
os.makedirs("audioset_16k", exist_ok=True)
print("AudioSet download would go here - using placeholder")

# FMA (using a subset)  
os.makedirs("fma_16k", exist_ok=True)
print("FMA download would go here - using placeholder")
```

## Step 3: Generate Spectrograms

```python
import sys
sys.path.insert(0, 'microWakeWord')

from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
from mmap_ninja.ragged import RaggedMmap

clips = Clips(
    input_directory="generated_samples",
    file_pattern='*.wav',
    max_clip_duration_s=None,
    remove_silence=False,
    random_split_seed=10,
    split_count=0.1,
)

augmenter = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={
        "SevenBandParametricEQ": 0.1,
        "TanhDistortion": 0.1,
        "PitchShift": 0.1,
        "BandStopFilter": 0.1,
        "AddColorNoise": 0.1,
        "AddBackgroundNoise": 0.75,
        "Gain": 1.0,
        "RIR": 0.5,
    },
    impulse_paths=["mit_rirs"],
    background_paths=["fma_16k", "audioset_16k"],
    background_min_snr_db=-5,
    background_max_snr_db=10,
    min_jitter_s=0.195,
    max_jitter_s=0.205,
)

os.makedirs("generated_augmented_features/training", exist_ok=True)

spectrograms = SpectrogramGeneration(
    clips=clips,
    augmenter=augmenter,
    slide_frames=10,
    step_ms=10,
)

RaggedMmap.from_generator(
    out_dir="generated_augmented_features/training/wakeword_mmap",
    sample_generator=spectrograms.spectrogram_generator(split="train", repeat=2),
    batch_size=100,
    verbose=True,
)

print("Spectrograms generated!")
```

## Step 4: Download Negative Datasets

```python
import urllib.request

os.makedirs("negative_datasets", exist_ok=True)

base_url = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
files = ['dinner_party.zip', 'dinner_party_eval.zip', 'no_speech.zip', 'speech.zip']

for fname in files:
    url = base_url + fname
    print(f"Downloading {fname}...")
    urllib.request.urlretrieve(url, f"negative_datasets/{fname}")
    subprocess.run(["unzip", "-q", f"negative_datasets/{fname}", "-d", "negative_datasets"], check=True)
    os.remove(f"negative_datasets/{fname}")

print("Negative datasets downloaded!")
```

## Step 5: Train and Convert

```python
import yaml

config = {
    "window_step_ms": 10,
    "train_dir": "trained_models/wakeword",
    "spectrogram_length": 204,
    "stride": 3,
    "features": [
        {
            "features_dir": "generated_augmented_features",
            "sampling_weight": 2.0,
            "penalty_weight": 1.0,
            "truth": True,
            "truncation_strategy": "truncate_start",
            "type": "mmap",
        },
        {
            "features_dir": "negative_datasets/speech",
            "sampling_weight": 10.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "random",
            "type": "mmap",
        },
        {
            "features_dir": "negative_datasets/dinner_party",
            "sampling_weight": 10.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "random",
            "type": "mmap",
        },
        {
            "features_dir": "negative_datasets/no_speech",
            "sampling_weight": 5.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "random",
            "type": "mmap",
        },
        {
            "features_dir": "negative_datasets/dinner_party_eval",
            "sampling_weight": 0.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "split",
            "type": "mmap",
        },
    ],
    "training_steps": [10000],
    "positive_class_weight": [1],
    "negative_class_weight": [20],
    "learning_rates": [0.001],
    "batch_size": 128,
    "time_mask_max_size": [0],
    "time_mask_count": [0],
    "freq_mask_max_size": [0],
    "freq_mask_count": [0],
    "eval_step_interval": 500,
    "clip_duration_ms": 1500,
    "target_minimization": 0.9,
    "minimization_metric": None,
    "maximization_metric": "average_viable_recall",
}

with open("training_parameters.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print("Config saved!")
```

## Step 6: Run Training

```bash
python3 -m microwakeword.model_train_eval \
  --training_config=training_parameters.yaml \
  --train=1 \
  --restore_checkpoint 1 \
  --test_tf_nonstreaming 0 \
  --test_tflite_nonstreaming 0 \
  --test_tflite_nonstreaming_quantized 0 \
  --test_tflite_streaming 0 \
  --test_tflite_streaming_quantized 1 \
  mixednet \
  --pointwise_filters "64,64,64,64" \
  --repeat_in_block "1, 1, 1, 1" \
  --mixconv_kernel_sizes '[5], [7,11], [9,15], [23]' \
  --residual_connection "0,0,0,0" \
  --first_conv_filters 32 \
  --first_conv_kernel_size 5 \
  --stride 3
```

## Output

The model will be at:
`trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`
