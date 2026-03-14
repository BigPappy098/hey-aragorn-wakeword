# microWakeWord Custom Trainer

Train a custom wake word model for ESPHome and Home Assistant using microWakeWord. No coding required.

---

## What This Does

1. Generates synthetic voice samples of your wake word using Piper TTS
2. Augments them with background noise and room acoustics
3. Trains a ~100KB TFLite model using the microWakeWord architecture
4. Pushes the .tflite and .json files to your GitHub repo, ready for ESPHome

---

## Minimum System Requirements

| Component     | Minimum                  | Recommended              |
|---------------|--------------------------|--------------------------|
| GPU           | RTX 3090 (24 GB VRAM)    | RTX 4090 (24 GB VRAM)    |
| CUDA          | 12.x                     | 12.8                     |
| RAM           | 16 GB                    | 32 GB                    |
| Disk          | 50 GB                    | 150 GB (network volume)  |
| Python        | 3.10                     | 3.11                     |
| TensorFlow    | 2.18.0-gpu               | 2.18.0-gpu               |

Training requires a GPU with at least 20 GB VRAM. CPU-only is not supported.

---

## Prerequisites

- A RunPod account (https://runpod.io)
- A GitHub account
- A GitHub Personal Access Token (PAT) with Contents: Read and Write on your repo
  Generate at: GitHub > Settings > Developer Settings > Personal Access Tokens > Fine-grained tokens

---

## Setup

### 1. Fork or clone this repo to your GitHub account

### 2. Create a RunPod pod

- Template: tensorflow/tensorflow:2.18.0-gpu
- GPU: RTX 4090 (or RTX 3090)
- Volume: 150 GB network volume mounted at /workspace
- Environment Variables (set in RunPod dashboard > Edit Pod):

| Key           | Value                          |
|---------------|--------------------------------|
| GITHUB_TOKEN  | Your GitHub PAT (github_pat_...) |
| GITHUB_REPO   | yourusername/your-repo-name    |

### 3. SSH into the pod

    ssh root@<pod-ip> -p <port>

### 4. Start a tmux session (keeps training running if you disconnect)

    tmux new-session -s training

### 5. Clone this repo

    git clone https://github.com/YOURUSERNAME/YOURREPO.git /workspace/training
    cd /workspace/training

### 6. Run setup

    python setup.py

You should see two green checkmarks when setup is complete.

---

## Training

    python train.py 2>&1 | tee training.log

You will be prompted for:

    Enter wake word phonetically (e.g. hey_air_uh_gorn):
    Number of voice samples [default: 1000, better quality: 2000-5000]:
    Training steps [default: 20000, better quality: 20000-30000]:

### Tips for choosing your wake word

- Spell it out phonetically using underscores between syllables
- Longer, more unique phrases reduce false positives
- Examples: hey_air_uh_gorn, ok_computer, hey_jarvis

### Training time estimates (RTX 4090)

| Steps   | Approx. Time |
|---------|--------------|
| 10,000  | ~1 hour      |
| 20,000  | ~2 hours     |
| 30,000  | ~3 hours     |

### Detach from tmux while training

Press Ctrl+B then D to detach. Reattach later with:

    tmux attach -t training

---

## Output

When training completes, two files are automatically pushed to your GitHub repo:

    models/
      your_wake_word.tflite   <- the model
      your_wake_word.json     <- ESPHome metadata

The script prints the exact URL to use in ESPHome when done:

    ESPHome model URL: https://raw.githubusercontent.com/YOURUSERNAME/YOURREPO/main/models/your_wake_word.json

---

## Using in ESPHome

Add to your ESPHome device config:

    micro_wake_word:
      models:
        - model: https://raw.githubusercontent.com/YOURUSERNAME/YOURREPO/main/models/your_wake_word.json
      on_wake_word_detected:
        - voice_assistant.start:

---

## Troubleshooting

### cuDNN version mismatch error
Handled automatically by train.py. If needed manually:

    export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")/lib:$LD_LIBRARY_PATH

### Training fails at Step 12 with FailedPreconditionError
    pip install nvidia-cudnn-cu12==9.3.0.75

### ZeroDivisionError at the end of training
This is a known bug in the microWakeWord eval script. It does NOT affect the model.
Training completed successfully. The fix is already patched in train.py.

### Checking if the model was saved
    find /workspace/training -name "*.tflite"

### Viewing training progress
    tail -f training.log

---

## What Good Training Looks Like

By the final validation batch you should see:

| Metric    | Target |
|-----------|--------|
| Accuracy  | > 99%  |
| Recall    | > 95%  |
| Precision | > 95%  |
| Loss      | < 0.01 |

---

## Repo Structure

    /
    |-- setup.py        <- Install dependencies, verify GPU
    |-- train.py        <- Main training script (interactive)
    |-- README.md       <- This file
    |-- models/
        |-- *.tflite    <- Trained model files
        |-- *.json      <- ESPHome metadata files

---

## Credits

- microWakeWord by Kevin Ahrendt (https://github.com/kahrendt/microWakeWord)
- piper-sample-generator by rhasspy (https://github.com/rhasspy/piper-sample-generator)
- ESPHome micro_wake_word (https://esphome.io/components/micro_wake_word.html)
'''

with open('/workspace/training/README.md', 'w') as f:
    f.write(content)
print("Written.")
