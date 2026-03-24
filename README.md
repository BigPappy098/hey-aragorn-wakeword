# microWakeWord Custom Trainer

Train a custom wake word model for [ESPHome](https://esphome.io) and [Home Assistant](https://www.home-assistant.io/) using [microWakeWord](https://github.com/TaterTotterson/micro-wake-word). No ML experience required.

You provide a phonetic wake word (like `hey_air_uh_gorn`), and this project generates thousands of synthetic voice samples, augments them with real-world background noise and room acoustics, trains a tiny (~100 KB) TFLite model, and pushes it to your GitHub repo — ready to drop into an ESPHome config.

Based on [TaterTotterson's microWakeWord-Trainer](https://github.com/TaterTotterson/microWakeWord-Trainer-Nvidia-Docker) (minus the web app — this is terminal-only).

---

## Quick Start (RunPod)

If you just want to get going fast, here's the short version. Detailed instructions below.

```bash
# On RunPod (after creating pod — see setup below):
apt-get update && apt-get install -y git wget curl unzip
git clone https://github.com/YOURUSERNAME/YOURREPO.git /workspace/training
cd /workspace/training
bash setup.sh
python -u train.py 2>&1 | tee training.log
```

---

## Prerequisites

### 1. Fork this repo

Go to this repo on GitHub and click **Fork** (top right). This gives you your own copy where trained models will be pushed.

### 2. Create a GitHub Personal Access Token (PAT)

1. Go to **GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens**
   (direct link: https://github.com/settings/personal-access-tokens/new)
2. Give it a name like `wakeword-training`
3. Under **Repository access**, select **Only select repositories** and pick your forked repo
4. Under **Permissions → Repository permissions**, set **Contents** to **Read and write**
5. Click **Generate token** and copy it — you'll need it during setup

---

## Setup Option A: RunPod (Cloud GPU) — Recommended

### Step 1: Create a RunPod Account

Go to [runpod.io](https://runpod.io) and create an account. Add a payment method (training costs ~$1-3).

### Step 2: Deploy a GPU Pod

1. Click **GPU Cloud** → **Deploy**
2. Select a GPU:

| GPU | VRAM | Cost | Training Time (~20k steps) |
|---|---|---|---|
| **RTX 4090** | 24 GB | ~$0.40/hr | ~2 hours |
| RTX 3090 | 24 GB | ~$0.30/hr | ~2-3 hours |
| RTX A5000 | 24 GB | ~$0.30/hr | ~2-3 hours |
| Any NVIDIA GPU | 8+ GB | Varies | Works, just slower |

3. Click **Deploy On Demand** (or Spot for cheaper, but it can be interrupted)

### Step 3: Configure the Pod

On the pod configuration screen:

| Setting | Value |
|---|---|
| **Container Image** | `tensorflow/tensorflow:latest-gpu` |
| **Container Disk** | 20 GB (minimum) |
| **Volume Disk** | **150 GB** (needed for augmentation datasets) |
| **Volume Mount Path** | `/workspace` |

> **Important — Volume Size:** The volume must be at least **100 GB**. The background noise datasets (WHAM, CHiME, FMA, AudioSet) need ~30 GB to download and ~12 GB converted. The negative datasets need another ~3 GB. Generated samples and training artifacts need ~5-10 GB.

> **Important — Volume vs Container Disk:** The **container disk** gets wiped every time you stop the pod. The **volume disk** (`/workspace`) persists across stops. Everything in this guide is cloned and run inside `/workspace/` so your datasets, models, and repo survive pod restarts. **Do not** clone or work outside of `/workspace/` or you'll lose everything when the pod stops.

4. Click **Environment Variables** and add:

| Variable | Value |
|---|---|
| `GITHUB_TOKEN` | Your PAT from above (`ghp_...` or `github_pat_...`) |
| `GITHUB_REPO` | `yourusername/your-repo-name` (e.g. `BigPappy098/hey-aragorn-wakeword`) |

5. Click **Deploy**

### Step 4: Connect to the Pod

Once the pod shows **Running**:

1. Click **Connect**
2. Choose **SSH** or **Web Terminal**

**If using SSH:**
```bash
ssh root@<pod-ip> -p <port>
```
(RunPod shows the SSH command on the Connect page)

**If using Web Terminal:** Just click the button — it opens a terminal in your browser.

### Step 5: Install System Packages

```bash
apt-get update && apt-get install -y git wget curl unzip
```

### Step 6: Clone Your Repo

```bash
git clone https://github.com/YOURUSERNAME/YOURREPO.git /workspace/training
cd /workspace/training
```

### Step 7: Run Setup

```bash
bash setup.sh
```

This installs TensorFlow, PyTorch, piper-tts, and all dependencies. Takes ~5 minutes.

### Step 8: Start Training

```bash
python -u train.py 2>&1 | tee training.log
```

The script is fully interactive — it will ask you for:
1. Wake word (phonetic spelling)
2. Number of samples (default: 10,000)
3. Training steps (default: 20,000)

Then it runs automatically. See [Training Walkthrough](#training-walkthrough) below for details.

### Step 9: Keeping the Pod Running

Training keeps running even if you close the browser tab — RunPod's web terminal stays alive in the background.

**Important:** Your repo and datasets live on the volume (`/workspace/`) and survive pod restarts. But `setup.sh` installs Python packages to the **container disk**, which gets wiped when the pod stops. To avoid re-running setup after a disconnect, run this in a separate terminal (or after training finishes):

```bash
sleep infinity
```

This keeps the pod from going idle and stopping. When you're truly done and want to stop billing: Go to RunPod dashboard → click **Stop** on your pod. Next time you start it, you'll need to re-run `bash setup.sh` (takes ~5 min) but your datasets and repo will still be there.

---

## Setup Option B: Local Machine or VPS

Works on any Ubuntu/Debian machine — with or without a GPU.

### 1. Clone your repo

```bash
git clone https://github.com/YOURUSERNAME/YOURREPO.git
cd YOURREPO
```

### 2. Run the local setup script

```bash
bash setup_local.sh
```

This installs system packages, creates a Python virtual environment, and installs all dependencies. It will ask where to put training data — press Enter for the current directory.

### 3. Configure your `.env` file

The setup script creates a `.env` file from the template. Edit it:

```bash
nano .env
```

Fill in:
```
WORK_DIR=~/wakeword-training
GITHUB_REPO=yourusername/your-repo-name
GITHUB_TOKEN=github_pat_xxxxxxxxxxxx
```

> **Security:** The `.env` file is gitignored — your token stays local.

### 4. Activate the venv and train

```bash
source ~/wakeword-training/venv/bin/activate
python3 -u train.py 2>&1 | tee training.log
```

---

## Training Walkthrough

Here's what happens when you run `train.py`.

### 1. Configure Your Wake Word

```
Enter wake word phonetically (e.g. hey_air_uh_gorn): hey_air_uh_gorn
Number of voice samples [default: 10000, better quality: 20000-50000]: 10000
Training steps [default: 20000, better quality: 20000-30000]: 20000
```

**Tips for phonetic spelling:**
- Use underscores between syllables: `hey_air_uh_gorn`, not `heyaragorn`
- Spell it how it **sounds**, not how it's written
- Longer, more unique phrases reduce false activations
- Examples: `hey_air_uh_gorn`, `ok_computer`, `hey_jar_vis`

### 2. Pronunciation Preview

The script generates one sample, pushes a `.wav` to GitHub, and asks if it sounds right:

```
Preview uploaded to: https://github.com/you/repo/blob/main/models/preview_hey_air_uh_gorn.wav
Does the pronunciation sound correct? [y/n]:
```

**Listen carefully.** If it sounds wrong, type `n` and re-run with a different phonetic spelling.

### 3. Real Recordings (Optional but Recommended)

If you have recordings of people saying the wake word, place them in `real_recordings/` before running.

**Two ways to provide recordings:**

| Method | How |
|---|---|
| **Long recording** | One file per person with 10-50 repetitions, 1-2 sec pauses. The script auto-splits them. |
| **Individual clips** | Name them `speaker01_take01.wav`, `speaker01_take02.wav`, etc. Short clips (< 4 sec) are used as-is. |

**Tips:**
- Multiple speakers improve model quality
- Phone or laptop mic is fine
- Supported formats: `.wav`, `.mp3`, `.m4a`, `.flac`
- The script mixes 50/50 synthetic + real recordings

### 4. Automatic Pipeline

From here, everything is automatic:

1. **Generate synthetic samples** using [Piper TTS](https://github.com/TaterTotterson/piper-sample-generator) (10,000+ WAV files)
2. **Download augmentation datasets** (~30 GB total, cached for future runs):
   - MIT Room Impulse Responses (reverb simulation)
   - WHAM (urban environment noise)
   - CHiME-Home (domestic environment recordings)
   - FMA xsmall (music background)
   - AudioSet balanced (diverse real-world sounds)
3. **Convert all audio to 16k mono WAV** (archives deleted after to save space)
4. **Generate spectrograms** with full augmentation (background noise, reverb, pitch shift, EQ, distortion)
5. **Download negative datasets** from Hugging Face (~3 GB, pre-computed features)
6. **Write training config** (`training_parameters.yaml`)

> **First run takes longer** due to dataset downloads. Subsequent runs reuse cached datasets.

### 5. Training

Training runs automatically. Progress is logged every 500 steps.

**Training parameters (matching TaterTotterson's working reference):**

| Parameter | Value |
|---|---|
| Model architecture | MixedNet (64 pointwise filters, 4 blocks) |
| Stride | 2 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Positive class weight | 1 |
| Negative class weight | 20 |
| Background noise SNR | 5-10 dB |

**Estimated training times:**

| GPU | 20k steps | 30k steps |
|---|---|---|
| RTX 4090 | ~2 hours | ~3 hours |
| RTX 3090 | ~2-3 hours | ~4 hours |
| 8-12 GB GPU | ~3-5 hours | ~5-8 hours |
| CPU-only | Many hours | Not recommended |

**CLI flags:**

| Flag | Effect |
|---|---|
| `--dry-run` | Validate the pipeline without training |
| `--force-train` | Override low-resource detection and attempt training anyway |

### 6. Output

When training completes, two files are pushed to your GitHub repo:

```
models/
  hey_air_uh_gorn.tflite    <- the trained model (~100 KB)
  hey_air_uh_gorn.json      <- ESPHome metadata
```

The script prints the ESPHome URL:
```
ESPHome URL: https://raw.githubusercontent.com/YOURUSERNAME/YOURREPO/main/models/hey_air_uh_gorn.json
```

---

## Using in ESPHome

Add this to your ESPHome device YAML:

```yaml
micro_wake_word:
  models:
    - model: https://raw.githubusercontent.com/YOURUSERNAME/YOURREPO/main/models/hey_air_uh_gorn.json
  on_wake_word_detected:
    - voice_assistant.start:
```

Flash the device and it will respond to your custom wake word.

---

## Retraining / Improving Your Model

If the model doesn't detect well enough, try these in order:

1. **Add real recordings** — Record yourself (and others) saying the wake word. Multiple speakers help a lot.
2. **Increase samples** — Try 20,000-50,000 synthetic samples instead of 10,000.
3. **Increase training steps** — Try 30,000-40,000 steps.
4. **Adjust the phonetic spelling** — If TTS pronunciation is off, the model learns the wrong sound.

To retrain, just re-run `python -u train.py 2>&1 | tee training.log` — the script cleans up previous artifacts automatically (but keeps downloaded datasets to save time).

---

## Troubleshooting

### cuDNN version mismatch

TensorFlow 2.18+ requires cuDNN 9.3+. The script handles this automatically. If you still see errors:

```bash
export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")/lib:$LD_LIBRARY_PATH
```

### "No GPU detected"

```bash
nvidia-smi          # should show your GPU
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

On RunPod this is handled by the container image. Locally you may need to install CUDA.

### Out of disk space

The augmentation datasets need ~30 GB to download and ~12 GB once converted. Make sure your RunPod volume is at least 100 GB. Archives are automatically deleted after extraction.

### Machine freezes during training

On machines with < 12 GB RAM and no GPU, training is skipped by default. Use `--force-train` to override (expect slow performance).

### PyTorch CUDA mismatch warning

The script auto-detects your CUDA version and installs the matching PyTorch build. If you see warnings about CUDA version mismatch, the script already handled it — it's just a pre-install leftover.

### Training seems stuck at step 0

The first step takes longer because TensorFlow compiles GPU kernels. Give it a few minutes.

### Checking if the model was saved

```bash
find /workspace/training -name "*.tflite" 2>/dev/null
# or locally:
find . -name "*.tflite"
```

---

## Repo Structure

```
/
├── train.py              <- Main training script (interactive, fully automatic)
├── setup.sh              <- RunPod setup script
├── setup_local.sh        <- Local/VPS setup script (creates venv)
├── .env.example          <- Template for local env vars
├── .gitignore            <- Keeps secrets, logs, and artifacts out of git
├── real_recordings/      <- Drop your voice clips here (optional)
├── models/               <- Trained models are saved and pushed here
│   ├── *.tflite
│   └── *.json
└── README.md
```

---

## Credits

- [microWakeWord](https://github.com/TaterTotterson/micro-wake-word) (TaterTotterson's fork)
- [piper-sample-generator](https://github.com/TaterTotterson/piper-sample-generator) (TaterTotterson's fork)
- [microWakeWord-Trainer](https://github.com/TaterTotterson/microWakeWord-Trainer-Nvidia-Docker) by TaterTotterson (reference implementation)
- [microWakeWord](https://github.com/kahrendt/microWakeWord) by Kevin Ahrendt (original)
- [ESPHome micro_wake_word](https://esphome.io/components/micro_wake_word.html)
