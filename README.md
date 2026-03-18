# microWakeWord Custom Trainer

Train a custom wake word model for [ESPHome](https://esphome.io) and [Home Assistant](https://www.home-assistant.io/) using [microWakeWord](https://github.com/kahrendt/microWakeWord). No ML experience required.

You provide a phonetic wake word (like `hey_air_uh_gorn`), and this project generates thousands of synthetic voice samples, augments them with background noise and room acoustics, trains a tiny (~100 KB) TFLite model, and pushes it to your GitHub repo — ready to drop into an ESPHome config.

---

## Choose Your Setup

| | RunPod (cloud GPU) | Local / VPS with GPU | Local / VPS (CPU-only) |
|---|---|---|---|
| **Speed** | ~2 hrs for 20k steps | ~2 hrs (depends on GPU) | Very slow (hours–days) |
| **Cost** | ~$1–2 per run | Free (your hardware) | Free |
| **Best for** | No GPU at home | You have a dedicated GPU | Testing / iterating on config |
| **Setup** | `bash setup.sh` | `bash setup_local.sh` | `bash setup_local.sh` |

> **Testing on a low-spec machine?** The script auto-detects available RAM and GPU. On machines with < 12 GB RAM and no GPU, training is skipped by default to prevent freezing. Use `--dry-run` to validate the full pipeline without training, or `--force-train` to attempt training anyway.

Any NVIDIA GPU will speed up training compared to CPU-only. Larger GPUs (20+ GB VRAM like RTX 3090/4090) can use the default batch size of 128. Smaller GPUs (8–12 GB) will work too — TensorFlow automatically adjusts, or you can reduce `batch_size` in the generated `training_parameters.yaml` if you hit out-of-memory errors. CPU-only mode automatically reduces batch size to 32 to avoid out-of-memory crashes.

---

## Prerequisites

You need two things before starting, regardless of which setup you choose.

### 1. Fork or clone this repo

Go to this repo on GitHub and click **Fork** (top right). This gives you your own copy where trained models will be pushed.

### 2. Create a GitHub Personal Access Token (PAT)

The training script pushes your finished model back to your GitHub repo. It needs a token to do that.

1. Go to **GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens**
   (direct link: https://github.com/settings/personal-access-tokens/new)
2. Give it a name like `wakeword-training`
3. Under **Repository access**, select **Only select repositories** and pick your forked repo
4. Under **Permissions → Repository permissions**, set **Contents** to **Read and write**
5. Click **Generate token** and copy it — you'll need it during setup

---

## Setup Option A: RunPod (Cloud GPU)

### 1. Create a RunPod pod

- Go to [runpod.io](https://runpod.io) and create an account
- Deploy a new GPU pod with these settings:

| Setting | Value |
|---|---|
| Template | `tensorflow/tensorflow:latest-gpu` (or a specific version like `2.21.0-gpu`) |
| GPU | RTX 4090 (recommended) or any NVIDIA GPU |
| Volume | 150 GB network volume mounted at `/workspace` |

- Under **Environment Variables** (Edit Pod → Environment Variables), add:

| Key | Value |
|---|---|
| `GITHUB_TOKEN` | Your PAT from above (`ghp_...` or `github_pat_...`) |
| `GITHUB_REPO` | `yourusername/your-repo-name` |

### 2. SSH into the pod and install basics

```bash
ssh root@<pod-ip> -p <port>
```

The RunPod TensorFlow template is minimal — update it and install the tools you'll need:

```bash
apt-get update && apt-get upgrade -y
apt-get install -y tmux git
```

### 3. Start a tmux session

```bash
tmux new-session -s training
```

tmux keeps your training running if your SSH connection drops. To detach later: press `Ctrl+B` then `D`. To reattach: `tmux attach -t training`.

### 4. Clone your repo and run setup

```bash
git clone https://github.com/YOURUSERNAME/YOURREPO.git /workspace/training
cd /workspace/training
bash setup.sh
```

### 5. Start training

```bash
python -u train.py 2>&1 | tee training.log
```

Jump to [Training Walkthrough](#training-walkthrough) below.

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

This installs system packages, creates a Python virtual environment, installs TensorFlow and dependencies, and auto-detects your GPU. It will ask you where to put the training data — press Enter to use your current directory, or type a different path:

```
[2/4] Where should the training data go?
  Press Enter for current directory: /home/you/hey-aragorn-wakeword
  Or type a path:
```

You can also skip the prompt by passing a path as an argument: `bash setup_local.sh /path/to/workdir`

### 3. Configure your `.env` file

The setup script automatically creates a `.env` file from the template. Open it and fill in your GitHub credentials:

```bash
nano .env
```

The file looks like this:

```
# Working directory for training data, models, and repos
WORK_DIR=~/wakeword-training

# GitHub repo (owner/name) for pushing trained models
GITHUB_REPO=yourusername/your-repo-name

# GitHub personal access token (with repo scope)
GITHUB_TOKEN=github_pat_xxxxxxxxxxxx
```

- `WORK_DIR` — already filled in by the setup script
- `GITHUB_REPO` — your GitHub username and repo name, e.g. `BigPappy098/hey-aragorn-wakeword`
- `GITHUB_TOKEN` — the PAT you created earlier

> **Security:** The `.env` file is gitignored and will never be committed. Your token stays local.

### 4. Activate the virtual environment and start training

```bash
source ~/wakeword-training/venv/bin/activate
python3 -u train.py 2>&1 | tee training.log
```

> **Tip:** The `-u` flag ensures interactive prompts display correctly when piping through `tee`.

---

## Training Walkthrough

Here's exactly what happens when you run `train.py`. The script is fully interactive — it walks you through everything.

### Step 1: Configure your wake word

You'll be prompted for three things:

```
Enter wake word phonetically (e.g. hey_air_uh_gorn): hey_air_uh_gorn
Number of voice samples [default: 1000, better quality: 2000-5000]: 1000
Training steps [default: 20000, better quality: 20000-30000]: 20000
```

**Tips for the phonetic spelling:**
- Use underscores between syllables: `hey_air_uh_gorn`, not `heyaragorn`
- Spell it how it sounds, not how it's written
- Longer, more unique phrases reduce false activations
- Examples: `hey_air_uh_gorn`, `ok_computer`, `hey_jar_vis`

### Step 2: Pronunciation preview

The script generates one sample and pushes a `.wav` file to your GitHub repo so you can listen to it. It prints a download link:

```
Preview uploaded to: https://github.com/you/repo/blob/main/models/preview_hey_air_uh_gorn.wav
Does the pronunciation sound correct? [y/n]:
```

**Listen carefully.** This is your only chance to catch a bad pronunciation before generating hundreds of samples. If it sounds wrong, type `n` and re-run with a different phonetic spelling.

### Step 3: Real recordings (optional)

If you have recordings of yourself (or others) saying the wake word, place them in the `real_recordings/` folder before running `train.py`. Supported formats: `.wav`, `.mp3`, `.m4a`, `.flac`.

The script will detect them and mix them 50/50 with synthetic samples. This can improve accuracy, but is completely optional — synthetic-only works fine.

**Recording tips:**
- Record 10–50 repetitions per clip with a 1–2 second pause between each
- Use your normal speaking voice and distance from mic
- Phone or laptop mic is fine — the script handles noise and normalization

### Step 4: Automatic generation and augmentation

From here, everything is automatic. The script will:

1. **Generate synthetic samples** using Piper TTS (~1000 WAV files)
2. **Download room acoustics data** from OpenSLR (~1.3 GB, cached for future runs)
3. **Augment samples** with background noise, reverb, pitch shifts, EQ, and distortion
4. **Download negative datasets** from Hugging Face (~1.3 GB, cached for future runs)
5. **Generate training config** (`training_parameters.yaml`)

### Step 5: Training

The model trains automatically. Progress is logged every 500 steps with metrics like accuracy, recall, precision, and loss.

**CLI flags:**

| Flag | Effect |
|---|---|
| `--dry-run` | Validate the full pipeline (sample generation, augmentation, config) but skip model training. Useful for testing on low-spec machines. |
| `--force-train` | Override the automatic low-resource detection and attempt training even on machines with limited RAM / no GPU. |

**Estimated training times (20,000 steps):**

| GPU | Time |
|---|---|
| RTX 4090 (24 GB) | ~2 hours |
| RTX 3090 (24 GB) | ~2–3 hours |
| Smaller GPU (8–12 GB) | ~3–5 hours (may need reduced batch size) |
| CPU-only | Many hours (not recommended for full runs) |

**Logs:** All output is saved to `training.log` in your repo directory (created by the `tee` command above). To monitor from a separate terminal:

```bash
tail -f training.log
```

**What good training looks like** (final validation):

| Metric | Target |
|---|---|
| Accuracy | > 99% |
| Recall | > 95% |
| Precision | > 95% |
| Loss | < 0.01 |

---

## Output

When training completes, two files are pushed to your GitHub repo:

```
models/
  hey_air_uh_gorn.tflite    <- the trained model (~100 KB)
  hey_air_uh_gorn.json      <- ESPHome metadata
```

The script prints the exact URL to use in ESPHome:

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

Flash the device and you're done — it will now respond to your custom wake word.

---

## Troubleshooting

### cuDNN version mismatch

TensorFlow 2.18 requires cuDNN 9.3+. The script handles this automatically by installing the correct version to a side directory. If you still see cuDNN errors, try:

```bash
export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")/lib:$LD_LIBRARY_PATH
```

### "No GPU detected" warning

If you have a GPU but the script doesn't see it, make sure CUDA 12.x and your NVIDIA drivers are installed:

```bash
nvidia-smi          # should show your GPU
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

On RunPod this is handled by the Docker template. On a local machine you may need to install CUDA yourself.

### Machine freezes during training

On machines with < 12 GB RAM and no GPU, the training step can consume all available memory and freeze the system. The script detects this automatically and skips training. Use `--dry-run` to validate the pipeline, then run the actual training on a GPU machine.

If you want to force training on a low-spec machine: `python3 -u train.py --force-train`. The script will reduce batch size to 16 and limit TensorFlow threads, but expect very slow performance.

### Out of memory (OOM) crash

On CPU, batch size is automatically reduced to 32 (or 16 on low-resource machines). On GPU, the default is 128. If you still hit OOM, edit `training_parameters.yaml` after it's generated and reduce `batch_size` to 64 or 32, then re-run from Step 12 onward.

### ZeroDivisionError at the end of training

This is a known bug in the upstream microWakeWord evaluation code. **It does not affect your model.** Training completed successfully and your `.tflite` file is fine. The fix is already patched in `train.py`.

### Training seems stuck at step 0

The first step takes longer than the rest because TensorFlow is compiling GPU kernels. Give it a few minutes.

### Checking if the model was saved

```bash
find ~/wakeword-training -name "*.tflite"
# or on RunPod:
find /workspace/training -name "*.tflite"
```

---

## Repo Structure

```
/
├── train.py              <- Main training script (interactive, does everything)
├── setup.sh              <- RunPod setup (bash — installs base deps)
├── setup_local.sh        <- Local/VPS setup (bash — creates venv + installs deps)
├── .env.example          <- Template for local env vars
├── .gitignore            <- Keeps secrets, logs, and training artifacts out of git
├── real_recordings/      <- Drop your own voice clips here (optional)
├── models/               <- Trained models are saved and pushed here
│   ├── *.tflite
│   └── *.json
└── README.md
```

---

## Credits

- [microWakeWord](https://github.com/kahrendt/microWakeWord) by Kevin Ahrendt
- [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) by rhasspy
- [ESPHome micro_wake_word](https://esphome.io/components/micro_wake_word.html)
