#!/usr/bin/env python3
"""microWakeWord Custom Trainer — with real-recording support & cuDNN fix."""

# ── Fix cuDNN path BEFORE anything else imports TF ────────────────────────────
# The pip-installed nvidia-cudnn-cu12 ships the version TF was compiled against.
# We must ensure LD_LIBRARY_PATH points there so the *subprocess* (training) also
# picks it up via the dynamic linker, not the older system-level cuDNN.
import subprocess as _sp, os as _os, sys as _sys, ctypes as _ctypes, glob as _glob

# ── Ensure cuDNN 9.3.0 libs are available (TF 2.18 requires >= 9.3) ──────────
# We can't just pip-upgrade nvidia-cudnn-cu12 because PyTorch pins ==9.1.0.70.
# Instead, download the 9.3.0 wheel to a separate directory and extract just
# the .so files, then point LD_LIBRARY_PATH there.
_CUDNN_VERSION = '9.3.0.75'
_CUDNN_LOCAL = '/usr/local/lib/cudnn-9.3'
_cudnn_lib = None

def _ensure_cudnn_93():
    """Install cuDNN 9.3.0 to a side directory without disturbing pip packages."""
    lib_dir = _os.path.join(_CUDNN_LOCAL, 'nvidia', 'cudnn', 'lib')
    if _os.path.isdir(lib_dir) and _glob.glob(_os.path.join(lib_dir, 'libcudnn.so*')):
        return lib_dir  # already installed
    print(f"[init] Installing cuDNN {_CUDNN_VERSION} to {_CUDNN_LOCAL}...")
    _sp.run([_sys.executable, '-m', 'pip', 'install', '-q',
             '--no-deps', f'--target={_CUDNN_LOCAL}',
             f'nvidia-cudnn-cu12=={_CUDNN_VERSION}'],
            check=True)
    return lib_dir

try:
    # First check if the existing pip cuDNN is already >= 9.3
    _existing = _sp.check_output(
        [_sys.executable, '-c',
         'from importlib.metadata import version; print(version("nvidia-cudnn-cu12"))'],
        text=True, timeout=30).strip()
    _existing_ok = tuple(int(x) for x in _existing.split('.')[:2]) >= (9, 3)
except Exception:
    _existing_ok = False

if _existing_ok:
    # Pip package claims >= 9.3, try to use it directly
    try:
        _cudnn_lib = _sp.check_output(
            [_sys.executable, '-c',
             'import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))'],
            text=True, timeout=30).strip() + '/lib'
    except Exception:
        # Module structure changed in newer versions — fall back to side-install
        _existing_ok = False

if not _existing_ok:
    # Need cuDNN 9.3 — install to a side directory to avoid pip conflicts
    try:
        _cudnn_lib = _ensure_cudnn_93()
        print(f"[init] cuDNN 9.3.0 libs ready at: {_cudnn_lib}")
    except Exception as _e:
        print(f"[init] WARNING: Failed to install cuDNN 9.3.0 ({_e}).")

# Set LD_LIBRARY_PATH and preload into current process
if _cudnn_lib and _os.path.isdir(_cudnn_lib):
    _os.environ['LD_LIBRARY_PATH'] = \
        f"{_cudnn_lib}:{_os.environ.get('LD_LIBRARY_PATH', '')}"
    for _f in sorted(_os.listdir(_cudnn_lib)):
        if _f.startswith('libcudnn') and '.so' in _f:
            try:
                _ctypes.CDLL(_os.path.join(_cudnn_lib, _f), mode=_ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    print(f"[init] cuDNN path set: {_cudnn_lib}")
else:
    print("[init] WARNING: No cuDNN 9.3 path available. Training may fail.")

# ── Load .env file if present (for non-RunPod setups) ────────────────────────
_env_file = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.env')
if _os.path.isfile(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _key, _, _val = _line.partition('=')
                _key, _val = _key.strip(), _val.strip()
                if _key and (_key not in _os.environ or not _os.environ[_key]):
                    _os.environ[_key] = _os.path.expanduser(_val)

# ── Normal imports ────────────────────────────────────────────────────────────
import os, sys, shutil, yaml, urllib.request, zipfile, json
import numpy as np

# ── CLI flags ────────────────────────────────────────────────────────────────
DRY_RUN      = '--dry-run'      in sys.argv
FORCE_TRAIN  = '--force-train'  in sys.argv

# ── Hardware helpers ─────────────────────────────────────────────────────────
def _get_ram_gb(kind='MemTotal'):
    """Read RAM info from /proc/meminfo. kind can be MemTotal or MemAvailable."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith(kind):
                    return int(line.split()[1]) / (1024 * 1024)  # kB → GB
    except OSError:
        pass
    return 0.0

TOTAL_RAM_GB = _get_ram_gb('MemTotal')

# ── Helpers ───────────────────────────────────────────────────────────────────
def prompt(msg, default=None, valid_fn=None):
    """Flush-safe input helper that avoids buffering issues."""
    sys.stdout.write(msg)
    sys.stdout.flush()
    raw = sys.stdin.readline().strip()
    if not raw and default is not None:
        return default
    if valid_fn and not valid_fn(raw):
        return default
    return raw


def git_push(branch='main'):
    """Sync with remote then push to GitHub."""
    # Fetch latest remote state
    subprocess.run(
        ['git', 'fetch', 'origin', branch],
        cwd=REPO_ROOT, capture_output=True, text=True
    )
    # Try a normal pull --rebase first
    pull = subprocess.run(
        ['git', 'pull', '--rebase', 'origin', branch],
        cwd=REPO_ROOT, capture_output=True, text=True
    )
    if pull.returncode != 0:
        # Abort any failed rebase
        subprocess.run(['git', 'rebase', '--abort'],
                       cwd=REPO_ROOT, capture_output=True, text=True)
        # Stash our commit, reset to remote, then re-apply
        print("  [git] Histories diverged — rebasing onto remote...")
        subprocess.run(['git', 'stash'], cwd=REPO_ROOT, capture_output=True)
        subprocess.run(['git', 'reset', '--hard', f'origin/{branch}'],
                       cwd=REPO_ROOT, capture_output=True)
        subprocess.run(['git', 'stash', 'pop'], cwd=REPO_ROOT,
                       capture_output=True)
        # Re-commit any unstaged changes from stash pop
        subprocess.run(['git', 'add', '-A'], cwd=REPO_ROOT, capture_output=True)
        subprocess.run(
            ['git', 'commit', '--allow-empty', '-m', 'Re-apply local changes on updated remote'],
            cwd=REPO_ROOT, capture_output=True, text=True
        )
    result = subprocess.run(
        ['git', 'push', 'origin', branch],
        cwd=REPO_ROOT, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"\n❌ git push failed (exit {result.returncode})")
        if result.stderr:
            safe_err = result.stderr.replace(GITHUB_TOKEN, '***')
            print(f"   stderr: {safe_err.strip()}")
        if result.stdout:
            print(f"   stdout: {result.stdout.strip()}")
        print("\n   Common causes:")
        print("   • Token expired or lacks 'repo' scope — generate a new one at:")
        print("     https://github.com/settings/tokens")
        print("   • GITHUB_REPO is wrong — should be 'User/repo-name'")
        print(f"   • Current token starts with: {GITHUB_TOKEN[:4]}...")
        sys.exit(1)
    print(f"  ✅ Pushed to {branch}")


def git_configure():
    """Idempotent git identity + remote setup.  Clones the repo if needed."""
    subprocess.run(['git', 'config', '--global', 'safe.directory', '*'],
                   check=True, capture_output=True)
    subprocess.run(['git', 'config', '--global', 'user.email', 'runpod@training.bot'],
                   check=True, capture_output=True)
    subprocess.run(['git', 'config', '--global', 'user.name', 'RunPod Training Bot'],
                   check=True, capture_output=True)
    subprocess.run(['git', 'config', '--global', 'pull.rebase', 'false'],
                   check=True, capture_output=True)
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return
    repo_url = f'https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git'
    # If REPO_ROOT isn't a git repo yet, init in place (dir may have files)
    if not os.path.isdir(os.path.join(REPO_ROOT, '.git')):
        print(f"[git] Initialising repo at {REPO_ROOT}...")
        os.makedirs(REPO_ROOT, exist_ok=True)
        subprocess.run(['git', 'init'], check=True, cwd=REPO_ROOT)
        subprocess.run(['git', 'remote', 'add', 'origin', repo_url],
                       check=True, cwd=REPO_ROOT)
        subprocess.run(['git', 'fetch', 'origin', 'main'],
                       check=True, cwd=REPO_ROOT)
        subprocess.run(['git', 'reset', 'origin/main'],
                       check=True, cwd=REPO_ROOT)
    else:
        subprocess.run(['git', 'remote', 'set-url', 'origin', repo_url],
                       check=True, capture_output=True, cwd=REPO_ROOT)

import subprocess                       # re-import at module scope for clarity

# ── INTERACTIVE CONFIG ────────────────────────────────────────────────────────
print("\n=== microWakeWord Custom Trainer ===\n")

while True:
    raw = prompt("Enter wake word phonetically (e.g. hey_air_uh_gorn): ")
    if raw:
        break
    print("  (cannot be empty)")
TARGET_WORD = raw.lower().replace(' ', '_')
print(f"  → Using: {TARGET_WORD}")

raw_samples = prompt("\nNumber of voice samples [default: 1000, better quality: 2000-5000]: ",
                     default='1000')
NUM_SAMPLES = int(raw_samples) if raw_samples.isdigit() else 1000
print(f"  → Samples: {NUM_SAMPLES}")

raw_steps = prompt("\nTraining steps [default: 20000, better quality: 20000-30000]: ",
                   default='20000')
TRAINING_STEPS = int(raw_steps) if raw_steps.isdigit() else 20000
print(f"  → Training steps: {TRAINING_STEPS}")

print()
GITHUB_REPO  = os.environ.get('GITHUB_REPO', '')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
USE_GITHUB = bool(GITHUB_TOKEN and GITHUB_REPO)

if USE_GITHUB:
    print(f"  → GitHub repo: {GITHUB_REPO}")
    print(f"  → GitHub token: {GITHUB_TOKEN[:4]}...{GITHUB_TOKEN[-4:]} ({len(GITHUB_TOKEN)} chars)")
else:
    print("  → GitHub: not configured (model will be saved locally only)")
    print("    Set GITHUB_TOKEN and GITHUB_REPO in .env to enable auto-push.")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get('WORK_DIR', _SCRIPT_DIR)
REPO_ROOT = os.environ.get('REPO_ROOT', _SCRIPT_DIR)
os.chdir(BASE)

# ── Clean up previous training data (keep downloads to save time) ────────────
print("[Cleanup] Removing previous training artifacts...")
for d in ['generated_samples', 'generated_augmented_features',
          'trained_models', 'microWakeWord']:
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f"  Removed {d}/")
for f in ['training_parameters.yaml']:
    if os.path.exists(f):
        os.remove(f)
        print(f"  Removed {f}")
print("[Cleanup] Done.")

for d in ['generated_samples', 'mit_rirs', 'negative_datasets', 'models']:
    os.makedirs(d, exist_ok=True)

# ── Check for real recordings ─────────────────────────────────────────────────
REAL_DIR = 'real_recordings'
os.makedirs(REAL_DIR, exist_ok=True)


def scan_real_recordings():
    return [
        os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR)
        if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))
    ]


real_audio_files = scan_real_recordings()
USING_REAL = False

if real_audio_files:
    print(f"🎤 Found {len(real_audio_files)} real recording file(s) in {REAL_DIR}/")
    print(f"   Will use 50/50 split: {NUM_SAMPLES // 2} synthetic + "
          f"{NUM_SAMPLES // 2} augmented real")
    USING_REAL = True
else:
    print(f"📂 No real recordings found in {REAL_DIR}/")
    if USE_GITHUB:
        ans = prompt("Would you like to upload real recordings to GitHub first? [y/n]: ",
                     default='n')
        if ans.lower() == 'y':
            print(f"\n📤 Upload your recording file(s) here:")
            print(f"   https://github.com/{GITHUB_REPO}/upload/main/real_recordings")
            print(f"\n   Supported formats: .wav  .mp3  .m4a  .flac")
            print(f"   Tip: One long file with 10-50 repetitions, 1-2 sec pause between each.\n")
            prompt("Press Enter when your files are uploaded and committed to GitHub...")
            git_configure()
            subprocess.run(['git', 'pull', 'origin', 'main'], check=True, cwd=REPO_ROOT)
            real_audio_files = scan_real_recordings()
            if real_audio_files:
                print(f"✅ Found {len(real_audio_files)} recording file(s) — using 50/50 split.")
                USING_REAL = True
            else:
                print("⚠️  No files found after pull. Continuing with synthetic-only.")
        else:
            print("📢 Skipping real recordings — using 100% synthetic samples.")
    else:
        print("   Place .wav/.mp3/.m4a/.flac files in real_recordings/ and re-run.")
        print("📢 Continuing with 100% synthetic samples.")

SYNTHETIC_COUNT = NUM_SAMPLES // 2 if USING_REAL else NUM_SAMPLES
REAL_TARGET     = NUM_SAMPLES // 2 if USING_REAL else 0

# ── Step 3: Clone repos ───────────────────────────────────────────────────────
print("\n[Step 3] Cloning microWakeWord + piper-sample-generator...")
if os.path.exists('microWakeWord'):
    shutil.rmtree('microWakeWord')
subprocess.run(['git', 'clone', 'https://github.com/kahrendt/microWakeWord.git'],
               check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', 'microWakeWord'],
               check=True)
# TensorBoard is required by tf.summary.scalar used during training
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tensorboard'],
               check=True)

# Patch: fix validate_nonstreaming for Keras 3 / TF 2.18 compatibility.
# Uses streaming from_generator() instead of get_data() to avoid OOM,
# and handles Keras 3 metric key naming (e.g. "binary_accuracy").
CUT_MARKER = '\n# === TRAIN.PY PATCH ==='
_train_py_path = 'microWakeWord/microwakeword/train.py'
with open(_train_py_path, 'r') as f:
    _src = f.read()
if CUT_MARKER in _src:
    _src = _src[:_src.index(CUT_MARKER)]

_patch = CUT_MARKER + '''
import tensorflow as tf

# numpy 2.x renamed trapz -> trapezoid
_trapz_fn = getattr(np, "trapezoid", np.trapz)

def validate_nonstreaming(config, data_processor, model, test_set):
    """Streaming validate_nonstreaming — uses from_generator() to avoid OOM.

    Instead of loading all validation data into RAM via get_data(), this
    streams spectrograms one at a time from mmap via get_feature_generator().
    """
    features_length = config["spectrogram_length"]
    bs = config["batch_size"]

    if data_processor.get_mode_size(test_set) == 0:
        return {
            "loss": 99.0, "accuracy": 0, "recall": 0, "precision": 0,
            "auc": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "recall_at_no_faph": 0, "cutoff_for_no_faph": 0,
            "ambient_false_positives": 0, "ambient_false_positives_per_hour": 0,
            "average_viable_recall": 0,
        }

    def _make_gen(mode, trunc):
        def gen():
            for provider in data_processor.feature_providers:
                if provider.get_mode_size(mode) == 0:
                    continue
                label = np.array([provider.label], dtype=np.float32)
                for spec in provider.get_feature_generator(mode, features_length, trunc):
                    yield spec.astype(np.float32), label
        return gen

    out_sig = (
        tf.TensorSpec(shape=(features_length, 40), dtype=tf.float32),
        tf.TensorSpec(shape=(1,),                   dtype=tf.float32),
    )

    # ── Primary test set ──
    gen_fn = _make_gen(test_set, "truncate_start")
    n_samples = sum(1 for _ in gen_fn())
    n_steps = n_samples // bs
    if n_steps == 0:
        return {
            "loss": 99.0, "accuracy": 0, "recall": 0, "precision": 0,
            "auc": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "recall_at_no_faph": 0, "cutoff_for_no_faph": 0,
            "ambient_false_positives": 0, "ambient_false_positives_per_hour": 0,
            "average_viable_recall": 0,
        }

    ds = (tf.data.Dataset.from_generator(gen_fn, output_signature=out_sig)
          .batch(bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE))
    model.reset_metrics()
    result = model.evaluate(ds, steps=n_steps, return_dict=True, verbose=0)

    def _v(d, k):
        v = d.get(k, d.get("binary_" + k, 0))
        return v.numpy() if hasattr(v, "numpy") else v

    metrics = {
        "accuracy": _v(result, "accuracy"),
        "recall":   _v(result, "recall"),
        "precision":_v(result, "precision"),
        "auc":      _v(result, "auc"),
        "loss":     _v(result, "loss"),
        "recall_at_no_faph": 0, "cutoff_for_no_faph": 0,
        "ambient_false_positives": 0, "ambient_false_positives_per_hour": 0,
        "average_viable_recall": 0,
    }
    test_set_fp = _v(result, "fp")

    # ── Ambient set ──
    ambient_mode = test_set + "_ambient"
    if data_processor.get_mode_size(ambient_mode) > 0:
        amb_gen = _make_gen(ambient_mode, "split")
        amb_n = sum(1 for _ in amb_gen())
        amb_steps = amb_n // bs
        if amb_steps > 0:
            amb_ds = (tf.data.Dataset.from_generator(amb_gen, output_signature=out_sig)
                      .batch(bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE))
            with swap_attribute(model, "reset_metrics", lambda: None):
                ambient_result = model.evaluate(amb_ds, steps=amb_steps,
                                                return_dict=True, verbose=0)

            duration_hours = data_processor.get_mode_duration(ambient_mode) / 3600.0
            all_tp = _v(ambient_result, "tp")
            ambient_fp = _v(ambient_result, "fp") - test_set_fp
            all_fn = _v(ambient_result, "fn")

            metrics["auc"]  = _v(ambient_result, "auc")
            metrics["loss"] = _v(ambient_result, "loss")

            recall_at_cutoffs = all_tp / (all_tp + all_fn)
            faph_at_cutoffs = ambient_fp / duration_hours

            target_faph_cutoff_probability = 1.0
            recall_at_no_faph = 0
            for index, cutoff in enumerate(np.linspace(0.0, 1.0, 101)):
                if faph_at_cutoffs[index] == 0:
                    target_faph_cutoff_probability = cutoff
                    recall_at_no_faph = recall_at_cutoffs[index]
                    break

            if faph_at_cutoffs[0] > 2:
                index_of_first_viable = 1
                while (index_of_first_viable < len(faph_at_cutoffs)
                       and faph_at_cutoffs[index_of_first_viable] > 2):
                    index_of_first_viable += 1
                if index_of_first_viable >= len(faph_at_cutoffs):
                    recall_at_2faph = 0
                else:
                    x0 = faph_at_cutoffs[index_of_first_viable - 1]
                    y0 = recall_at_cutoffs[index_of_first_viable - 1]
                    x1 = faph_at_cutoffs[index_of_first_viable]
                    y1 = recall_at_cutoffs[index_of_first_viable]
                    recall_at_2faph = (y0 * (x1 - 2.0) + y1 * (2.0 - x0)) / (x1 - x0)
            else:
                index_of_first_viable = 0
                recall_at_2faph = recall_at_cutoffs[0]

            x_coordinates = [2.0]
            y_coordinates = [recall_at_2faph]
            for index in range(index_of_first_viable, len(recall_at_cutoffs)):
                if faph_at_cutoffs[index] != x_coordinates[-1]:
                    x_coordinates.append(faph_at_cutoffs[index])
                    y_coordinates.append(recall_at_cutoffs[index])

            average_viable_recall = (
                _trapz_fn(np.flip(y_coordinates), np.flip(x_coordinates)) / 2.0
            )

            metrics["recall_at_no_faph"] = recall_at_no_faph
            metrics["cutoff_for_no_faph"] = target_faph_cutoff_probability
            metrics["ambient_false_positives"] = ambient_fp[50]
            metrics["ambient_false_positives_per_hour"] = faph_at_cutoffs[50]
            metrics["average_viable_recall"] = average_viable_recall

    return metrics
'''

with open(_train_py_path, 'w') as f:
    f.write(_src + _patch)

if not os.path.exists('piper-sample-generator'):
    subprocess.run(
        ['git', 'clone', 'https://github.com/rhasspy/piper-sample-generator.git'],
        check=True)
print("✅ Repos ready")

# ── Ensure PyTorch + piper-tts are available (needed by piper-sample-generator)
try:
    import torch  # noqa: F401
except ImportError:
    print("[dep] PyTorch not found — installing CPU-only version...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'torch', 'torchvision', 'torchaudio',
                    '--index-url', 'https://download.pytorch.org/whl/cpu'],
                   check=True)

try:
    import piper  # noqa: F401
except ImportError:
    print("[dep] piper-tts not found — installing...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'piper-tts'],
                   check=True)

try:
    import audiomentations
    if not hasattr(audiomentations, 'AddColorNoise'):
        raise ImportError("audiomentations too old, needs >=0.35.0")
except ImportError:
    print("[dep] audiomentations >=0.35.0 not found — installing...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    '--upgrade', '--force-reinstall',
                    'audiomentations>=0.35.0', 'webrtcvad'],
                   check=True)
    import importlib
    import audiomentations
    importlib.reload(audiomentations)

try:
    import torchcodec  # noqa: F401
except ImportError:
    print("[dep] torchcodec not found — installing (needed by datasets for audio)...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torchcodec'],
                   check=True)

# ── Step 4: Piper voice model ─────────────────────────────────────────────────
print("\n[Step 4] Downloading Piper model...")
os.makedirs('piper-sample-generator/models', exist_ok=True)
model_path = 'piper-sample-generator/models/en_US-libritts_r-medium.pt'
if not os.path.exists(model_path):
    urllib.request.urlretrieve(
        'https://github.com/rhasspy/piper-sample-generator/releases/download/'
        'v2.0.0/en_US-libritts_r-medium.pt',
        model_path)
print("✅ Piper model ready")

# ── Preview step ──────────────────────────────────────────────────────────────
print("\n[Preview] Generating 1 sample to preview pronunciation...")
preview_dir = '/tmp/preview_sample'
if os.path.exists(preview_dir):
    shutil.rmtree(preview_dir)
os.makedirs(preview_dir)
# Detect CPU capability — older CPUs without AVX2 need conservative settings
def _has_avx2():
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('flags'):
                    return 'avx2' in line.split()
    except OSError:
        pass
    return False

HAS_AVX2 = _has_avx2()
PIPER_BATCH_SIZE = 100 if HAS_AVX2 else 10
if not HAS_AVX2:
    print(f"  ⚠ CPU lacks AVX2 — using batch-size {PIPER_BATCH_SIZE} and safe kernel dispatch")

# LOW_RESOURCE is refined later once GPU detection runs; for now base it on RAM
LOW_RESOURCE = TOTAL_RAM_GB < 12
if DRY_RUN:
    print("  ℹ Dry-run mode: will validate pipeline but skip model training.")
elif LOW_RESOURCE:
    print(f"  ⚠ Low-resource machine detected ({TOTAL_RAM_GB:.1f} GB RAM).")
    print(f"    Training will be skipped unless you pass --force-train.")

piper_env = {
    **os.environ,
    'PYTHONPATH': os.path.abspath('piper-sample-generator'),
    **({'ATEN_CPU_CAPABILITY': 'default'} if not HAS_AVX2 else {}),
}
subprocess.run([
    sys.executable, '-m', 'piper_sample_generator',
    TARGET_WORD, '--max-samples', '1', '--batch-size', '1',
    '--model', model_path, '--output-dir', preview_dir
], text=True, env=piper_env, check=True)

preview_files = [f for f in os.listdir(preview_dir) if f.endswith('.wav')]
if preview_files:
    preview_dest = f'models/preview_{TARGET_WORD}.wav'
    os.makedirs(os.path.join(REPO_ROOT, 'models'), exist_ok=True)
    shutil.copy(os.path.join(preview_dir, preview_files[0]),
                os.path.join(REPO_ROOT, preview_dest))
    if USE_GITHUB:
        git_configure()
        subprocess.run(['git', 'add', preview_dest], check=True, cwd=REPO_ROOT)
        subprocess.run(['git', 'commit', '-m',
                        f'Preview sample: {TARGET_WORD}'], check=True, cwd=REPO_ROOT)
        git_push()
        print(f"\n🔊 Preview pushed to GitHub!")
        print(f"   Go to: https://github.com/{GITHUB_REPO}/blob/main/{preview_dest}")
    else:
        print(f"\n🔊 Preview saved to: {os.path.join(REPO_ROOT, preview_dest)}")
    print(f"   Play the .wav file to check pronunciation.\n")
    while True:
        answer = prompt("Does the pronunciation sound correct? [y/n]: ", default='')
        if answer.lower() in ('y', 'n'):
            break
        print("  Please enter y or n.")
    if answer.lower() == 'n':
        print("\n❌ Exiting. Re-run train.py with a different phonetic spelling.")
        sys.exit(0)
print("✅ Pronunciation confirmed, continuing...\n")

# ── Step 7: Generate synthetic samples ────────────────────────────────────────
print(f"\n[Step 7] Generating {SYNTHETIC_COUNT} synthetic samples for '{TARGET_WORD}'...")
if os.path.exists('generated_samples'):
    shutil.rmtree('generated_samples')
os.makedirs('generated_samples')
subprocess.run([
    sys.executable, '-m', 'piper_sample_generator',
    TARGET_WORD, '--max-samples', str(SYNTHETIC_COUNT), '--batch-size', str(PIPER_BATCH_SIZE),
    '--model', model_path, '--output-dir', 'generated_samples'
], text=True, env=piper_env, check=True)
synth_count = len([f for f in os.listdir('generated_samples') if f.endswith('.wav')])
print(f"✅ {synth_count} synthetic samples generated")

# ── Step 7b: Process + augment real recordings ───────────────────────────────
if USING_REAL:
    print(f"\n[Step 7b] Processing real recordings → target {REAL_TARGET} augmented clips...")
    if not shutil.which('ffmpeg'):
        apt_cmd = ['apt-get', 'install', '-y', '-q', 'ffmpeg']
        if os.getuid() != 0:
            apt_cmd = ['sudo'] + apt_cmd
        subprocess.run(apt_cmd, check=True, capture_output=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'soundfile', 'librosa'],
                   check=True, capture_output=True)
    import soundfile as sf
    import librosa

    def adaptive_split(audio_np, sr, min_clip_ms=250, max_clip_ms=3500,
                        min_silence_ms=400, keep_silence_ms=80):
        """Split a recording into individual utterances using adaptive thresholding.

        Instead of a fixed dB threshold, we compute the RMS energy of each frame
        and set the speech/silence boundary relative to the actual recording's
        noise floor vs. peak energy. This handles phone recordings, laptop mics,
        and noisy environments far better than a static -35 dB cutoff.
        """
        frame_ms  = 20
        frame_len = int(sr * frame_ms / 1000)
        if len(audio_np) < frame_len * 2:
            return []

        # Compute per-frame RMS energy
        n_frames = len(audio_np) // frame_len
        frames = audio_np[:n_frames * frame_len].reshape(n_frames, frame_len)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))

        # Adaptive threshold: midpoint between the noise floor (10th percentile)
        # and the speech level (90th percentile) on a log scale
        rms_safe = np.clip(rms, 1e-10, None)           # avoid log(0)
        rms_db   = 20 * np.log10(rms_safe)
        p10, p90 = np.percentile(rms_db, [10, 90])
        # Weight toward the quiet side so we don't clip the tails of utterances
        threshold_db = p10 + 0.35 * (p90 - p10)
        threshold    = 10 ** (threshold_db / 20)
        print(f"    Adaptive threshold: {threshold_db:.1f} dB  "
              f"(noise floor ~{p10:.1f} dB, speech peak ~{p90:.1f} dB)")

        is_speech = rms > threshold

        # Convert ms params to frame counts
        min_sil_frames  = max(1, min_silence_ms // frame_ms)
        keep_frames     = max(1, keep_silence_ms // frame_ms)
        min_clip_frames = max(1, min_clip_ms // frame_ms)
        max_clip_frames = max(1, max_clip_ms // frame_ms)

        # Walk through frames, collecting speech regions
        clips = []
        i = 0
        while i < n_frames:
            if is_speech[i]:
                start = max(0, i - keep_frames)
                j = i
                while j < n_frames:
                    if not is_speech[j]:
                        sil_start = j
                        while j < n_frames and not is_speech[j]:
                            j += 1
                        if j - sil_start >= min_sil_frames:
                            # Found a real gap — end this clip
                            end = min(n_frames, sil_start + keep_frames)
                            clips.append((start, end))
                            i = j
                            break
                    else:
                        j += 1
                else:
                    # Reached end of file while in speech
                    clips.append((start, min(n_frames, j)))
                    i = j
            else:
                i += 1

        # Convert frame indices to audio samples and filter by duration
        result = []
        too_short = 0
        too_long  = 0
        for (s, e) in clips:
            dur_frames = e - s
            if dur_frames < min_clip_frames:
                too_short += 1
                continue
            if dur_frames > max_clip_frames:
                too_long += 1
                continue
            result.append(audio_np[s * frame_len : e * frame_len])

        if too_short or too_long:
            print(f"    Filtered out: {too_short} too short (<{min_clip_ms}ms), "
                  f"{too_long} too long (>{max_clip_ms}ms)")

        return result

    raw_clips = []
    for fpath in real_audio_files:
        print(f"  Loading {os.path.basename(fpath)}...")
        audio_np, sr = librosa.load(fpath, sr=16000, mono=True)
        dur_s = len(audio_np) / sr
        print(f"    Audio duration: {dur_s:.1f}s")
        clips = adaptive_split(audio_np, sr)
        print(f"    {len(clips)} valid clips extracted")
        raw_clips.extend(clips)

    print(f"  Total individual clips: {len(raw_clips)}")

    if len(raw_clips) == 0:
        print("  ⚠️  No valid clips detected — falling back to synthetic-only.")
        USING_REAL = False
    else:
        reps_per_clip = max(1, -(-REAL_TARGET // len(raw_clips)))  # ceiling div
        if reps_per_clip > 200:
            print(f"  ⚠️  WARNING: Only {len(raw_clips)} clip(s) → each will be "
                  f"augmented ~{reps_per_clip}x to reach {REAL_TARGET}.")
            print(f"       For best quality, record at least 15-20 repetitions.")
            print(f"       Capping augmentation repetitions at 200 per clip to "
                  f"reduce overfitting risk.")
            reps_per_clip = 200
            REAL_TARGET = min(REAL_TARGET, len(raw_clips) * (reps_per_clip + 1))
            print(f"       Adjusted real target: {REAL_TARGET}")

        augmented_count = 0
        for i, clip in enumerate(raw_clips):
            if augmented_count >= REAL_TARGET:
                break
            # Write original
            orig_path = f'generated_samples/real_{i:04d}_orig.wav'
            sf.write(orig_path, clip, 16000)
            augmented_count += 1
            # Write augmented variants
            for rep in range(reps_per_clip):
                if augmented_count >= REAL_TARGET:
                    break
                aug = clip.copy()
                speed = np.random.uniform(0.85, 1.15)
                new_len = int(len(aug) / speed)
                aug = np.interp(np.linspace(0, len(aug) - 1, new_len),
                                np.arange(len(aug)), aug).astype(np.float32)
                aug = aug * np.random.uniform(0.6, 1.4)
                aug = aug + (np.random.randn(len(aug)).astype(np.float32)
                             * np.random.uniform(0.0, 0.008))
                aug = np.clip(aug, -1.0, 1.0)
                sf.write(f'generated_samples/real_{i:04d}_aug{rep:03d}.wav',
                         aug, 16000)
                augmented_count += 1

        total_samples = len([f for f in os.listdir('generated_samples')
                             if f.endswith('.wav')])
        print(f"✅ {augmented_count} real/augmented clips written")
        print(f"✅ {total_samples} total samples "
              f"({synth_count} synthetic + {augmented_count} real)")

# ── Step 8: Augmentation data ─────────────────────────────────────────────────
print("\n[Step 8] Downloading RIRs + background noise (~1.3 GB)...")
if not os.listdir('mit_rirs'):
    subprocess.run(['wget', '-q', '--show-progress', '-O', '/tmp/rirs_noises.zip',
                    'https://www.openslr.org/resources/28/rirs_noises.zip'], check=True)
    subprocess.run(['unzip', '-q', '/tmp/rirs_noises.zip', '-d', 'mit_rirs'], check=True)
    os.remove('/tmp/rirs_noises.zip')
print("✅ Augmentation data ready")

# ── Step 9: Spectrograms ──────────────────────────────────────────────────────
print("\n[Step 9] Generating spectrograms...")
sys.path.insert(0, 'microWakeWord')
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
from mmap_ninja.ragged import RaggedMmap

MMAP_TRAIN = 'generated_augmented_features/training/wakeword_mmap'
MMAP_TEST  = 'generated_augmented_features/testing/wakeword_mmap'
os.makedirs(os.path.dirname(MMAP_TRAIN), exist_ok=True)
os.makedirs(os.path.dirname(MMAP_TEST),  exist_ok=True)

clips_obj = Clips(input_directory='generated_samples', file_pattern='*.wav',
                  max_clip_duration_s=None, remove_silence=False,
                  random_split_seed=10, split_count=0.1)
augmenter = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={
        'SevenBandParametricEQ': 0.1, 'TanhDistortion': 0.1,
        'PitchShift': 0.1, 'BandStopFilter': 0.1,
        'AddColorNoise': 0.1, 'AddBackgroundNoise': 0.75,
        'Gain': 1.0, 'RIR': 0.5,
    },
    impulse_paths=['mit_rirs'],
    background_paths=['mit_rirs/RIRS_NOISES/pointsource_noises'],
    background_min_snr_db=-5, background_max_snr_db=10,
    min_jitter_s=0.195, max_jitter_s=0.205,
)
spectrograms = SpectrogramGeneration(clips=clips_obj, augmenter=augmenter,
                                     slide_frames=10, step_ms=10)

print("  Writing train split...")
RaggedMmap.from_generator(
    out_dir=MMAP_TRAIN,
    sample_generator=spectrograms.spectrogram_generator(split='train', repeat=2),
    batch_size=100, verbose=True
)
print("  Writing test split...")
RaggedMmap.from_generator(
    out_dir=MMAP_TEST,
    sample_generator=spectrograms.spectrogram_generator(split='test', repeat=1),
    batch_size=100, verbose=True
)
# Create validation → testing symlink (wake word data only has train/test splits)
val_dir = 'generated_augmented_features/validation'
test_dir = 'generated_augmented_features/testing'
if os.path.islink(val_dir):
    os.unlink(val_dir)
elif os.path.isdir(val_dir):
    shutil.rmtree(val_dir)
os.symlink(os.path.basename(test_dir), val_dir)
print("  Created validation/ → testing/ symlink")

mmap_tr = RaggedMmap(MMAP_TRAIN)
mmap_te = RaggedMmap(MMAP_TEST)
assert len(mmap_tr) > 0, '❌ train mmap is empty'
assert len(mmap_te) > 0, '❌ test mmap is empty — increase NUM_SAMPLES'
print(f"✅ {len(mmap_tr)} train | {len(mmap_te)} test spectrograms saved")

# ── Step 10: Negative datasets ────────────────────────────────────────────────
print("\n[Step 10] Downloading negative datasets...")
from huggingface_hub import hf_hub_download, list_repo_files
HF_TOKEN = os.environ.get('HF_TOKEN', '') or None
repo_id, repo_type = 'kahrendt/microwakeword', 'dataset'
zip_files = [f for f in list_repo_files(repo_id, repo_type=repo_type, token=HF_TOKEN)
             if f.endswith('.zip')]
for fname in zip_files:
    base = os.path.splitext(os.path.basename(fname))[0]
    if not os.path.exists(f'negative_datasets/{base}'):
        print(f"  Downloading {fname}...")
        local = hf_hub_download(repo_id=repo_id, filename=fname,
                                repo_type=repo_type, token=HF_TOKEN)
        with zipfile.ZipFile(local, 'r') as zf:
            zf.extractall('negative_datasets')
hf_cache = os.path.expanduser('~/.cache/huggingface')
if os.path.exists(hf_cache):
    shutil.rmtree(hf_cache)
print("✅ Negative datasets ready")

# ── Detect GPU early (needed for batch_size in config) ────────────────────────
_has_gpu = False
_gpu_env = {**os.environ}
if _cudnn_lib:
    _gpu_env['LD_LIBRARY_PATH'] = f"{_cudnn_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
try:
    _gpu_probe = subprocess.run(
        [sys.executable, '-c',
         'import tensorflow as tf; print(len(tf.config.list_physical_devices("GPU")))'],
        text=True, capture_output=True, timeout=60, env=_gpu_env)
    _has_gpu = _gpu_probe.stdout.strip() not in ('0', '')
except Exception:
    pass

# Refine LOW_RESOURCE now that we know about GPU
LOW_RESOURCE = (not _has_gpu) and TOTAL_RAM_GB < 12

# ── Step 11: Training config ──────────────────────────────────────────────────
print("\n[Step 11] Writing training config...")
neg_dirs = sorted([d for d in os.listdir('negative_datasets')
                   if os.path.isdir(f'negative_datasets/{d}')
                   and not d.startswith('__')])
eval_dirs  = [d for d in neg_dirs if 'eval' in d]
train_dirs = [d for d in neg_dirs if 'eval' not in d]

neg_features = []
for d in train_dirs:
    neg_features.append({
        'features_dir': f'negative_datasets/{d}', 'sampling_weight': 10.0,
        'penalty_weight': 1.0, 'truth': False,
        'truncation_strategy': 'random', 'type': 'mmap'
    })
for d in eval_dirs:
    neg_features.append({
        'features_dir': f'negative_datasets/{d}', 'sampling_weight': 0.0,
        'penalty_weight': 1.0, 'truth': False,
        'truncation_strategy': 'split', 'type': 'mmap'
    })

config = {
    'window_step_ms': 10,
    'train_dir': 'trained_models/wakeword',
    'spectrogram_length': 204,
    'stride': 3,
    'features': [
        {   # positive train — used during training
            'features_dir': 'generated_augmented_features',
            'sampling_weight': 2.0, 'penalty_weight': 1.0, 'truth': True,
            'truncation_strategy': 'truncate_start', 'type': 'mmap'
        },
        {   # positive test — eval only (sampling_weight=0 keeps it out of training)
            'features_dir': 'generated_augmented_features',
            'sampling_weight': 0.0, 'penalty_weight': 1.0, 'truth': True,
            'truncation_strategy': 'split', 'type': 'mmap'
        },
    ] + neg_features,
    'training_steps': [TRAINING_STEPS],
    'positive_class_weight': [1],
    'negative_class_weight': [20],
    'learning_rates': [0.001],
    'batch_size': 16 if LOW_RESOURCE else (32 if not _has_gpu else 128),
    'eval_step_interval': 500,
    'clip_duration_ms': 1500,
    'target_minimization': 0.9,
    'minimization_metric': None,
    'maximization_metric': 'average_viable_recall',
}
with open('training_parameters.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print(f"✅ Config written (batch_size={config['batch_size']}, GPU={'yes' if _has_gpu else 'no'})")

# ── Step 12: Train ────────────────────────────────────────────────────────────
_skip_training = DRY_RUN or (LOW_RESOURCE and not FORCE_TRAIN)

if _skip_training:
    _reason = "dry-run mode" if DRY_RUN else f"low-resource machine ({TOTAL_RAM_GB:.1f} GB RAM, no GPU)"
    print(f"\n[Step 12] ⏭ Skipping training — {_reason}.")
    print(f"         Pipeline validated successfully up to this point.")
    print(f"         To train on this machine anyway: python3 train.py --force-train")
    print(f"         For best results, run on a machine with a GPU.\n")
else:
    _avail_ram = _get_ram_gb('MemAvailable')
    if _avail_ram < 4.0 and not FORCE_TRAIN:
        print(f"\n[Step 12] ⚠ Only {_avail_ram:.1f} GB RAM available — training would likely freeze this machine.")
        print(f"         Skipping training. Use --force-train to override.\n")
        _skip_training = True

if not _skip_training:
    print(f"\n[Step 12] Training (~{max(1, TRAINING_STEPS // 10000)} hour(s))...")
    if os.path.exists('trained_models/wakeword'):
        shutil.rmtree('trained_models/wakeword')

    # Build training env — explicitly pass the cuDNN-augmented LD_LIBRARY_PATH
    train_env = {
        **os.environ,
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_CPP_MIN_LOG_LEVEL': '2',
    }
    if _cudnn_lib:
        train_env['LD_LIBRARY_PATH'] = f"{_cudnn_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    # Limit TF threads on low-resource machines to reduce memory pressure
    if LOW_RESOURCE:
        train_env['TF_NUM_INTEROP_THREADS'] = '1'
        train_env['TF_NUM_INTRAOP_THREADS'] = '2'

    if _has_gpu:
        # Verify cuDNN availability before launching the long training run
        print("[Step 12] GPU detected — verifying cuDNN...")
        cudnn_check = subprocess.run(
            [sys.executable, '-c', '''
import os, ctypes, glob
ld = os.environ.get("LD_LIBRARY_PATH", "")
print(f"LD_LIBRARY_PATH = {ld}")
for p in ld.split(":"):
    libs = glob.glob(os.path.join(p, "libcudnn*.so*"))
    if libs:
        print(f"  cuDNN libs in {p}: {[os.path.basename(l) for l in libs[:5]]}")
try:
    libcudnn = ctypes.CDLL("libcudnn.so.9")
    ver = libcudnn.cudnnGetVersion()
    major, minor, patch = ver // 10000, (ver % 10000) // 100, ver % 100
    print(f"cuDNN runtime version: {major}.{minor}.{patch}")
    if (major, minor) < (9, 3):
        print(f"CUDNN_TOO_OLD")
except Exception as e:
    print(f"cuDNN version check failed: {e}")
try:
    import tensorflow as tf
    print(f"TF version: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs visible: {gpus}")
    if gpus:
        with tf.device("/GPU:0"):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0], [1.0]])
            c = tf.matmul(a, b)
            print(f"GPU matmul OK: {c.numpy().flatten()}")
except Exception as e:
    print(f"GPU check failed: {e}")
'''],
            text=True, env=train_env, capture_output=True
        )
        print(cudnn_check.stdout)
        if 'CUDNN_TOO_OLD' in (cudnn_check.stdout or ''):
            print("ERROR: cuDNN runtime version is < 9.3.0 — TF 2.18 requires >= 9.3.0.")
            print("       Try: pip install nvidia-cudnn-cu12>=9.3,<10")
            sys.exit(1)
        if cudnn_check.stderr:
            for line in cudnn_check.stderr.splitlines():
                if 'cudnn' in line.lower() or 'CuDNN' in line:
                    print(f"  WARN: {line.strip()}")
    else:
        print("[Step 12] No GPU detected — falling back to CPU-only mode.")
        print("         Training on CPU is supported but will be significantly slower.")
        print("         For production training, a GPU instance is strongly recommended.")
        print(f"\n         Note: {TRAINING_STEPS} steps on CPU will be very slow.")
        print(f"         This is fine for testing/debugging your configuration.")
        print(f"         For the final training run, use a GPU instance (any GPU helps).\n")

    result = subprocess.run([
        sys.executable, '-m', 'microwakeword.model_train_eval',
        '--training_config=training_parameters.yaml',
        '--train=1', '--restore_checkpoint', '0',
        '--test_tf_nonstreaming', '0',
        '--test_tflite_nonstreaming', '0',
        '--test_tflite_nonstreaming_quantized', '0',
        '--test_tflite_streaming', '0',
        '--test_tflite_streaming_quantized', '1',
        'mixednet',
        '--pointwise_filters', '64,64,64,64',
        '--repeat_in_block', '1, 1, 1, 1',
        '--mixconv_kernel_sizes', '[5], [7,11], [9,15], [23]',
        '--residual_connection', '0,0,0,0',
        '--first_conv_filters', '32',
        '--first_conv_kernel_size', '5',
        '--stride', '3',
    ], text=True, env=train_env, stderr=subprocess.PIPE)

    MODEL_SRC = ('trained_models/wakeword/tflite_stream_state_internal_quant/'
                 'stream_state_internal_quant.tflite')

    if not os.path.exists(MODEL_SRC):
        print("\n❌ Training failed — model file not found.")
        print("   Common causes:")
        print("   1. cuDNN version mismatch (check errors above)")
        print("   2. Out of GPU memory (reduce batch_size in config)")
        print("   3. Corrupt training data\n")
        if result.stderr:
            print("--- Last 4000 chars of stderr ---")
            print(result.stderr[-4000:])
        elif result.returncode != 0:
            print(f"   Process exited with code {result.returncode} (no stderr captured)")
        sys.exit(1)

    print("✅ Training complete")

    # ── Step 13: Generate JSON + optionally push to GitHub ──────────────────
    size_kb = os.path.getsize(MODEL_SRC) / 1024
    print(f"\n[Step 13] Saving model ({size_kb:.1f} KB)...")

    MODEL_DEST = f'models/{TARGET_WORD}.tflite'
    JSON_DEST  = f'models/{TARGET_WORD}.json'

    os.makedirs(os.path.join(REPO_ROOT, 'models'), exist_ok=True)
    shutil.copy(MODEL_SRC, os.path.join(REPO_ROOT, MODEL_DEST))

    wake_word_json = {
        'type': 'micro',
        'wake_word': TARGET_WORD.replace('_', ' '),
        'author': 'Training Bot',
        'version': 2,
        'model': f'{TARGET_WORD}.tflite',
        'trained_languages': ['en'],
        'micro': {
            'probability_cutoff': 0.97,
            'sliding_window_size': 5,
            'feature_step_size': 10,
            'tensor_arena_size': 22860,
            'minimum_esphome_version': '2024.7.0'
        }
    }
    with open(os.path.join(REPO_ROOT, JSON_DEST), 'w') as f:
        json.dump(wake_word_json, f, indent=2)

    print(f"  Model: {os.path.join(REPO_ROOT, MODEL_DEST)}")
    print(f"  JSON:  {os.path.join(REPO_ROOT, JSON_DEST)}")

    if USE_GITHUB:
        git_configure()
        subprocess.run(['git', 'add', MODEL_DEST, JSON_DEST], check=True, cwd=REPO_ROOT)
        subprocess.run(['git', 'commit', '-m',
                        f'Trained model: {TARGET_WORD} ({size_kb:.1f} KB)'], check=True, cwd=REPO_ROOT)
        git_push()
        print(f"\n🎉 Done! Model + JSON pushed to {GITHUB_REPO}/models/")
        print(f"   ESPHome URL: https://raw.githubusercontent.com/"
              f"{GITHUB_REPO}/main/models/{TARGET_WORD}.json")
    else:
        print(f"\n🎉 Done! Model + JSON saved locally.")
        print(f"   To use with ESPHome, host the .json file and update the URL.")
