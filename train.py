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
_CUDNN_VERSION_SPEC = '>=9.3,<10'  # TF 2.18 needs cuDNN >= 9.3
_CUDNN_LOCAL = '/usr/local/lib/cudnn-9.3'
_cudnn_lib = None

def _ensure_cudnn_93():
    """Install cuDNN >= 9.3 to a side directory without disturbing pip packages."""
    lib_dir = _os.path.join(_CUDNN_LOCAL, 'nvidia', 'cudnn', 'lib')
    if _os.path.isdir(lib_dir) and _glob.glob(_os.path.join(lib_dir, 'libcudnn.so*')):
        return lib_dir  # already installed
    print(f"[init] Installing cuDNN ({_CUDNN_VERSION_SPEC}) to {_CUDNN_LOCAL}...")
    _sp.run([_sys.executable, '-m', 'pip', 'install', '-q',
             '--no-deps', f'--target={_CUDNN_LOCAL}',
             f'nvidia-cudnn-cu12{_CUDNN_VERSION_SPEC}'],
            check=True)
    return lib_dir

try:
    # First check if the existing pip cuDNN is already >= 9.3
    _existing = _sp.check_output(
        [_sys.executable, '-c',
         'from importlib.metadata import version; print(version("nvidia-cudnn-cu12"))'],
        text=True, timeout=30, stderr=_sp.DEVNULL).strip()
    _existing_ok = tuple(int(x) for x in _existing.split('.')[:2]) >= (9, 3)
except Exception:
    _existing_ok = False

if _existing_ok:
    # Pip package claims >= 9.3, try to use it directly
    try:
        _cudnn_lib = _sp.check_output(
            [_sys.executable, '-c',
             'import nvidia.cudnn, os; '
             'f = getattr(nvidia.cudnn, "__file__", None); '
             'print(os.path.dirname(f) if f else "")'],
            text=True, timeout=30, stderr=_sp.DEVNULL).strip()
        if _cudnn_lib:
            _cudnn_lib += '/lib'
        else:
            _existing_ok = False
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

def _check_nvidia_smi():
    """Early GPU/driver sanity check. Prints info and returns True if GPU usable."""
    import subprocess as sp

    has_smi = shutil.which('nvidia-smi')
    has_dev = os.path.exists('/dev/nvidia0')

    if not has_smi and not has_dev:
        print("[hw] No NVIDIA GPU detected (no nvidia-smi, no /dev/nvidia0).")
        return False

    if has_smi:
        try:
            # Try structured query first (not all drivers support all fields)
            r = sp.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                         '--format=csv,noheader'], text=True, capture_output=True, timeout=15)
            if r.returncode == 0 and r.stdout.strip():
                for line in r.stdout.strip().splitlines():
                    print(f"[hw] GPU: {line.strip()}")
                return True
            # Fall back to plain nvidia-smi
            r = sp.run(['nvidia-smi'], text=True, capture_output=True, timeout=15)
            if r.returncode == 0:
                for line in r.stdout.splitlines()[:4]:
                    if line.strip():
                        print(f"[hw] {line.strip()}")
                return True
            # nvidia-smi failed but /dev/nvidia0 exists — GPU likely usable
            if has_dev:
                print(f"[hw] nvidia-smi returned exit {r.returncode} but /dev/nvidia0 "
                      f"exists — assuming GPU is available")
                return True
            print(f"[hw] nvidia-smi failed (exit {r.returncode}): {r.stderr.strip()}")
            return False
        except Exception as e:
            if has_dev:
                print(f"[hw] nvidia-smi errored ({e}) but /dev/nvidia0 exists — "
                      f"assuming GPU is available")
                return True
            print(f"[hw] nvidia-smi check failed: {e}")
            return False

    # No nvidia-smi but /dev/nvidia0 exists (container with GPU pass-through)
    print("[hw] /dev/nvidia0 detected (nvidia-smi not in PATH) — GPU likely available")
    return True

TOTAL_RAM_GB = _get_ram_gb('MemTotal')
HAS_NVIDIA_DRIVER = _check_nvidia_smi()

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
    """Push to GitHub, merging remote changes if needed.

    Uses merge (not rebase) when histories diverge to avoid losing binary
    files like .tflite models.  ``-X ours`` keeps our local version on
    any conflict — correct because RunPod has the freshly trained files.
    """
    subprocess.run(
        ['git', 'fetch', 'origin', branch],
        cwd=REPO_ROOT, capture_output=True, text=True
    )

    # Try pushing directly first (fast-forward)
    result = subprocess.run(
        ['git', 'push', 'origin', branch],
        cwd=REPO_ROOT, capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  ✅ Pushed to {branch}")
        return

    # Push rejected — remote has new commits.  Merge them in.
    print("  [git] Remote has new commits — merging...")
    merge = subprocess.run(
        ['git', 'merge', f'origin/{branch}', '-X', 'ours',
         '--allow-unrelated-histories',
         '-m', 'Merge remote changes (keep local trained files)'],
        cwd=REPO_ROOT, capture_output=True, text=True
    )
    if merge.returncode != 0:
        # -X ours doesn't auto-resolve delete/modify conflicts (e.g. remote
        # deleted a model file that we just re-created).  Force-keep ours.
        print("  [git] Resolving delete/modify conflicts (keeping our files)...")
        # Stage all our versions of conflicted files
        subprocess.run(['git', 'checkout', '--ours', '.'],
                       cwd=REPO_ROOT, capture_output=True)
        subprocess.run(['git', 'add', '-A'],
                       cwd=REPO_ROOT, capture_output=True)
        commit = subprocess.run(
            ['git', 'commit', '--no-edit'],
            cwd=REPO_ROOT, capture_output=True, text=True
        )
        if commit.returncode != 0:
            # Last resort: abort and report
            subprocess.run(['git', 'merge', '--abort'],
                           cwd=REPO_ROOT, capture_output=True)
            print("  ❌ Merge failed — see stderr below")
            if merge.stderr:
                print(f"     {merge.stderr.strip()}")
            sys.exit(1)
        print("  [git] Conflicts resolved")

    # Retry push after merge
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

raw_samples = prompt("\nNumber of voice samples [default: 10000, better quality: 20000-50000]: ",
                     default='10000')
NUM_SAMPLES = int(raw_samples) if raw_samples.isdigit() else 10000
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
          'personal_augmented_features', 'personal_samples_trimmed',
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
            print(f"   Options:")
            print(f"     A) One long file per person with 10-50 repetitions, 1-2 sec pauses.")
            print(f"     B) Individual clips: speaker01_take01.wav, speaker01_take02.wav, etc.")
            print(f"   Multiple speakers improve model quality!\n")
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
        print("   Tip: Record one long file per person with 10-50 repetitions,")
        print("        or individual clips named speaker01_take01.wav, speaker01_take02.wav, etc.")
        print("📢 Continuing with 100% synthetic samples.")

SYNTHETIC_COUNT = NUM_SAMPLES // 2 if USING_REAL else NUM_SAMPLES
REAL_TARGET     = NUM_SAMPLES // 2 if USING_REAL else 0

# ── Step 3: Clone repos ───────────────────────────────────────────────────────
print("\n[Step 3] Cloning microWakeWord + piper-sample-generator...")
if os.path.exists('microWakeWord'):
    shutil.rmtree('microWakeWord')
subprocess.run(['git', 'clone', 'https://github.com/TaterTotterson/micro-wake-word.git',
                'microWakeWord'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', 'microWakeWord'],
               check=True)
# Explicit installs matching the working Colab notebook
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'tensorboard', 'datasets<3', 'onnxruntime', 'tqdm'],
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

# Fix numpy 2.x compat: getattr(np,'trapezoid',np.trapz) crashes because
# Python eagerly evaluates np.trapz even when trapezoid exists.
_src = _src.replace(
    "getattr(np, 'trapezoid', np.trapz)",
    "np.trapezoid if hasattr(np, 'trapezoid') else np.trapz",
)

_patch = CUT_MARKER + r'''
import tensorflow as _tf_patch

# numpy >= 2.0 renamed trapz -> trapezoid; use hasattr to avoid eager eval crash
_trapz_fn = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

def validate_nonstreaming(config, data_processor, model, test_set):
    """Memory-efficient, Keras 3-compatible replacement for validate_nonstreaming.
    Streams validation data from mmap via tf.data.Dataset.from_generator()
    instead of loading everything into RAM at once."""

    features_length = config["spectrogram_length"]
    bs = config["batch_size"]

    total_size = data_processor.get_mode_size(test_set)
    if total_size == 0:
        return {
            "accuracy": 0, "recall": 0, "precision": 0, "auc": 0, "loss": 99,
            "recall_at_no_faph": 0, "cutoff_for_no_faph": 0,
            "ambient_false_positives": 0, "ambient_false_positives_per_hour": 0,
            "average_viable_recall": 0,
        }

    def _make_gen(mode, trunc):
        """Returns a generator function that streams (spectrogram, label) pairs
        from all feature providers' mmap files — no bulk RAM allocation."""
        def gen():
            for provider in data_processor.feature_providers:
                if provider.get_mode_size(mode) == 0:
                    continue
                label = np.array([provider.label], dtype=np.float32)
                for spec in provider.get_feature_generator(mode, features_length, trunc):
                    yield spec.astype(np.float32), label
        return gen

    out_sig = (
        _tf_patch.TensorSpec(shape=(features_length, 40), dtype=_tf_patch.float32),
        _tf_patch.TensorSpec(shape=(1,), dtype=_tf_patch.float32),
    )

    # Count samples and compute batch steps so Keras knows when data ends
    gen_fn = _make_gen(test_set, "truncate_start")
    n_samples = sum(1 for _ in gen_fn())
    n_steps = n_samples // bs

    if n_steps == 0:
        return {
            "accuracy": 0, "recall": 0, "precision": 0, "auc": 0, "loss": 99,
            "recall_at_no_faph": 0, "cutoff_for_no_faph": 0,
            "ambient_false_positives": 0, "ambient_false_positives_per_hour": 0,
            "average_viable_recall": 0,
        }

    ds = _tf_patch.data.Dataset.from_generator(
        gen_fn, output_signature=out_sig
    ).batch(bs, drop_remainder=True).prefetch(_tf_patch.data.AUTOTUNE)

    model.reset_metrics()
    result = model.evaluate(ds, steps=n_steps, return_dict=True, verbose=0)

    if isinstance(result, (list, tuple)):
        names = [m.name for m in model.metrics]
        result = dict(zip(names, result))

    def _as_numpy(v):
        return v.numpy() if hasattr(v, 'numpy') else np.array(v)

    metrics = {}
    metrics["accuracy"] = result["accuracy"]
    metrics["recall"] = result["recall"]
    metrics["precision"] = result["precision"]
    metrics["auc"] = result["auc"]
    metrics["loss"] = result["loss"]
    metrics["recall_at_no_faph"] = 0
    metrics["cutoff_for_no_faph"] = 0
    metrics["ambient_false_positives"] = 0
    metrics["ambient_false_positives_per_hour"] = 0
    metrics["average_viable_recall"] = 0

    test_set_fp = _as_numpy(result["fp"])

    if data_processor.get_mode_size("validation_ambient") > 0:
        amb_gen_fn = _make_gen(test_set + "_ambient", "split")
        amb_count = sum(1 for _ in amb_gen_fn())
        amb_steps = amb_count // bs

        if amb_steps > 0:
            amb_ds = _tf_patch.data.Dataset.from_generator(
                amb_gen_fn, output_signature=out_sig
            ).batch(bs, drop_remainder=True).prefetch(_tf_patch.data.AUTOTUNE)

            with swap_attribute(model, "reset_metrics", lambda: None):
                ambient_predictions = model.evaluate(
                    amb_ds, steps=amb_steps, return_dict=True, verbose=0,
                )

            if isinstance(ambient_predictions, (list, tuple)):
                names = [m.name for m in model.metrics]
                ambient_predictions = dict(zip(names, ambient_predictions))

            duration_of_ambient_set = (
                data_processor.get_mode_duration("validation_ambient") / 3600.0
            )

            all_true_positives = _as_numpy(ambient_predictions["tp"])
            ambient_false_positives = _as_numpy(ambient_predictions["fp"]) - test_set_fp
            all_false_negatives = _as_numpy(ambient_predictions["fn"])

            metrics["auc"] = ambient_predictions["auc"]
            metrics["loss"] = ambient_predictions["loss"]

            recall_at_cutoffs = (
                all_true_positives / (all_true_positives + all_false_negatives)
            )
            faph_at_cutoffs = ambient_false_positives / duration_of_ambient_set

            target_faph_cutoff_probability = 1.0
            recall_at_no_faph = 0
            for index, cutoff in enumerate(np.linspace(0.0, 1.0, 101)):
                if faph_at_cutoffs[index] == 0:
                    target_faph_cutoff_probability = cutoff
                    recall_at_no_faph = recall_at_cutoffs[index]
                    break

            if faph_at_cutoffs[0] > 2:
                index_of_first_viable = 1
                while faph_at_cutoffs[index_of_first_viable] > 2:
                    index_of_first_viable += 1
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
            metrics["ambient_false_positives"] = ambient_false_positives[50]
            metrics["ambient_false_positives_per_hour"] = faph_at_cutoffs[50]
            metrics["average_viable_recall"] = average_viable_recall

    return metrics
'''

with open(_train_py_path, 'w') as f:
    f.write(_src + _patch)

# Also fix np.trapz calls in test.py (NumPy 2.x removed trapz, renamed to trapezoid)
_test_py_path = 'microWakeWord/microwakeword/test.py'
if os.path.exists(_test_py_path):
    with open(_test_py_path, 'r') as f:
        _test_src = f.read()
    if 'np.trapz(' in _test_src:
        _test_src = _test_src.replace('np.trapz(', '(np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(')
        if 'import numpy as np' not in _test_src:
            _test_src = 'import numpy as np\n' + _test_src
        with open(_test_py_path, 'w') as f:
            f.write(_test_src)
        print("[patch] Fixed np.trapz → np.trapezoid in test.py")

if not os.path.exists('piper-sample-generator'):
    subprocess.run(
        ['git', 'clone', 'https://github.com/TaterTotterson/piper-sample-generator.git'],
        check=True)
else:
    # Ensure we have TaterTotterson's fork, not rhasspy's
    _remote = subprocess.run(
        ['git', '-C', 'piper-sample-generator', 'remote', 'get-url', 'origin'],
        capture_output=True, text=True).stdout.strip()
    if 'TaterTotterson' not in _remote:
        print("[dep] Replacing rhasspy piper-sample-generator with TaterTotterson's fork...")
        shutil.rmtree('piper-sample-generator')
        subprocess.run(
            ['git', 'clone', 'https://github.com/TaterTotterson/piper-sample-generator.git'],
            check=True)
print("✅ Repos ready")

# piper-sample-generator is run via subprocess with its repo dir on sys.path.
# We do NOT pip-install it because its pyproject.toml pins numpy>=2 and
# audiomentations==0.33 which conflict with TF's numpy 1.x.
PSG_DIR = os.path.abspath('piper-sample-generator')

# Add to PYTHONPATH permanently so all subprocesses can find the module.
_existing_pp = os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = PSG_DIR + (':' + _existing_pp if _existing_pp else '')

# Verify the module is actually findable
_main_py = os.path.join(PSG_DIR, 'piper_sample_generator', '__main__.py')
if not os.path.isfile(_main_py):
    print(f"ERROR: Cannot find {_main_py}")
    print(f"  PSG_DIR = {PSG_DIR}")
    print(f"  Contents: {os.listdir(PSG_DIR) if os.path.isdir(PSG_DIR) else 'DIR NOT FOUND'}")
    sys.exit(1)
print(f"  piper-sample-generator: {PSG_DIR}")

# ── Ensure PyTorch + piper-tts are available (needed by piper-sample-generator)
def _detect_cuda_version():
    """Detect the CUDA driver version to install a matching PyTorch build."""
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            # Get CUDA version from nvidia-smi
            cuda_out = subprocess.run(
                ['nvidia-smi'], capture_output=True, text=True)
            # Parse "CUDA Version: XX.Y" from nvidia-smi output
            import re
            m = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', cuda_out.stdout)
            if m:
                return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return None, None

def _pytorch_install_args():
    """Return pip install args for the best PyTorch CUDA build."""
    _is_gpu = os.path.exists('/dev/nvidia0') or shutil.which('nvidia-smi')
    if not _is_gpu:
        return ['--index-url', 'https://download.pytorch.org/whl/cpu']
    major, minor = _detect_cuda_version()
    if major is None:
        return []  # default index, let pip figure it out
    cuda_tag = f"cu{major}{minor}"
    # PyTorch provides wheels for specific CUDA versions (cu118, cu121, cu124, cu126, etc.)
    # Pick the best matching one that doesn't exceed the driver's CUDA version
    available = ['cu118', 'cu121', 'cu124', 'cu126']
    best = None
    for tag in available:
        tag_ver = int(tag[2:])  # e.g. 'cu124' → 124
        driver_ver = major * 10 + minor  # e.g. 12, 4 → 124
        if tag_ver <= driver_ver:
            best = tag
    if best:
        print(f"[dep] CUDA driver {major}.{minor} detected → using PyTorch {best}")
        return ['--index-url', f'https://download.pytorch.org/whl/{best}']
    return []  # fallback to default

_need_pytorch_install = False
try:
    import torch  # noqa: F401
    # Check if existing PyTorch can actually see the GPU
    if (os.path.exists('/dev/nvidia0') or shutil.which('nvidia-smi')):
        if not torch.cuda.is_available():
            print(f"[dep] PyTorch {torch.__version__} installed but can't see GPU — reinstalling...")
            _need_pytorch_install = True
        else:
            print(f"[dep] PyTorch {torch.__version__} with CUDA — OK")
    else:
        print(f"[dep] PyTorch {torch.__version__} (CPU) — OK")
except ImportError:
    _need_pytorch_install = True

if _need_pytorch_install:
    _pt_args = _pytorch_install_args()
    _is_gpu = os.path.exists('/dev/nvidia0') or shutil.which('nvidia-smi')
    _mode = "GPU" if _is_gpu else "CPU-only"
    print(f"[dep] Installing PyTorch ({_mode})...")
    # Uninstall first — pip won't swap CUDA variants (cu130→cu124) without this
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y',
                    'torch', 'torchvision', 'torchaudio'],
                   capture_output=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'torch', 'torchvision', 'torchaudio'] + _pt_args,
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

# ── Ensure datasets uses soundfile for audio decoding ────────────────────────
# datasets 3.x requires torchcodec for audio, but torchcodec needs a matching
# PyTorch ABI + FFmpeg version — it won't work on most RunPod images.
# datasets <3 uses soundfile/librosa which works everywhere and is fine for WAVs.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'datasets<3'],
               check=True)
os.environ['HF_AUDIO_DECODER'] = 'soundfile'
print("[dep] datasets<3 + soundfile (audio decoder)")

# ── Ensure system packages are installed (ffmpeg, espeak-ng) ─────────────────
_sys_pkgs = []
if not shutil.which('ffmpeg'):
    _sys_pkgs.append('ffmpeg')
if not shutil.which('espeak-ng'):
    _sys_pkgs.append('espeak-ng')
    _sys_pkgs.append('libespeak-ng-dev')
if _sys_pkgs:
    print(f"[dep] Installing system packages: {' '.join(_sys_pkgs)}...")
    _apt = ['apt-get', 'install', '-y', '-q'] + _sys_pkgs
    if os.getuid() != 0:
        _apt = ['sudo'] + _apt
    try:
        subprocess.run(_apt, check=True, capture_output=True)
        print(f"[dep] System packages installed")
    except Exception as e:
        print(f"[dep] WARNING: Could not install system packages ({e}).")
else:
    print("[dep] System packages OK (ffmpeg, espeak-ng)")

# ── Step 4: Piper voice model ─────────────────────────────────────────────────
print("\n[Step 4] Downloading Piper model...")
os.makedirs('piper-sample-generator/models', exist_ok=True)
model_path = 'piper-sample-generator/models/en_US-libritts_r-medium.pt'
if not os.path.exists(model_path):
    urllib.request.urlretrieve(
        'https://github.com/TaterTotterson/piper-sample-generator/releases/download/'
        'models/en_US-libritts_r-medium.pt',
        model_path)
# Also download the companion JSON config if missing
_model_json = model_path + '.json'
if not os.path.exists(_model_json):
    try:
        urllib.request.urlretrieve(
            'https://github.com/TaterTotterson/piper-sample-generator/releases/download/'
            'models/en_US-libritts_r-medium.pt.json',
            _model_json)
    except Exception:
        pass  # Not critical — generator works without it
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

if not HAS_AVX2:
    os.environ['ATEN_CPU_CAPABILITY'] = 'default'
subprocess.run([
    sys.executable, '-m', 'piper_sample_generator',
    TARGET_WORD, '--max-samples', '1', '--batch-size', '1',
    '--model', model_path, '--output-dir', preview_dir
], text=True, check=True)

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
], text=True, check=True)
synth_count = len([f for f in os.listdir('generated_samples') if f.endswith('.wav')])
print(f"✅ {synth_count} synthetic samples generated")

# ── Step 7b: Process + augment real recordings ───────────────────────────────
if USING_REAL:
    print(f"\n[Step 7b] Processing real recordings → target {REAL_TARGET} augmented clips...")
    # soundfile + librosa should already be installed; ensure they are
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'soundfile', 'librosa', 'scipy', 'tqdm'],
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
        if dur_s <= 4.0:
            # Short file — treat as a single clip (individual take recording)
            print(f"    Short clip — using as-is")
            raw_clips.append(audio_np)
        else:
            # Long file — split into individual utterances
            clips = adaptive_split(audio_np, sr)
            print(f"    {len(clips)} valid clips extracted")
            raw_clips.extend(clips)

    print(f"  Total individual clips: {len(raw_clips)}")

    if len(raw_clips) == 0:
        print("  ⚠️  No valid clips detected — falling back to synthetic-only.")
        USING_REAL = False
    else:
        # Write trimmed clips to a separate directory (NOT generated_samples/).
        # They'll be run through microWakeWord's full augmentation pipeline in
        # Step 9b (same RIR, background noise, EQ, etc. as synthetic samples)
        # and stored in personal_augmented_features/ with higher sampling weight.
        PERSONAL_CLIPS_DIR = 'personal_samples_trimmed'
        if os.path.exists(PERSONAL_CLIPS_DIR):
            shutil.rmtree(PERSONAL_CLIPS_DIR)
        os.makedirs(PERSONAL_CLIPS_DIR, exist_ok=True)
        for i, clip in enumerate(raw_clips):
            sf.write(f'{PERSONAL_CLIPS_DIR}/clip_{i:04d}.wav', clip, 16000)
        print(f"✅ {len(raw_clips)} trimmed clips saved to {PERSONAL_CLIPS_DIR}/")
        print(f"   (Will be augmented via microWakeWord pipeline in Step 9b)")

# ── Step 8: Augmentation data — RIRs + background noise ──────────────────────
# Match TaterTotterson's full augmentation pipeline:
#   impulse_paths: MIT RIRs (resampled to 16k)
#   background_paths: WHAM + CHiME + FMA + AudioSet (all 16k mono WAV)
print("\n[Step 8] Downloading augmentation datasets (RIRs + background noise)...")
os.makedirs('training_datasets', exist_ok=True)

import scipy.io.wavfile as _wavfile

def _convert_to_16k(src_dir, dst_dir, glob_pattern='*.wav', label='audio'):
    """Convert all audio files in src_dir matching glob_pattern to 16k mono WAV."""
    import librosa as _lr
    from pathlib import Path
    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    existing = set(os.listdir(dst_dir)) if os.path.isdir(dst_dir) else set()
    files = list(src.rglob(glob_pattern))
    if not files:
        print(f"    No {glob_pattern} files found in {src_dir}")
        return 0
    ok, skipped, bad = 0, 0, 0
    for i, p in enumerate(files):
        out_name = '__'.join(p.relative_to(src).parts)  # flatten dir structure
        if not out_name.lower().endswith('.wav'):
            out_name = p.stem + '.wav'
        if out_name in existing:
            skipped += 1
            continue
        try:
            y, _ = _lr.load(str(p), sr=16000, mono=True)
            if y.size == 0:
                raise ValueError("empty")
            x = np.clip(y, -1.0, 1.0)
            _wavfile.write(str(dst / out_name), 16000, (x * 32767).astype(np.int16))
            ok += 1
        except Exception:
            bad += 1
        if (i + 1) % 500 == 0 or (i + 1) == len(files):
            print(f"    {label}: {i+1}/{len(files)} ({ok} ok, {skipped} skip, {bad} fail)")
    print(f"  {label}: {ok} converted, {skipped} skipped, {bad} failed")
    return ok + skipped

def _download_extract(url, archive_name, extract_dir, extract_cmd=None):
    """Download an archive and extract it, skipping if extract_dir already has files."""
    archive_path = f'training_datasets/downloads/{archive_name}'
    os.makedirs('training_datasets/downloads', exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    if not os.path.exists(archive_path):
        print(f"    Downloading {archive_name}...")
        try:
            subprocess.run(['wget', '-q', '--show-progress', '-O', archive_path, url],
                           check=True, timeout=3600)
        except Exception as e:
            print(f"    WARNING: Download failed ({e}). Trying curl...")
            subprocess.run(['curl', '-sfL', url, '-o', archive_path],
                           check=True, timeout=3600)
    if extract_cmd:
        print(f"    Extracting {archive_name}...")
        subprocess.run(extract_cmd, check=True)
    else:
        # Auto-detect by extension
        if archive_name.endswith('.zip'):
            print(f"    Extracting {archive_name}...")
            subprocess.run(['unzip', '-q', '-o', archive_path, '-d', extract_dir], check=True)
        elif archive_name.endswith('.tar.gz') or archive_name.endswith('.tgz'):
            print(f"    Extracting {archive_name}...")
            subprocess.run(['tar', '-xzf', archive_path, '-C', extract_dir], check=True)
        elif archive_name.endswith('.tar'):
            print(f"    Extracting {archive_name}...")
            subprocess.run(['tar', '-xf', archive_path, '-C', extract_dir], check=True)
    # Clean up archive to save space
    if os.path.exists(archive_path):
        os.remove(archive_path)
        print(f"    Removed {archive_name} (saving disk space)")

# --- MIT RIRs (room impulse responses) ---
MIT_RIRS_16K = 'training_datasets/mit_rirs_16k'
if not os.path.isdir('mit_rirs') or not os.listdir('mit_rirs'):
    _download_extract(
        'https://www.openslr.org/resources/28/rirs_noises.zip',
        'rirs_noises.zip', 'mit_rirs')
if not os.path.isdir(MIT_RIRS_16K) or len(os.listdir(MIT_RIRS_16K)) < 10:
    print("  Converting MIT RIRs → 16k mono WAV...")
    _convert_to_16k('mit_rirs', MIT_RIRS_16K, '*.wav', 'MIT RIRs')
else:
    print("  MIT RIRs 16k: already converted")

# --- WHAM (real-world noise from urban environments) ---
WHAM_16K = 'training_datasets/wham_16k'
if not os.path.isdir(WHAM_16K) or len(os.listdir(WHAM_16K)) < 10:
    _wham_raw = 'training_datasets/wham'
    if not os.path.isdir(_wham_raw) or len(os.listdir(_wham_raw)) < 5:
        _download_extract(
            'https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip',
            'wham_noise.zip', _wham_raw)
    print("  Converting WHAM → 16k mono WAV...")
    _convert_to_16k(_wham_raw, WHAM_16K, '*.wav', 'WHAM')
    # Clean up raw files to save space
    if os.path.isdir(_wham_raw):
        shutil.rmtree(_wham_raw)
        print("    Removed raw WHAM files (saving disk space)")
else:
    print("  WHAM 16k: already converted")

# --- CHiME-Home (domestic environment recordings) ---
CHIME_16K = 'training_datasets/chime_16k'
if not os.path.isdir(CHIME_16K) or len(os.listdir(CHIME_16K)) < 10:
    _chime_raw = 'training_datasets/chime'
    if not os.path.isdir(_chime_raw) or len(os.listdir(_chime_raw)) < 5:
        _download_extract(
            'https://archive.org/download/chime-home/chime_home.tar.gz',
            'chime_home.tar.gz', _chime_raw)
    print("  Converting CHiME → 16k mono WAV...")
    _convert_to_16k(_chime_raw, CHIME_16K, '*.48kHz.wav', 'CHiME')
    if os.path.isdir(_chime_raw):
        shutil.rmtree(_chime_raw)
        print("    Removed raw CHiME files (saving disk space)")
else:
    print("  CHiME 16k: already converted")

# --- FMA xsmall (Free Music Archive — music background noise) ---
FMA_16K = 'training_datasets/fma_16k'
if not os.path.isdir(FMA_16K) or len(os.listdir(FMA_16K)) < 10:
    _fma_raw = 'training_datasets/fma'
    if not os.path.isdir(_fma_raw) or len(os.listdir(_fma_raw)) < 5:
        _download_extract(
            'https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip',
            'fma_xs.zip', _fma_raw)
    print("  Converting FMA → 16k mono WAV...")
    _convert_to_16k(_fma_raw, FMA_16K, '*.mp3', 'FMA')
    if os.path.isdir(_fma_raw):
        shutil.rmtree(_fma_raw)
        print("    Removed raw FMA files (saving disk space)")
else:
    print("  FMA 16k: already converted")

# --- AudioSet (balanced training — diverse real-world sounds) ---
AUDIOSET_16K = 'training_datasets/audioset_16k'
if not os.path.isdir(AUDIOSET_16K) or len(os.listdir(AUDIOSET_16K)) < 100:
    _audioset_raw = 'training_datasets/audioset'
    os.makedirs(_audioset_raw, exist_ok=True)
    os.makedirs('training_datasets/downloads', exist_ok=True)
    # AudioSet balanced has 10 tarballs — try multiple HuggingFace revisions
    _as_base = 'https://huggingface.co/datasets/agkphysics/AudioSet/resolve'
    _as_revs = [
        '6762f044d1c88619c7f2006486036192128fb07e',
        '0049167e89f259a010c3f070fe3666d9e5242836',
        'main',
    ]
    _as_patterns = ['data/bal_train0', 'data/bal_train/bal_train0']
    _as_rev = None
    _as_pat = None
    for _rev in _as_revs:
        for _pat in _as_patterns:
            _test_url = f'{_as_base}/{_rev}/{_pat}0.tar'
            try:
                _probe = subprocess.run(
                    ['curl', '-I', '-L', '--fail', '-s', _test_url],
                    capture_output=True, timeout=30)
                if _probe.returncode == 0:
                    _as_rev, _as_pat = _rev, _pat
                    break
            except Exception:
                pass
        if _as_rev:
            break

    if _as_rev:
        print(f"  Downloading AudioSet balanced (10 tarballs)...")
        for _i in range(10):
            _tar_name = f'bal_train0{_i}.tar'
            _tar_path = f'training_datasets/downloads/{_tar_name}'
            if not os.path.exists(_tar_path):
                _url = f'{_as_base}/{_as_rev}/{_as_pat}{_i}.tar'
                print(f"    Downloading {_tar_name}...")
                try:
                    subprocess.run(['curl', '-L', '-s', '--fail', _url, '-o', _tar_path],
                                   check=True, timeout=1800)
                except Exception as e:
                    print(f"    WARNING: Failed to download {_tar_name} ({e}), skipping")
                    continue
            if os.path.exists(_tar_path):
                print(f"    Extracting {_tar_name}...")
                subprocess.run(['tar', '-xf', _tar_path, '-C', _audioset_raw], check=False)
                os.remove(_tar_path)
        print("  Converting AudioSet → 16k mono WAV...")
        _convert_to_16k(_audioset_raw, AUDIOSET_16K, '*.flac', 'AudioSet')
        if os.path.isdir(_audioset_raw):
            shutil.rmtree(_audioset_raw)
            print("    Removed raw AudioSet files (saving disk space)")
    else:
        print("  WARNING: Could not locate AudioSet on HuggingFace — skipping.")
        print("           (Training will still work with remaining noise sources)")
else:
    print("  AudioSet 16k: already converted")

# Build the paths lists for the augmenter
_impulse_paths = [MIT_RIRS_16K]
_background_paths = []
for _bp in [WHAM_16K, CHIME_16K, FMA_16K, AUDIOSET_16K]:
    if os.path.isdir(_bp) and os.listdir(_bp):
        _background_paths.append(_bp)
if not _background_paths:
    # Fallback: use the point-source noises from MIT RIRs
    _bg_fallback = 'mit_rirs/RIRS_NOISES/pointsource_noises'
    if os.path.isdir(_bg_fallback):
        _background_paths.append(_bg_fallback)
    print("  WARNING: No background noise datasets available. Using MIT point-source noises as fallback.")
else:
    print(f"  Background noise sources: {len(_background_paths)} datasets ready")
print(f"  Impulse response sources: {len(_impulse_paths)} datasets ready")
print("✅ Augmentation data ready")

# ── Step 9: Spectrograms ──────────────────────────────────────────────────────
print("\n[Step 9] Generating spectrograms...")

# Set TF env vars before importing (match TaterTotterson's augmenter settings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['NVIDIA_TF32_OVERRIDE'] = '1'
os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = '512'
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found')

sys.path.insert(0, 'microWakeWord')
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
from mmap_ninja.ragged import RaggedMmap

augmenter = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={
        'SevenBandParametricEQ': 0.1, 'TanhDistortion': 0.05,
        'PitchShift': 0.15, 'BandStopFilter': 0.1,
        'AddColorNoise': 0.1, 'AddBackgroundNoise': 0.7,
        'Gain': 0.8, 'RIR': 0.7,
    },
    impulse_paths=_impulse_paths,
    background_paths=_background_paths,
    background_min_snr_db=5, background_max_snr_db=10,
    min_jitter_s=0.2, max_jitter_s=0.3,
)

# ── Proper 3-way split (matching TaterTotterson's approach) ──
# Monkey-patch Clips.audio_generator for deterministic 80/10/10 split
import types as _types, random as _random

_split_cfg = {
    'training':   {'name': 'train',      'repetition': 2, 'slide_frames': 10},
    'validation': {'name': 'validation', 'repetition': 1, 'slide_frames': 10},
    'testing':    {'name': 'test',       'repetition': 1, 'slide_frames': 1},
}

def _bind_wav_generator(clips_obj, wav_dir):
    """Patch Clips.audio_generator for deterministic 80/10/10 train/val/test split.
    Matches TaterTotterson's bind_wav_generator approach."""
    import librosa as _lr
    def audio_generator_from_wavs(self, split='train', repeat=1):
        import glob as _g
        files = sorted(_g.glob(os.path.join(wav_dir, '*.wav')))
        if not files:
            return
        rng = _random.Random(10)
        files_shuf = files[:]
        rng.shuffle(files_shuf)
        n = len(files_shuf)
        n_val = max(1, int(0.10 * n))
        n_test = max(1, int(0.10 * n))
        n_train = max(0, n - n_val - n_test)
        splits = {
            'train':      files_shuf[:n_train],
            'validation': files_shuf[n_train:n_train + n_val],
            'test':       files_shuf[n_train + n_val:],
        }
        file_list = splits.get(split, [])
        if not file_list:
            return
        for _ in range(max(1, int(repeat))):
            for p in file_list:
                y, _sr = _lr.load(p, sr=16000, mono=True)
                yield y.astype(np.float32, copy=False)
    clips_obj.audio_generator = _types.MethodType(audio_generator_from_wavs, clips_obj)

def _generate_feature_set(input_wav_dir, out_root_dir, label):
    """Generate augmented spectrogram features with proper 3-way split."""
    import glob as _g
    files = _g.glob(os.path.join(input_wav_dir, '*.wav'))
    if not files:
        print(f"  No WAVs found for {label} in {input_wav_dir} — skipping")
        return False
    print(f"\n  Augmenting {len(files)} samples ({label})...")
    clips = Clips(
        input_directory=input_wav_dir, file_pattern='*.wav',
        max_clip_duration_s=5, remove_silence=True,
        random_split_seed=10, split_count=0.1)
    _bind_wav_generator(clips, input_wav_dir)
    for split_name, cfg in _split_cfg.items():
        out_dir = os.path.join(out_root_dir, split_name, 'wakeword_mmap')
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        print(f"    Writing {split_name} split ({label})...")
        spectros = SpectrogramGeneration(
            clips=clips, augmenter=augmenter,
            slide_frames=cfg['slide_frames'], step_ms=10)
        RaggedMmap.from_generator(
            out_dir=out_dir,
            sample_generator=spectros.spectrogram_generator(
                split=cfg['name'], repeat=cfg['repetition']),
            batch_size=100, verbose=False)
        mmap_check = RaggedMmap(out_dir)
        print(f"      {split_name}: {len(mmap_check)} spectrograms")
    print(f"  ✅ Features ready: {out_root_dir}/*/wakeword_mmap")
    return True

_generate_feature_set('generated_samples', 'generated_augmented_features', 'synthetic')

# ── Step 9b: Generate spectrograms for personal recordings ───────────────────
if USING_REAL and os.path.isdir('personal_samples_trimmed'):
    print("\n[Step 9b] Generating spectrograms for personal recordings...")
    _generate_feature_set('personal_samples_trimmed', 'personal_augmented_features', 'personal')

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
    # Match reference: speech/dinner_party datasets get 12.0, no_speech gets 5.0
    if 'no_speech' in d:
        _neg_weight = 5.0
    else:
        _neg_weight = 12.0
    neg_features.append({
        'features_dir': f'negative_datasets/{d}', 'sampling_weight': _neg_weight,
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
    'features': [
        {   # positive (synthetic) — single entry; microWakeWord reads train/val/test from subdirs
            'features_dir': 'generated_augmented_features',
            'sampling_weight': 2.0, 'penalty_weight': 1.0, 'truth': True,
            'truncation_strategy': 'truncate_start', 'type': 'mmap'
        },
    ] + ([
        {   # personal recordings — weighted 3x for emphasis
            'features_dir': 'personal_augmented_features',
            'sampling_weight': 3.0, 'penalty_weight': 1.0, 'truth': True,
            'truncation_strategy': 'truncate_start', 'type': 'mmap'
        },
    ] if USING_REAL and os.path.isdir('personal_augmented_features') else [])
    + neg_features,
    'training_steps': [TRAINING_STEPS],
    'positive_class_weight': [1],
    'negative_class_weight': [20],
    'learning_rates': [0.001],
    'batch_size': 16,
    'eval_step_interval': 500,
    'clip_duration_ms': 1500,
    'target_minimization': 0.9,
    'minimization_metric': None,
    'maximization_metric': 'average_viable_recall',
    'freq_mask_count': [0],
    'freq_mask_max_size': [0],
    'time_mask_count': [0],
    'time_mask_max_size': [0],
}
with open('training_parameters.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print(f"✅ Config written (batch_size=16, stride=2, GPU={'yes' if _has_gpu else 'no'})")

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

    # Build training env — match TaterTotterson's TF config exactly
    train_env = {
        **os.environ,
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_CPP_MIN_LOG_LEVEL': '2',
        'TF_GPU_ALLOCATOR': 'cuda_malloc_async',
        'TF_XLA_FLAGS': '--tf_xla_auto_jit=0',
        'NVIDIA_TF32_OVERRIDE': '1',
        'TF_CUDNN_WORKSPACE_LIMIT_IN_MB': '512',
        'GLOG_minloglevel': '2',
        'GRPC_VERBOSITY': 'ERROR',
    }
    if _cudnn_lib:
        train_env['LD_LIBRARY_PATH'] = f"{_cudnn_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    # Limit TF threads on low-resource machines to reduce memory pressure
    if LOW_RESOURCE:
        train_env['TF_NUM_INTEROP_THREADS'] = '1'
        train_env['TF_NUM_INTRAOP_THREADS'] = '2'

    if _has_gpu:
        # Verify cuDNN availability before launching the long training run.
        # Tests both matmul AND conv2d — cuDNN errors only surface on convolutions.
        print("[Step 12] GPU detected — verifying CUDA + cuDNN...")
        cudnn_check = subprocess.run(
            [sys.executable, '-c', '''
import os, sys, ctypes, glob

# 1) Show nvidia-smi info
import subprocess
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                          "--format=csv,noheader"], text=True, capture_output=True, timeout=10)
    print(f"GPU: {smi.stdout.strip()}")
except Exception as e:
    print(f"nvidia-smi failed: {e}")

# 2) Check cuDNN shared library version
ld = os.environ.get("LD_LIBRARY_PATH", "")
print(f"LD_LIBRARY_PATH = {ld}")
for p in ld.split(":"):
    if not p:
        continue
    libs = glob.glob(os.path.join(p, "libcudnn*.so*"))
    if libs:
        print(f"  cuDNN libs in {p}: {[os.path.basename(l) for l in libs[:5]]}")
try:
    libcudnn = ctypes.CDLL("libcudnn.so.9")
    ver = libcudnn.cudnnGetVersion()
    major, minor, patch = ver // 10000, (ver % 10000) // 100, ver % 100
    print(f"cuDNN runtime version: {major}.{minor}.{patch}")
    if (major, minor) < (9, 3):
        print("CUDNN_TOO_OLD")
except Exception as e:
    print(f"cuDNN version check failed: {e}")
    print("CUDNN_MISSING")

# 3) TensorFlow GPU + conv2d test (catches cuDNN kernel load errors)
try:
    import tensorflow as tf
    print(f"TF version: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs visible: {gpus}")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        with tf.device("/GPU:0"):
            # matmul test
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            c = tf.matmul(a, a)
            print(f"GPU matmul OK: {c.numpy().flatten()}")
            # conv2d test — this is what actually triggers cuDNN kernel loading
            x = tf.random.normal([1, 8, 8, 3])
            w = tf.random.normal([3, 3, 3, 16])
            y = tf.nn.conv2d(x, w, strides=1, padding="SAME")
            print(f"GPU conv2d OK: output shape {y.shape}")
    else:
        print("GPU_NOT_VISIBLE")
except Exception as e:
    print(f"GPU check failed: {e}")
    print("GPU_CHECK_FAILED")
'''],
            text=True, env=train_env, capture_output=True
        )
        print(cudnn_check.stdout)
        if cudnn_check.stderr:
            # Show cuDNN/CUDA relevant warnings
            for line in cudnn_check.stderr.splitlines():
                ll = line.lower()
                if any(kw in ll for kw in ['cudnn', 'cuda', 'cublas', 'error', 'failed', 'not found']):
                    print(f"  ⚠ {line.strip()}")
        if 'CUDNN_TOO_OLD' in (cudnn_check.stdout or ''):
            print("\n❌ cuDNN runtime version is < 9.3.0 — TF 2.18 requires >= 9.3.0.")
            print("   Fix: pip install --force-reinstall 'nvidia-cudnn-cu12>=9.3,<10'")
            sys.exit(1)
        if 'CUDNN_MISSING' in (cudnn_check.stdout or ''):
            print("\n❌ Could not load libcudnn.so.9 — cuDNN is missing or not in LD_LIBRARY_PATH.")
            print("   Fix: pip install 'nvidia-cudnn-cu12>=9.3,<10'")
            sys.exit(1)
        if 'GPU_NOT_VISIBLE' in (cudnn_check.stdout or ''):
            print("\n⚠ TensorFlow can't see the GPU. Check CUDA_VISIBLE_DEVICES and driver.")
            _has_gpu = False
        if 'GPU_CHECK_FAILED' in (cudnn_check.stdout or ''):
            print("\n❌ GPU check failed — see errors above. Training would crash.")
            print("   Common fixes:")
            print("   1. pip install --force-reinstall 'nvidia-cudnn-cu12>=9.3,<10'")
            print("   2. Ensure CUDA 12.x toolkit is installed")
            print("   3. Check nvidia-smi works correctly")
            sys.exit(1)
    else:
        print("[Step 12] No GPU detected — falling back to CPU-only mode.")
        print("         Training on CPU is supported but will be significantly slower.")
        print("         For production training, a GPU instance is strongly recommended.")
        print(f"\n         Note: {TRAINING_STEPS} steps on CPU will be very slow.")
        print(f"         This is fine for testing/debugging your configuration.")
        print(f"         For the final training run, use a GPU instance (any GPU helps).\n")

    # Stream stderr to console in real-time AND capture it for post-mortem.
    # Previously stderr=subprocess.PIPE hid all CUDA errors until after exit.
    _train_log = os.path.join(BASE, 'training_stderr.log')
    _log_fh = open(_train_log, 'w')
    print(f"  (stderr also logged to {_train_log})")
    train_proc = subprocess.Popen([
        sys.executable, '-m', 'microwakeword.model_train_eval',
        '--training_config=training_parameters.yaml',
        '--train=1', '--restore_checkpoint', '1',
        '--test_tf_nonstreaming', '0',
        '--test_tflite_nonstreaming', '0',
        '--test_tflite_nonstreaming_quantized', '0',
        '--test_tflite_streaming', '0',
        '--test_tflite_streaming_quantized', '1',
        '--use_weights', 'best_weights',
        'mixednet',
        '--pointwise_filters', '64,64,64,64',
        '--repeat_in_block', '1,1,1,1',
        '--mixconv_kernel_sizes', '[5], [7,11], [9,15], [23]',
        '--residual_connection', '0,0,0,0',
        '--first_conv_filters', '32',
        '--first_conv_kernel_size', '5',
        '--stride', '2',
    ], text=True, env=train_env, stderr=subprocess.PIPE)

    # Tee stderr to both console and log file in real-time
    import threading
    def _tee_stderr(proc, log_fh):
        for line in proc.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
            log_fh.write(line)
            log_fh.flush()
    _tee_thread = threading.Thread(target=_tee_stderr, args=(train_proc, _log_fh), daemon=True)
    _tee_thread.start()
    train_proc.wait()
    _tee_thread.join(timeout=5)
    _log_fh.close()

    MODEL_SRC = ('trained_models/wakeword/tflite_stream_state_internal_quant/'
                 'stream_state_internal_quant.tflite')

    if not os.path.exists(MODEL_SRC):
        print("\n❌ Training failed — model file not found.")
        print("   Common causes:")
        print("   1. cuDNN version mismatch (check errors above)")
        print("   2. Out of GPU memory (reduce batch_size in config)")
        print("   3. Corrupt training data\n")
        if os.path.exists(_train_log):
            with open(_train_log) as f:
                _stderr_tail = f.read()[-4000:]
            if _stderr_tail.strip():
                print("--- Last 4000 chars of training stderr ---")
                print(_stderr_tail)
        if train_proc.returncode != 0:
            print(f"\n   Process exited with code {train_proc.returncode}")
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
            'probability_cutoff': 0.5,
            'sliding_window_size': 10,
            'feature_step_size': 10,
            'tensor_arena_size': 30000,
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
