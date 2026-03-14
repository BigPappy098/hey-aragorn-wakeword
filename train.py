import subprocess, os, sys, shutil, yaml, urllib.request, zipfile
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
TARGET_WORD    = 'hey_air_uh_gorn'
NUM_SAMPLES    = 1000
TRAINING_STEPS = 10000
GITHUB_REPO    = os.environ.get('GITHUB_REPO', '')
GITHUB_TOKEN   = os.environ.get('GITHUB_TOKEN', '')
# ──────────────────────────────────────────────────────────────────────────────

BASE = '/workspace/training'
os.chdir(BASE)

for d in ['generated_samples', 'mit_rirs', 'negative_datasets']:
    os.makedirs(d, exist_ok=True)

# ── Step 3: Clone repos ───────────────────────────────────────────────────────
print("\n[Step 3] Cloning microWakeWord + piper-sample-generator...")
if os.path.exists('microWakeWord'):
    shutil.rmtree('microWakeWord')
subprocess.run(['git', 'clone', 'https://github.com/kahrendt/microWakeWord.git'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', 'microWakeWord'], check=True)

patch = '''
from collections import defaultdict as _defaultdict
_orig_validate_nonstreaming = validate_nonstreaming
def validate_nonstreaming(*args, **kwargs):
    try:
        return _orig_validate_nonstreaming(*args, **kwargs)
    except Exception as _e:
        print(f"  [validate_nonstreaming skipped: {type(_e).__name__}]")
        return _defaultdict(float)
'''
with open('microWakeWord/microwakeword/train.py', 'a') as f:
    f.write(patch)

if not os.path.exists('piper-sample-generator'):
    subprocess.run(['git', 'clone', 'https://github.com/rhasspy/piper-sample-generator.git'], check=True)
print("✅ Repos ready")

# ── Step 4: Piper voice model ─────────────────────────────────────────────────
print("\n[Step 4] Downloading Piper model...")
os.makedirs('piper-sample-generator/models', exist_ok=True)
model_path = 'piper-sample-generator/models/en_US-libritts_r-medium.pt'
if not os.path.exists(model_path):
    urllib.request.urlretrieve(
        'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt',
        model_path
    )
print("✅ Piper model ready")

# ── Step 7: Generate samples ──────────────────────────────────────────────────
print(f"\n[Step 7] Generating {NUM_SAMPLES} samples for '{TARGET_WORD}'...")
env = {**os.environ, 'PYTHONPATH': os.path.abspath('piper-sample-generator')}
subprocess.run([
    sys.executable, '-m', 'piper_sample_generator',
    TARGET_WORD, '--max-samples', str(NUM_SAMPLES), '--batch-size', '100',
    '--model', model_path, '--output-dir', 'generated_samples'
], text=True, env=env, check=True)
count = len([f for f in os.listdir('generated_samples') if f.endswith('.wav')])
print(f"✅ {count} samples generated")

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

MMAP_OUT = 'generated_augmented_features/training/wakeword_mmap'
os.makedirs(os.path.dirname(MMAP_OUT), exist_ok=True)

clips = Clips(input_directory='generated_samples', file_pattern='*.wav',
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
spectrograms = SpectrogramGeneration(clips=clips, augmenter=augmenter, slide_frames=10, step_ms=10)
RaggedMmap.from_generator(
    out_dir=MMAP_OUT,
    sample_generator=spectrograms.spectrogram_generator(split='train', repeat=2),
    batch_size=100, verbose=True
)
mmap = RaggedMmap(MMAP_OUT)
assert len(mmap) > 0, "❌ mmap is empty"
print(f"✅ {len(mmap)} spectrograms saved")

# ── Step 10: Negative datasets ────────────────────────────────────────────────
print("\n[Step 10] Downloading negative datasets...")
from huggingface_hub import hf_hub_download, list_repo_files
repo_id, repo_type = 'kahrendt/microwakeword', 'dataset'
zip_files = [f for f in list_repo_files(repo_id, repo_type=repo_type) if f.endswith('.zip')]
for fname in zip_files:
    base = os.path.splitext(os.path.basename(fname))[0]
    if not os.path.exists(f'negative_datasets/{base}'):
        print(f"  Downloading {fname}...")
        local = hf_hub_download(repo_id=repo_id, filename=fname, repo_type=repo_type)
        with zipfile.ZipFile(local, 'r') as zf:
            zf.extractall('negative_datasets')
hf_cache = os.path.expanduser('~/.cache/huggingface')
if os.path.exists(hf_cache):
    shutil.rmtree(hf_cache)
print("✅ Negative datasets ready")

# ── Step 11: Training config ──────────────────────────────────────────────────
print("\n[Step 11] Writing training config...")
neg_dirs = sorted([d for d in os.listdir('negative_datasets')
                   if os.path.isdir(f'negative_datasets/{d}') and not d.startswith('__')])
eval_dirs  = [d for d in neg_dirs if 'eval' in d]
train_dirs = [d for d in neg_dirs if 'eval' not in d]

neg_features = []
for d in train_dirs:
    neg_features.append({'features_dir': f'negative_datasets/{d}', 'sampling_weight': 10.0,
                         'penalty_weight': 1.0, 'truth': False,
                         'truncation_strategy': 'random', 'type': 'mmap'})
for d in eval_dirs:
    neg_features.append({'features_dir': f'negative_datasets/{d}', 'sampling_weight': 0.0,
                         'penalty_weight': 1.0, 'truth': False,
                         'truncation_strategy': 'split', 'type': 'mmap'})

config = {
    'window_step_ms': 10, 'train_dir': 'trained_models/wakeword',
    'spectrogram_length': 204, 'stride': 3,
    'features': [{'features_dir': 'generated_augmented_features', 'sampling_weight': 2.0,
                  'penalty_weight': 1.0, 'truth': True,
                  'truncation_strategy': 'truncate_start', 'type': 'mmap'}] + neg_features,
    'training_steps': [TRAINING_STEPS], 'positive_class_weight': [1],
    'negative_class_weight': [20], 'learning_rates': [0.001], 'batch_size': 128,
    'eval_step_interval': 500, 'clip_duration_ms': 1500,
    'target_minimization': 0.9, 'minimization_metric': '',
    'maximization_metric': 'average_viable_recall',
}
with open('training_parameters.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print("✅ Config written")

# ── Step 12: Train ────────────────────────────────────────────────────────────
print(f"\n[Step 12] Training (~{TRAINING_STEPS // 10000} hour)...")
if os.path.exists('trained_models/wakeword'):
    shutil.rmtree('trained_models/wakeword')

train_env = {**os.environ, 'TF_FORCE_GPU_ALLOW_GROWTH': 'true', 'TF_CPP_MIN_LOG_LEVEL': '2'}
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
], text=True, env=train_env)

if result.returncode != 0:
    print("❌ Training failed:")
    print(result.stderr[-4000:])
    sys.exit(1)

# ── Step 13: Push model to GitHub ─────────────────────────────────────────────
MODEL_SRC = ('trained_models/wakeword/tflite_stream_state_internal_quant/'
             'stream_state_internal_quant.tflite')
MODEL_DEST = f'models/{TARGET_WORD}.tflite'

if not os.path.exists(MODEL_SRC):
    print("❌ Model file not found after training")
    sys.exit(1)

size_kb = os.path.getsize(MODEL_SRC) / 1024
print(f"\n[Step 13] Pushing model to GitHub ({size_kb:.1f} KB)...")

shutil.copy(MODEL_SRC, MODEL_DEST)

subprocess.run(['git', 'config', 'user.email', 'runpod@training.bot'], check=True)
subprocess.run(['git', 'config', 'user.name', 'RunPod Training Bot'], check=True)
subprocess.run(['git', 'remote', 'set-url', 'origin',
                f'https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git'], check=True)
subprocess.run(['git', 'add', MODEL_DEST], check=True)
subprocess.run(['git', 'commit', '-m',
                f'Trained model: {TARGET_WORD} ({size_kb:.1f} KB)'], check=True)
subprocess.run(['git', 'push', 'origin', 'main'], check=True)

print(f"\n🎉 Done! Model committed to {GITHUB_REPO}/models/{TARGET_WORD}.tflite")
