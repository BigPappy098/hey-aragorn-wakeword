import subprocess, os as _os
_cudnn_lib = subprocess.check_output(
    ['python3', '-c', 'import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))'],
    text=True).strip() + '/lib'
_os.environ['LD_LIBRARY_PATH'] = f"{_cudnn_lib}:{_os.environ.get('LD_LIBRARY_PATH', '')}"

import subprocess, os, sys, shutil, yaml, urllib.request, zipfile, json
import numpy as np

# ── INTERACTIVE CONFIG ────────────────────────────────────────────────────────
print("\n=== microWakeWord Custom Trainer ===\n")

while True:
    raw = input("Enter wake word phonetically (e.g. hey_air_uh_gorn): ").strip()
    if raw:
        break
TARGET_WORD = raw.lower().replace(' ', '_')
print(f"  \u2192 Using: {TARGET_WORD}")

raw_samples = input("\nNumber of voice samples [default: 1000, better quality: 2000-5000]: ").strip()
NUM_SAMPLES = int(raw_samples) if raw_samples.isdigit() else 1000
print(f"  \u2192 Samples: {NUM_SAMPLES}")

raw_steps = input("\nTraining steps [default: 20000, better quality: 20000-30000]: ").strip()
TRAINING_STEPS = int(raw_steps) if raw_steps.isdigit() else 20000
print(f"  \u2192 Training steps: {TRAINING_STEPS}")

print()
GITHUB_REPO  = os.environ.get('GITHUB_REPO', '')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')

BASE = '/workspace/training'
os.chdir(BASE)

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
    print(f"\U0001f3a4 Found {len(real_audio_files)} real recording file(s) in {REAL_DIR}/")
    print(f"   Will use 50/50 split: {NUM_SAMPLES // 2} synthetic + {NUM_SAMPLES // 2} augmented real")
    USING_REAL = True
else:
    print(f"\U0001f4c2 No real recordings found in {REAL_DIR}/")
    ans = input("Would you like to upload real recordings to GitHub first? [y/n]: ").strip().lower()
    if ans == 'y':
        print(f"\n\U0001f4e4 Upload your recording file(s) here:")
        print(f"   https://github.com/{GITHUB_REPO}/upload/main/real_recordings")
        print(f"\n   Supported formats: .wav  .mp3  .m4a  .flac")
        print(f"   Tip: One long file with 10-50 repetitions, 1-2 second pause between each.\n")
        input("Press Enter when your files are uploaded and committed to GitHub...")
        subprocess.run(['git', 'remote', 'set-url', 'origin',
                        f'https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git'], check=True)
        subprocess.run(['git', 'pull', 'origin', 'main'], check=True)
        real_audio_files = scan_real_recordings()
        if real_audio_files:
            print(f"\u2705 Found {len(real_audio_files)} recording file(s) \u2014 using 50/50 split.")
            USING_REAL = True
        else:
            print("\u26a0\ufe0f  No files found after pull. Continuing with synthetic-only.")
    else:
        print("\U0001f4e2 Skipping real recordings \u2014 using 100% synthetic samples.")

SYNTHETIC_COUNT = NUM_SAMPLES // 2 if USING_REAL else NUM_SAMPLES
REAL_TARGET     = NUM_SAMPLES // 2 if USING_REAL else 0

# ── Step 3: Clone repos ───────────────────────────────────────────────────────
print("\n[Step 3] Cloning microWakeWord + piper-sample-generator...")
if os.path.exists('microWakeWord'):
    shutil.rmtree('microWakeWord')
subprocess.run(['git', 'clone', 'https://github.com/kahrendt/microWakeWord.git'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', 'microWakeWord'], check=True)

patch_lines = [
    'from collections import defaultdict as _defaultdict\n',
    '_orig_validate_nonstreaming = validate_nonstreaming\n',
    'def validate_nonstreaming(*args, **kwargs):\n',
    '    try:\n',
    '        return _orig_validate_nonstreaming(*args, **kwargs)\n',
    '    except Exception as _e:\n',
    '        print(f" [validate_nonstreaming skipped: {type(_e).__name__}]")\n',
    '        return _defaultdict(float)\n',
]
with open('microWakeWord/microwakeword/train.py', 'a') as f:
    f.writelines(patch_lines)

if not os.path.exists('piper-sample-generator'):
    subprocess.run(['git', 'clone', 'https://github.com/rhasspy/piper-sample-generator.git'], check=True)
print("\u2705 Repos ready")

# ── Step 4: Piper voice model ─────────────────────────────────────────────────
print("\n[Step 4] Downloading Piper model...")
os.makedirs('piper-sample-generator/models', exist_ok=True)
model_path = 'piper-sample-generator/models/en_US-libritts_r-medium.pt'
if not os.path.exists(model_path):
    urllib.request.urlretrieve(
        'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt',
        model_path)
print("\u2705 Piper model ready")

# ── Preview step ──────────────────────────────────────────────────────────────
print("\n[Preview] Generating 1 sample to preview pronunciation...")
preview_dir = '/tmp/preview_sample'
if os.path.exists(preview_dir):
    shutil.rmtree(preview_dir)
os.makedirs(preview_dir)

env = {**os.environ, 'PYTHONPATH': os.path.abspath('piper-sample-generator')}
subprocess.run([
    sys.executable, '-m', 'piper_sample_generator',
    TARGET_WORD, '--max-samples', '1', '--batch-size', '1',
    '--model', model_path, '--output-dir', preview_dir
], text=True, env=env, check=True)

preview_files = [f for f in os.listdir(preview_dir) if f.endswith('.wav')]
if preview_files:
    preview_dest = f'models/preview_{TARGET_WORD}.wav'
    shutil.copy(os.path.join(preview_dir, preview_files[0]), preview_dest)
    subprocess.run(['git', 'config', 'user.email', 'runpod@training.bot'], check=True)
    subprocess.run(['git', 'config', 'user.name',  'RunPod Training Bot'], check=True)
    subprocess.run(['git', 'remote', 'set-url', 'origin',
                    f'https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git'], check=True)
    subprocess.run(['git', 'add', preview_dest], check=True)
    subprocess.run(['git', 'commit', '-m', f'Preview sample: {TARGET_WORD}'], check=True)
    subprocess.run(['git', 'push', 'origin', 'main'], check=True)
    print(f"\n\U0001f50a Preview pushed to GitHub!")
    print(f"   Go to: https://github.com/{GITHUB_REPO}/blob/main/{preview_dest}")
    print(f"   Download the .wav file and play it to check pronunciation.\n")
    while True:
        answer = input("Does the pronunciation sound correct? [y/n]: ").strip().lower()
        if answer in ('y', 'n'):
            break
    if answer == 'n':
        print("\n\u274c Exiting. Re-run train.py and try a different phonetic spelling.")
        sys.exit(0)
print("\u2705 Pronunciation confirmed, continuing...\n")

# ── Step 7: Generate synthetic samples ───────────────────────────────────────
print(f"\n[Step 7] Generating {SYNTHETIC_COUNT} synthetic samples for '{TARGET_WORD}'...")
if os.path.exists('generated_samples'):
    shutil.rmtree('generated_samples')
os.makedirs('generated_samples')
subprocess.run([
    sys.executable, '-m', 'piper_sample_generator',
    TARGET_WORD, '--max-samples', str(SYNTHETIC_COUNT), '--batch-size', '100',
    '--model', model_path, '--output-dir', 'generated_samples'
], text=True, env=env, check=True)
synth_count = len([f for f in os.listdir('generated_samples') if f.endswith('.wav')])
print(f"\u2705 {synth_count} synthetic samples generated")

# ── Step 7b: Augment real recordings ─────────────────────────────────────────
if USING_REAL:
    print(f"\n[Step 7b] Processing real recordings \u2192 target {REAL_TARGET} augmented clips...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pydub', 'soundfile', 'scipy'], check=True)
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    import soundfile as sf
    from scipy import signal as scipy_signal
    raw_clips = []
    for fpath in real_audio_files:
        print(f"  Splitting {os.path.basename(fpath)}...")
        audio = AudioSegment.from_file(fpath).set_channels(1).set_frame_rate(16000)
        clips = split_on_silence(audio, min_silence_len=350, silence_thresh=audio.dBFS - 14, keep_silence=100)
        valid = [c for c in clips if 300 <= len(c) <= 3500]
        print(f"    {len(valid)} valid clips found (of {len(clips)} total)")
        raw_clips.extend(valid)
    print(f"  Total individual clips: {len(raw_clips)}")
    if len(raw_clips) == 0:
        print("  \u26a0\ufe0f  No valid clips detected \u2014 falling back to synthetic-only.")
        USING_REAL = False
    else:
        reps_needed = max(1, -(-REAL_TARGET // len(raw_clips)))
        augmented_count = 0
        for i, clip in enumerate(raw_clips):
            if augmented_count >= REAL_TARGET:
                break
            orig_path = f'generated_samples/real_{i:04d}_orig.wav'
            clip.export(orig_path, format='wav')
            augmented_count += 1
            audio_np, sr = sf.read(orig_path)
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            for rep in range(reps_needed):
                if augmented_count >= REAL_TARGET:
                    break
                aug = audio_np.copy()
                speed = np.random.uniform(0.85, 1.15)
                aug = scipy_signal.resample(aug, int(len(aug) / speed))
                aug = aug * np.random.uniform(0.6, 1.4)
                aug = aug + np.random.randn(len(aug)) * np.random.uniform(0.0, 0.008)
                aug = np.clip(aug, -1.0, 1.0).astype(np.float32)
                sf.write(f'generated_samples/real_{i:04d}_aug{rep:03d}.wav', aug, sr)
                augmented_count += 1
        total_samples = len([f for f in os.listdir('generated_samples') if f.endswith('.wav')])
        print(f"\u2705 {augmented_count} real/augmented clips added")
        print(f"\u2705 {total_samples} total samples ({synth_count} synthetic + {augmented_count} real)")

# ── Step 8: Augmentation data ─────────────────────────────────────────────────
print("\n[Step 8] Downloading RIRs + background noise (~1.3 GB)...")
if not os.listdir('mit_rirs'):
    subprocess.run(['wget', '-q', '--show-progress', '-O', '/tmp/rirs_noises.zip',
                    'https://www.openslr.org/resources/28/rirs_noises.zip'], check=True)
    subprocess.run(['unzip', '-q', '/tmp/rirs_noises.zip', '-d', 'mit_rirs'], check=True)
    os.remove('/tmp/rirs_noises.zip')
print("\u2705 Augmentation data ready")

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
assert len(mmap) > 0, '\u274c mmap is empty'
print(f"\u2705 {len(mmap)} spectrograms saved")

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
print("\u2705 Negative datasets ready")

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
print("\u2705 Config written")

# ── Step 12: Train ────────────────────────────────────────────────────────────
print(f"\n[Step 12] Training (~{max(1, TRAINING_STEPS // 10000)} hour(s))...")
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
MODEL_SRC = ('trained_models/wakeword/tflite_stream_state_internal_quant/'
             'stream_state_internal_quant.tflite')
if not os.path.exists(MODEL_SRC):
    print("\u274c Training failed \u2014 model file not found")
    if result.stderr:
        print(result.stderr[-4000:])
    sys.exit(1)
print("\u2705 Training complete")

# ── Step 13: Generate JSON + Push to GitHub ───────────────────────────────────
size_kb = os.path.getsize(MODEL_SRC) / 1024
print(f"\n[Step 13] Generating JSON and pushing to GitHub ({size_kb:.1f} KB)...")
MODEL_DEST = f'models/{TARGET_WORD}.tflite'
JSON_DEST  = f'models/{TARGET_WORD}.json'
shutil.copy(MODEL_SRC, MODEL_DEST)
wake_word_json = {
    'type': 'micro',
    'wake_word': TARGET_WORD.replace('_', ' '),
    'author': 'RunPod Training Bot',
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
with open(JSON_DEST, 'w') as f:
    json.dump(wake_word_json, f, indent=2)
subprocess.run(['git', 'add', MODEL_DEST, JSON_DEST], check=True)
subprocess.run(['git', 'commit', '-m', f'Trained model: {TARGET_WORD} ({size_kb:.1f} KB)'], check=True)
subprocess.run(['git', 'push', 'origin', 'main'], check=True)
print(f"\n\U0001f389 Done! Model + JSON committed to {GITHUB_REPO}/models/")
print(f"   ESPHome URL: https://raw.githubusercontent.com/{GITHUB_REPO}/main/models/{TARGET_WORD}.json")