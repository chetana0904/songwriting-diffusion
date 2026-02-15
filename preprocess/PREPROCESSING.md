# Preprocessing for DiffRhythm Training

This script takes a folder of audio files and produces the training data that DiffRhythm expects.

## What You Need

1. **Audio files** — MP3, WAV, or FLAC files of your songs.
2. **LRC files** (optional) — Timestamped lyrics for each song. If a song has no `.lrc` file, it's treated as instrumental.
3. **A GPU** (recommended) — Processing runs on CUDA, MPS (Apple Silicon), or CPU. GPU is much faster.

Models are downloaded automatically from HuggingFace on the first run (~2 GB total).

## Quick Start

```bash
python preprocess/preprocess_data.py --input-dir /path/to/songs --output-dir dataset
```

That's it. When it finishes, you'll have a `dataset/` folder ready for training.

## Input Format

Put your audio files and matching `.lrc` files in a single directory. They're paired by filename:

```
my_songs/
├── track01.mp3
├── track01.lrc      <- lyrics for track01
├── track02.wav
├── track02.lrc      <- lyrics for track02
├── instrumental.flac <- no .lrc file = treated as instrumental
└── ...
```

Supported audio formats: `.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`

### LRC File Format

Standard LRC with timestamps in `[MM:SS.xx]` format:

```
[00:10.00]First line of lyrics
[00:15.23]Second line of lyrics
[00:20.45]Third line of lyrics
```

## Output

The script produces three `.pt` files per song plus a manifest:

```
dataset/
├── latent/
│   └── track01.pt      # VAE-encoded audio  [1, 64, T]
├── style/
│   └── track01.pt      # MuQ style embedding [1, 512]
├── lrc/
│   └── track01.pt      # Tokenized lyrics {time, lrc}
└── train.scp           # Manifest file for training
```

The `train.scp` manifest is what you point your training config at.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir` | *required* | Directory with your audio and `.lrc` files |
| `--output-dir` | `dataset` | Where to write the processed output |
| `--device` | `cuda` | `cuda`, `mps`, or `cpu` |
| `--cache-dir` | `./pretrained` | Where to store downloaded models |
| `--skip-existing` | on | Skip files already processed. Use `--no-skip-existing` to redo everything. |
| `--verbose` / `-v` | off | Show detailed per-file progress |

## Resuming After a Crash

Just run the same command again. By default, the script skips any song that already has all three output files, so it picks up where it left off.

```bash
# Same command — already-processed files are skipped automatically
python preprocess/preprocess_data.py --input-dir /path/to/songs --output-dir dataset
```

## Example: Apple Silicon Mac

```bash
python preprocess/preprocess_data.py \
    --input-dir ~/Music/my_songs \
    --output-dir dataset \
    --device mps \
    --verbose
```

## Example: NVIDIA GPU

```bash
python preprocess/preprocess_data.py \
    --input-dir /data/songs \
    --output-dir dataset \
    --device cuda
```

## Example: CPU Only

```bash
python preprocess/preprocess_data.py \
    --input-dir /data/songs \
    --output-dir dataset \
    --device cpu
```

CPU is significantly slower but works without any GPU.

## What Happens Under the Hood

For each audio file, the script runs three steps:

1. **VAE Latent Encoding** — Resamples audio to 44.1kHz stereo, normalizes to -6 dBFS, encodes through the DiffRhythm VAE into a compressed latent representation.
2. **Style Embedding** — Extracts a 10-second clip from the middle of the track, runs it through MuQ-MuLan at 24kHz to get a 512-dimensional style vector.
3. **Lyrics Tokenization** — Parses the `.lrc` file, converts text to phoneme tokens via G2P (grapheme-to-phoneme), and shifts token IDs by +1 to reserve 0 for padding.

If no `.lrc` file exists, step 3 creates a minimal dummy entry so the training dataset doesn't crash on instrumental tracks.

## Audio Requirements

- **Minimum duration**: 10 seconds (required by the MuQ style model)
- **Sample rate**: Any (automatically resampled to 44.1kHz for the VAE and 24kHz for MuQ)
- **Channels**: Any (automatically converted to stereo for the VAE)

## Running the Tests

```bash
python preprocess/test/test_preprocess_data.py
```

Auto-detects your device. Force a specific one with:

```bash
python preprocess/test/test_preprocess_data.py cpu
```

## Troubleshooting

**"No audio files found"** — Check that `--input-dir` points to the directory containing your audio files (not a parent directory). Only the top level is scanned, not subdirectories.

**GPU out of memory** — The script uses chunked encoding for long audio to avoid this. If it still happens, try `--device cpu` for the problematic files, or split very long audio files (>10 minutes) into shorter segments.

**"Audio file is too short"** — The MuQ style model needs at least 10 seconds of audio. Remove or pad shorter files.

**Missing model downloads** — Models are downloaded from HuggingFace on the first run. If you're behind a firewall, download `ASLP-lab/DiffRhythm-vae` and `OpenMuQ/MuQ-MuLan-large` manually into `--cache-dir`.
