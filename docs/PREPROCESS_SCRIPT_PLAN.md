# Unified Preprocessing Script — Simple Plan

Combine `audio_processing.py`, `style_extraction.py`, and `lyrics_tokenization.py` into a single CLI script at `preprocess/preprocess_data.py`.

---

## What It Does

Takes a directory of audio files (with matching `.lrc` lyrics files) and produces the three `.pt` files per sample that DiffRhythm training expects, plus a `train.scp` manifest.

```
Input:                              Output:
songs/                              dataset/
├── song001.mp3                     ├── latent/song001.pt   [1, 64, T]
├── song001.lrc                     ├── style/song001.pt    [1, 512]
├── song002.wav                     ├── lrc/song001.pt      {time, lrc}
├── song002.lrc                     ├── latent/song002.pt
└── ...                             ├── style/song002.pt
                                    ├── lrc/song002.pt
                                    └── train.scp
```

Audio and LRC files are matched by filename stem (e.g. `song001.mp3` pairs with `song001.lrc`). If no `.lrc` exists, the track is treated as instrumental and gets a dummy lyrics entry.

---

## Usage

```bash
python preprocess/preprocess_data.py \
    --input-dir /path/to/songs \
    --output-dir dataset \
    --device cuda
```

That's it for the common case. The script:
1. Finds all audio files in `--input-dir`
2. Loads the VAE and MuQ models once
3. For each file: encodes latent, extracts style, tokenizes lyrics
4. Writes `train.scp` at the end

---

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir` | *required* | Directory with audio (`.mp3`, `.wav`, `.flac`) and `.lrc` files |
| `--output-dir` | `"dataset"` | Where to write `latent/`, `style/`, `lrc/`, and `train.scp` |
| `--device` | `"cuda"` | PyTorch device: `cuda`, `cpu`, or `mps` |
| `--cache-dir` | `"./pretrained"` | Where to cache HuggingFace model downloads |
| `--skip-existing` | on | Skip samples whose `.pt` files already exist (resume support). Use `--no-skip-existing` to force re-process. |
| `--verbose` / `-v` | off | Print per-file progress |

---

## How It Works

### Modules Used

The script is a thin wrapper. All real work is done by the existing modules:

| Module | What it does | Model loaded |
|--------|-------------|--------------|
| `preprocess.audio_processing` | `load_vae()`, `encode_audio_to_latent()`, `save_latent()` | `ASLP-lab/DiffRhythm-vae` |
| `preprocess.style_extraction` | `load_muq()`, `extract_style_from_audio()`, `save_style_file()` | `OpenMuQ/MuQ-MuLan-large` |
| `preprocess.lyrics_tokenization` | `tokenize_lyrics()`, `create_instrumental_lrc()`, `save_lrc_tokens()` | None (in-repo G2P) |

### Processing Flow

```
1. Parse CLI args
2. Scan --input-dir for audio files, pair with .lrc files by stem
3. Filter out already-processed samples (if --skip-existing)
4. Load VAE and MuQ models onto --device
5. For each sample:
   (a) encode_audio_to_latent()  →  save to  latent/{id}.pt
   (b) extract_style_from_audio()  →  save to  style/{id}.pt
   (c) tokenize_lyrics()  →  save to  lrc/{id}.pt
       (or create_instrumental_lrc() if no .lrc file)
6. Write train.scp manifest
7. Print summary (N processed, N skipped, N failed)
```

### File Discovery

```python
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}

for each file in input_dir:
    if file.suffix in AUDIO_EXTENSIONS:
        sample_id = file.stem           # "song001"
        lrc_path  = file.with_suffix('.lrc')  # "song001.lrc" or None
```

Only scans the top-level directory (no recursion into subdirectories).

### Skip-Existing Check

A sample is skipped only if **all three** output files exist:

```python
def should_skip(sample_id, output_dir):
    return (
        exists(f"{output_dir}/latent/{sample_id}.pt") and
        exists(f"{output_dir}/style/{sample_id}.pt") and
        exists(f"{output_dir}/lrc/{sample_id}.pt")
    )
```

If a previous run crashed mid-sample (only some files written), it gets re-processed.

---

## Error Handling

- Each sample is wrapped in try/except. If one file fails, the error is printed and processing continues with the next file.
- Only fully-successful samples are added to `train.scp`.
- A summary is printed at the end showing how many succeeded, failed, and were skipped.

---

## Manifest Format

The `train.scp` file uses the pipe-delimited format that `dataset/dataset.py` expects:

```
song001|dataset/lrc/song001.pt|dataset/latent/song001.pt|dataset/style/song001.pt
song002|dataset/lrc/song002.pt|dataset/latent/song002.pt|dataset/style/song002.pt
```

---

## Script Outline

```python
#!/usr/bin/env python3
"""Unified DiffRhythm data preprocessing script."""

import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

from tqdm import tqdm
import librosa

from preprocess.audio_processing import (
    load_vae, encode_audio_to_latent, save_latent,
)
from preprocess.style_extraction import (
    load_muq, extract_style_from_audio, save_style_file,
)
from preprocess.lyrics_tokenization import (
    tokenize_lyrics, create_instrumental_lrc, save_lrc_tokens,
)

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}


@dataclass
class Sample:
    sample_id: str
    audio_path: str
    lrc_path: Optional[str]   # None for instrumental


def discover_samples(input_dir: str) -> List[Sample]:
    """Find audio/LRC pairs by matching filenames."""
    ...


def should_skip(sample_id: str, output_dir: str) -> bool:
    """Check if all three output files already exist."""
    ...


def process_sample(sample, output_dir, vae, muq, device) -> bool:
    """Run all three stages for one sample. Returns True on success."""
    # 1. Latent
    latent = encode_audio_to_latent(sample.audio_path, vae, device=device)
    save_latent(latent, f"{output_dir}/latent/{sample.sample_id}.pt")

    # 2. Style
    style = extract_style_from_audio(sample.audio_path, muq, device=device)
    save_style_file(style, f"{output_dir}/style/{sample.sample_id}.pt")

    # 3. Lyrics
    if sample.lrc_path:
        lrc_data = tokenize_lyrics(sample.lrc_path)
    else:
        duration = librosa.get_duration(path=sample.audio_path)
        lrc_data = create_instrumental_lrc(duration)
    save_lrc_tokens(lrc_data, f"{output_dir}/lrc/{sample.sample_id}.pt")

    return True


def write_manifest(sample_ids, output_dir):
    """Write train.scp for all successful samples."""
    ...


def main():
    parser = argparse.ArgumentParser(description="Preprocess data for DiffRhythm training")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="dataset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="./pretrained")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Discover and filter
    samples = discover_samples(args.input_dir)
    if args.skip_existing:
        samples = [s for s in samples if not should_skip(s.sample_id, args.output_dir)]

    # Load models
    vae = load_vae(device=args.device, cache_dir=args.cache_dir)
    muq = load_muq(device=args.device, cache_dir=args.cache_dir)

    # Process
    succeeded = []
    failed = []
    for sample in tqdm(samples):
        try:
            process_sample(sample, args.output_dir, vae, muq, args.device)
            succeeded.append(sample.sample_id)
        except Exception as e:
            print(f"FAILED {sample.sample_id}: {e}")
            failed.append(sample.sample_id)

    # Manifest
    write_manifest(succeeded, args.output_dir)

    # Summary
    print(f"Done: {len(succeeded)} succeeded, {len(failed)} failed")
```
