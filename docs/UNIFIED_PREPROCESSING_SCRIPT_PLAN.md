# Unified Preprocessing Script — Implementation Plan

This document outlines how to combine the three existing preprocessing modules (`audio_processing.py`, `style_extraction.py`, `lyrics_tokenization.py`) into a single, production-ready CLI script (`preprocess/preprocess_data.py`) with full command-line option support.

---

## Table of Contents

1. [Goals](#1-goals)
2. [Existing Modules to Integrate](#2-existing-modules-to-integrate)
3. [Command-Line Interface Design](#3-command-line-interface-design)
4. [Subcommand Architecture](#4-subcommand-architecture)
5. [Detailed Option Reference](#5-detailed-option-reference)
6. [Input Discovery Logic](#6-input-discovery-logic)
7. [Processing Pipeline](#7-processing-pipeline)
8. [Output Structure](#8-output-structure)
9. [Error Handling and Resume](#9-error-handling-and-resume)
10. [Implementation Approach](#10-implementation-approach)
11. [Open Questions](#11-open-questions)

---

## 1. Goals

- **Single entry point** — One script to run the entire preprocessing pipeline, or any subset of it.
- **Thin orchestrator** — All heavy logic stays in the existing modules; the script only wires them together, handles CLI parsing, progress reporting, and error recovery.
- **Incremental / resumable** — Skip files that are already processed, so a crashed run can be restarted without re-doing everything.
- **Flexible** — Allow running only certain stages (e.g., just latents, or just lyrics) for debugging or partial re-processing.
- **Simple** — Single process, single GPU, one file at a time. No parallelization complexity.

---

## 2. Existing Modules to Integrate

Each module is already fully implemented and tested. The unified script imports from them directly.

| Module | Key Functions Used | Models Required |
|--------|-------------------|-----------------|
| `preprocess.audio_processing` | `load_vae()`, `encode_audio_to_latent()`, `save_latent()`, `verify_latent()` | `ASLP-lab/DiffRhythm-vae` |
| `preprocess.style_extraction` | `load_muq()`, `extract_style_from_audio()`, `extract_style_from_text()`, `extract_style_combined()`, `save_style_file()`, `verify_style_file()` | `OpenMuQ/MuQ-MuLan-large` |
| `preprocess.lyrics_tokenization` | `tokenize_lyrics()`, `create_instrumental_lrc()`, `save_lrc_tokens()`, `verify_lrc()` | None (uses in-repo G2P) |

---

## 3. Command-Line Interface Design

The script should support two usage patterns:

### Pattern A: Full Pipeline (default)

Run all three stages on a directory of audio + LRC files:

```bash
python preprocess/preprocess_data.py \
    --input-dir /path/to/songs \
    --output-dir dataset \
    --device cuda \
    --manifest train.scp
```

### Pattern B: Subcommands for Individual Stages

Run a single stage for debugging, re-processing, or when stages need different hardware:

```bash
# Only generate latents
python preprocess/preprocess_data.py latent \
    --input-dir /path/to/songs \
    --output-dir dataset \
    --device cuda

# Only extract style embeddings
python preprocess/preprocess_data.py style \
    --input-dir /path/to/songs \
    --output-dir dataset \
    --device cuda

# Only tokenize lyrics
python preprocess/preprocess_data.py lyrics \
    --input-dir /path/to/songs \
    --output-dir dataset

# Only generate the manifest (from already-processed files)
python preprocess/preprocess_data.py manifest \
    --output-dir dataset \
    --manifest train.scp

# Only run verification on existing outputs
python preprocess/preprocess_data.py verify \
    --output-dir dataset
```

When no subcommand is given, the script runs the full pipeline (`all` is implied).

---

## 4. Subcommand Architecture

```
preprocess_data.py
├── (no subcommand / "all")  →  Run full pipeline
├── latent                   →  Audio → VAE latent only
├── style                    →  Audio → MuQ style embedding only
├── lyrics                   →  LRC → Tokenized lyrics only
├── manifest                 →  Generate train.scp from existing .pt files
└── verify                   →  Validate all .pt files and manifest
```

Implementation with `argparse` subparsers, where each subcommand inherits shared options from a parent parser.

---

## 5. Detailed Option Reference

### 5.1 Global Options (all subcommands)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input-dir` | `str` | *required* | Directory containing source audio (`.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`) and `.lrc` files. |
| `--output-dir` | `str` | `"dataset"` | Root output directory. Subdirs `latent/`, `style/`, `lrc/` created automatically. |
| `--device` | `str` | `"cuda"` | PyTorch device (`cuda`, `cpu`, `mps`). Auto-detect if set to `"auto"`. |
| `--cache-dir` | `str` | `"./pretrained"` | Directory for caching downloaded HuggingFace models. |
| `--verbose` / `-v` | `flag` | `False` | Print detailed per-file progress. |
| `--skip-existing` | `flag` | `True` (default on) | Skip files whose output `.pt` already exists. Use `--no-skip-existing` to force re-process. |
| `--file-list` | `str` | `None` | Path to a text file listing specific filenames (stems) to process, one per line. If provided, only these files are processed from `--input-dir`. |
| `--max-files` | `int` | `None` | Process at most N files (useful for testing / dry runs). |
| `--log-file` | `str` | `None` | Path to write a JSON log of results (successes, failures, skips). |

### 5.2 Latent-Specific Options (`latent` subcommand and full pipeline)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--chunked / --no-chunked` | `flag` | `True` | Use chunked VAE encoding for long audio. Disable for very short clips. |
| `--chunk-size` | `int` | `128` | Chunk size in latent frames for chunked encoding. |
| `--overlap` | `int` | `32` | Overlap in latent frames between chunks. |
| `--normalize-audio / --no-normalize-audio` | `flag` | `True` | Peak-normalize audio to `--target-dbfs` before encoding. |
| `--target-dbfs` | `float` | `-6.0` | Target peak level in dBFS for normalization. |

### 5.3 Style-Specific Options (`style` subcommand and full pipeline)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--style-mode` | `str` | `"audio"` | Style extraction mode: `audio`, `text`, or `combined`. |
| `--style-text` | `str` | `None` | Text prompt for `text` or `combined` mode. Applied to **all** files uniformly. |
| `--style-text-file` | `str` | `None` | Path to a TSV/JSON mapping `sample_id → text_prompt` for per-file text prompts. |
| `--style-alpha` | `float` | `0.5` | Blending weight for `combined` mode (1.0 = pure audio, 0.0 = pure text). |
| `--style-normalize / --no-style-normalize` | `flag` | `True` | L2-normalize combined embeddings (recommended). |

### 5.4 Lyrics-Specific Options (`lyrics` subcommand and full pipeline)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--instrumental-handling` | `str` | `"auto"` | What to do when no `.lrc` exists: `auto` (create dummy), `skip` (skip file), `error` (raise). |

### 5.5 Manifest Options (`manifest` subcommand and full pipeline)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--manifest` | `str` | `"train.scp"` | Filename for the manifest (written inside `--output-dir`). |
| `--manifest-format` | `str` | `"pipe"` | Manifest format: `pipe` (default `id\|lrc\|latent\|style`) or `json`. |
| `--relative-paths / --absolute-paths` | `flag` | relative | Whether manifest paths are relative to `--output-dir` or absolute. |

### 5.6 Verify Options (`verify` subcommand)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--verify-latent / --no-verify-latent` | `flag` | `True` | Verify latent `.pt` files. |
| `--verify-style / --no-verify-style` | `flag` | `True` | Verify style `.pt` files. |
| `--verify-lrc / --no-verify-lrc` | `flag` | `True` | Verify LRC `.pt` files. |
| `--verify-manifest` | `str` | `None` | Path to manifest file to cross-check against on-disk files. |

---

## 6. Input Discovery Logic

The script needs to find audio/LRC pairs in `--input-dir`. The approach:

```
1. Scan --input-dir for audio files matching AUDIO_EXTENSIONS
   AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}

2. For each audio file, derive:
   - sample_id = stem of filename (e.g., "song001" from "song001.mp3")
   - lrc_path  = same directory, same stem, ".lrc" extension (or None if missing)

3. If --file-list is provided, filter to only matching sample_ids.

4. If --max-files is provided, truncate the list.

5. If --skip-existing is active (default), check which output files already exist
   and remove those samples from the processing queue.
```

### Edge Cases to Handle

- **Duplicate stems** (e.g., `song.mp3` and `song.wav` in same dir): Warn and skip the duplicate, or take the first one alphabetically.
- **Nested directories**: Only scan top-level by default; add `--recursive` flag to walk subdirectories.
- **No LRC file**: Depends on `--instrumental-handling` (see above).
- **LRC but no audio**: Skip with a warning (LRC alone is useless).

---

## 7. Processing Pipeline

### 7.1 Full Pipeline Execution Order

```
┌──────────────────────────────────────────────────────────────┐
│  1. Parse CLI arguments                                      │
│  2. Discover audio/LRC pairs                                 │
│  3. Filter by --file-list, --max-files, --skip-existing      │
│  4. Load models (only those needed for selected stages)      │
│     ├── VAE model   (if running latent stage)                │
│     └── MuQ model   (if running style stage)                 │
│  5. Process each sample:                                     │
│     ├── (a) Encode audio → latent/.pt                        │
│     ├── (b) Extract style → style/.pt                        │
│     └── (c) Tokenize lyrics → lrc/.pt                        │
│  6. Generate manifest (train.scp)                            │
│  7. (Optional) Run verification                              │
│  8. Print summary / write log                                │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 Model Loading Strategy

Models are large and expensive to load. The script should:

1. **Load each model once** at startup, not per-file.
2. **Only load models needed** for the selected subcommand:
   - `latent` → load VAE only
   - `style` → load MuQ only
   - `lyrics` → no model needed
   - `all` → load both VAE and MuQ
   - `manifest` / `verify` → no models needed
3. **Lazy loading** — don't load a model until the first file that actually needs it (relevant if `--skip-existing` skips all files of one type).

### 7.3 Per-Sample Processing

For each sample, processing is wrapped in a try/except so one failure doesn't abort the entire run:

```python
for sample in tqdm(samples, desc="Processing"):
    results = {}
    
    # Stage 1: Latent
    if run_latent:
        try:
            latent = encode_audio_to_latent(sample.audio_path, vae, ...)
            save_latent(latent, sample.latent_output)
            results['latent'] = 'success'
        except Exception as e:
            results['latent'] = f'error: {e}'
            log_error(sample.id, 'latent', e)
            if not args.continue_on_error:
                raise
    
    # Stage 2: Style (similar pattern)
    # Stage 3: Lyrics (similar pattern)
    
    # Only add to manifest if ALL required stages succeeded
    if all(v == 'success' for v in results.values()):
        manifest_lines.append(format_manifest_line(sample))
```

---

## 8. Output Structure

```
{output-dir}/
├── latent/
│   ├── song001.pt          # [1, 64, T] float32
│   ├── song002.pt
│   └── ...
├── style/
│   ├── song001.pt          # [1, 512] float32
│   ├── song002.pt
│   └── ...
├── lrc/
│   ├── song001.pt          # {'time': [...], 'lrc': [[...]]}
│   ├── song002.pt
│   └── ...
├── train.scp               # Manifest file
└── preprocess.log          # (optional) JSON log of processing results
```

### Manifest Format (default `pipe`)

```
song001|dataset/lrc/song001.pt|dataset/latent/song001.pt|dataset/style/song001.pt
song002|dataset/lrc/song002.pt|dataset/latent/song002.pt|dataset/style/song002.pt
```

Order matches what `dataset/dataset.py` expects: `utt_id|lrc_path|latent_path|style_path`.

---

## 9. Error Handling and Resume

### 9.1 Skip-Existing (Resume Support)

The default `--skip-existing` behavior checks for each sample whether **all** required output files exist:

```python
def should_skip(sample_id, output_dir, stages):
    """Return True if all required outputs already exist."""
    checks = []
    if 'latent' in stages:
        checks.append(os.path.exists(f"{output_dir}/latent/{sample_id}.pt"))
    if 'style' in stages:
        checks.append(os.path.exists(f"{output_dir}/style/{sample_id}.pt"))
    if 'lrc' in stages:
        checks.append(os.path.exists(f"{output_dir}/lrc/{sample_id}.pt"))
    return all(checks)
```

If only some outputs exist (partial failure), the sample is re-processed for all stages. A future enhancement could track per-stage completion.

### 9.2 Error Logging

When `--log-file` is provided, write a JSON log:

```json
{
  "run_config": { "input_dir": "...", "device": "cuda", ... },
  "timestamp": "2026-02-14T12:00:00",
  "results": {
    "song001": {"latent": "success", "style": "success", "lrc": "success"},
    "song002": {"latent": "error: GPU OOM", "style": "skipped", "lrc": "skipped"}
  },
  "summary": {
    "total": 100,
    "success": 98,
    "failed": 2,
    "skipped": 0
  }
}
```

### 9.3 `--continue-on-error` (default: True)

By default, if one file fails, processing continues with the next file. Set `--no-continue-on-error` to abort on the first failure (useful for debugging).

---

## 10. Implementation Approach

### 10.1 File Structure

The unified script lives at `preprocess/preprocess_data.py` (replacing the current placeholder). It imports everything from the existing modules — no code duplication.

```python
# preprocess/preprocess_data.py (outline)

#!/usr/bin/env python3
"""Unified DiffRhythm data preprocessing script."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import ...

from tqdm import tqdm

# Import from our existing modules
from preprocess.audio_processing import (
    load_vae, encode_audio_to_latent, save_latent, verify_latent,
)
from preprocess.style_extraction import (
    load_muq, extract_style_from_audio, extract_style_from_text,
    extract_style_combined, save_style_file, verify_style_file,
)
from preprocess.lyrics_tokenization import (
    tokenize_lyrics, create_instrumental_lrc, save_lrc_tokens, verify_lrc,
)
```

### 10.2 Key Classes / Functions

```python
@dataclass
class SampleInfo:
    """Represents one audio sample to process."""
    sample_id: str
    audio_path: str
    lrc_path: Optional[str]  # None for instrumental

def build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser with subcommands."""
    ...

def discover_samples(input_dir, file_list=None, recursive=False) -> List[SampleInfo]:
    """Find all audio/LRC pairs in input directory."""
    ...

def run_full_pipeline(args, samples):
    """Run all stages for a list of samples."""
    ...

def run_latent_stage(args, samples):
    """Run just the latent encoding stage."""
    ...

def run_style_stage(args, samples):
    """Run just the style extraction stage."""
    ...

def run_lyrics_stage(args, samples):
    """Run just the lyrics tokenization stage."""
    ...

def run_manifest_stage(args):
    """Generate train.scp from existing .pt files."""
    ...

def run_verify_stage(args):
    """Verify all output files."""
    ...

def main():
    parser = build_parser()
    args = parser.parse_args()
    ...
```

### 10.3 `utils.py` Contents

The currently-empty `preprocess/utils.py` should house shared helpers:

```python
# preprocess/utils.py

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}

def auto_detect_device() -> str:
    """Return best available device: cuda > mps > cpu."""
    ...

def discover_audio_files(input_dir, recursive=False) -> List[Path]:
    """Find all audio files in a directory."""
    ...

def get_audio_duration(audio_path) -> float:
    """Get duration in seconds using librosa (needed for instrumental LRC)."""
    ...

def format_manifest_line(sample_id, output_dir, relative=True) -> str:
    """Format a single manifest line."""
    ...

def setup_logging(log_file=None, verbose=False):
    """Configure logging for the preprocessing run."""
    ...
```

### 10.4 Handling the Style Text Map

For `--style-text-file`, support a simple TSV format:

```
# style_prompts.tsv
song001\tSmooth jazz piano with walking bass
song002\tHigh-energy electronic dance track
song003\tAcoustic folk ballad with fingerpicking
```

Or JSON:

```json
{
  "song001": "Smooth jazz piano with walking bass",
  "song002": "High-energy electronic dance track"
}
```

The script detects format by file extension (`.json` vs `.tsv`).

---

## 11. Open Questions

These are decisions to make before or during implementation:

### Q1: Recursive directory scanning?

Should `--input-dir` scan subdirectories by default, or require an explicit `--recursive` flag? Recommendation: **require `--recursive`** to avoid surprises with deeply nested music libraries.

### Q2: GPU memory management between models

The VAE and MuQ both live on GPU during the full pipeline. On GPUs with limited VRAM, should we:
- (a) Load both at once (simpler, uses more VRAM)
- (b) Process all latents first, unload VAE, then load MuQ for styles (sequential, less VRAM)
- (c) Let the user control this via a `--low-vram` flag

Recommendation: **(a) by default, (c) as an option.** Most users with GPUs will have enough VRAM for both. The `--low-vram` flag would run stages sequentially with model unloading between them.

### Q3: Audio duration filtering

Should we add `--min-duration` / `--max-duration` options to skip audio files that are too short (< 10s, which will fail MuQ) or too long (which may OOM)? Recommendation: **Yes, add them** with defaults of `--min-duration 10` and `--max-duration 600`.

### Q4: Dry-run mode

A `--dry-run` flag that discovers files, prints what would be processed, and exits without loading models or writing files. Recommendation: **Yes, add it.** Very useful for checking `--input-dir` layout before committing to a long run.

---

## Example Usage Scenarios

### Scenario 1: First-time full preprocessing

```bash
python preprocess/preprocess_data.py \
    --input-dir /data/my_songs \
    --output-dir dataset \
    --device cuda \
    --verbose
```

### Scenario 2: Resume after crash

```bash
# Same command — --skip-existing is on by default
python preprocess/preprocess_data.py \
    --input-dir /data/my_songs \
    --output-dir dataset \
    --device cuda
```

### Scenario 3: Re-do just style embeddings with text prompts

```bash
python preprocess/preprocess_data.py style \
    --input-dir /data/my_songs \
    --output-dir dataset \
    --style-mode combined \
    --style-text-file prompts.tsv \
    --style-alpha 0.7 \
    --no-skip-existing
```

### Scenario 4: Process on CPU (no GPU)

```bash
python preprocess/preprocess_data.py \
    --input-dir /data/my_songs \
    --output-dir dataset \
    --device cpu \
    --no-chunked
```

### Scenario 5: Quick test with 5 files

```bash
python preprocess/preprocess_data.py \
    --input-dir /data/my_songs \
    --output-dir dataset_test \
    --max-files 5 \
    --verbose
```

### Scenario 6: Verify existing dataset

```bash
python preprocess/preprocess_data.py verify \
    --output-dir dataset \
    --verify-manifest dataset/train.scp \
    --verbose
```

### Scenario 7: Dry run to check what would be processed

```bash
python preprocess/preprocess_data.py \
    --input-dir /data/my_songs \
    --output-dir dataset \
    --dry-run
```

### Scenario 8: Generate manifest from already-processed files

```bash
python preprocess/preprocess_data.py manifest \
    --output-dir dataset \
    --manifest train.scp \
    --relative-paths
```
