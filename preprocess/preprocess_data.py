#!/usr/bin/env python3
"""
Unified DiffRhythm Data Preprocessing Script

Takes a directory of audio files (with matching .lrc lyrics files) and produces
the three .pt files per sample that DiffRhythm training expects, plus a train.scp
manifest.

Usage:
    python preprocess/preprocess_data.py \
        --input-dir /path/to/songs \
        --output-dir dataset \
        --device cuda

Input directory layout (audio and LRC matched by filename stem):
    songs/
    ├── song001.mp3
    ├── song001.lrc
    ├── song002.wav
    ├── song002.lrc       <- matched to song002.wav by stem
    ├── instrumental.flac  <- no .lrc = treated as instrumental
    └── ...

Output:
    dataset/
    ├── latent/song001.pt   # [1, 64, T] VAE-encoded audio
    ├── style/song001.pt    # [1, 512]   MuQ style embedding
    ├── lrc/song001.pt      # {'time': [...], 'lrc': [[...]]} tokenized lyrics
    └── train.scp           # Manifest for training
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so all imports work regardless of cwd.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import librosa
from tqdm import tqdm

from preprocess.audio_processing import (
    load_vae,
    encode_audio_to_latent,
    save_latent,
)
from preprocess.style_extraction import (
    load_muq,
    extract_style_from_audio,
    save_style_file,
)
from preprocess.lyrics_tokenization import (
    tokenize_lyrics,
    create_instrumental_lrc,
    save_lrc_tokens,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """One audio sample to preprocess."""

    sample_id: str
    audio_path: str
    lrc_path: Optional[str]  # None for instrumental tracks


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_samples(input_dir: str) -> List[Sample]:
    """
    Scan *input_dir* for audio files and pair them with .lrc files by stem.

    Only the top-level directory is scanned (no recursion).  If two audio files
    share the same stem (e.g. song.mp3 and song.wav), only the first one
    encountered (sorted alphabetically) is kept and a warning is printed.
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: '{input_dir}'")

    # Collect audio files, sorted for deterministic ordering
    audio_files = sorted(
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )

    # De-duplicate by stem (first alphabetical file wins)
    seen_stems: Dict[str, Path] = {}
    for f in audio_files:
        stem = f.stem
        if stem in seen_stems:
            print(
                f"Warning: duplicate stem '{stem}' — "
                f"using {seen_stems[stem].name}, skipping {f.name}"
            )
            continue
        seen_stems[stem] = f

    # Build Sample list, pairing with .lrc if present
    samples: List[Sample] = []
    for stem, audio_file in seen_stems.items():
        lrc_file = audio_file.with_suffix(".lrc")
        lrc_path = str(lrc_file) if lrc_file.exists() else None
        samples.append(Sample(
            sample_id=stem,
            audio_path=str(audio_file),
            lrc_path=lrc_path,
        ))

    return samples


# ---------------------------------------------------------------------------
# Skip-existing check
# ---------------------------------------------------------------------------

def should_skip(sample_id: str, output_dir: str) -> bool:
    """Return True if all three output .pt files already exist for this sample."""
    return (
        os.path.exists(os.path.join(output_dir, "latent", f"{sample_id}.pt"))
        and os.path.exists(os.path.join(output_dir, "style", f"{sample_id}.pt"))
        and os.path.exists(os.path.join(output_dir, "lrc", f"{sample_id}.pt"))
    )


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def process_sample(
    sample: Sample,
    output_dir: str,
    vae_model,
    muq_model,
    device: str,
    verbose: bool = False,
) -> None:
    """
    Run all three preprocessing stages for one sample.

    1. Encode audio → VAE latent   → latent/{id}.pt
    2. Extract style → MuQ embed   → style/{id}.pt
    3. Tokenize lyrics (or dummy)  → lrc/{id}.pt

    Raises on failure so the caller can catch and continue.
    """
    sid = sample.sample_id
    latent_path = os.path.join(output_dir, "latent", f"{sid}.pt")
    style_path = os.path.join(output_dir, "style", f"{sid}.pt")
    lrc_out_path = os.path.join(output_dir, "lrc", f"{sid}.pt")

    # --- Stage 1: VAE latent ---
    if verbose:
        print(f"  [{sid}] Encoding audio latent...")
    latent = encode_audio_to_latent(sample.audio_path, vae_model, device=device)
    save_latent(latent, latent_path)

    # --- Stage 2: Style embedding ---
    if verbose:
        print(f"  [{sid}] Extracting style embedding...")
    style = extract_style_from_audio(sample.audio_path, muq_model, device=device)
    save_style_file(style, style_path)

    # --- Stage 3: Lyrics tokenization ---
    if verbose:
        print(f"  [{sid}] Tokenizing lyrics...")
    if sample.lrc_path:
        lrc_data = tokenize_lyrics(sample.lrc_path)
    else:
        duration = librosa.get_duration(path=sample.audio_path)
        lrc_data = create_instrumental_lrc(duration)
        if verbose:
            print(f"  [{sid}] No .lrc file — created instrumental dummy ({duration:.1f}s)")
    save_lrc_tokens(lrc_data, lrc_out_path)


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------

def write_manifest(sample_ids: List[str], output_dir: str) -> str:
    """
    Write train.scp listing all successfully processed samples.

    Format (pipe-delimited, matches dataset/dataset.py):
        {id}|{lrc_path}|{latent_path}|{style_path}

    Returns the path to the manifest file.
    """
    manifest_path = os.path.join(output_dir, "train.scp")

    with open(manifest_path, "w", encoding="utf-8") as f:
        for sid in sorted(sample_ids):
            lrc_path = os.path.join(output_dir, "lrc", f"{sid}.pt")
            latent_path = os.path.join(output_dir, "latent", f"{sid}.pt")
            style_path = os.path.join(output_dir, "style", f"{sid}.pt")
            f.write(f"{sid}|{lrc_path}|{latent_path}|{style_path}\n")

    return manifest_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess audio data for DiffRhythm training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python preprocess/preprocess_data.py \\\n"
            "      --input-dir /data/my_songs \\\n"
            "      --output-dir dataset \\\n"
            "      --device cuda\n"
        ),
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing audio (.mp3/.wav/.flac) and .lrc files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="dataset",
        help="Root output directory (default: dataset)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="PyTorch device: cuda, cpu, or mps (default: cuda)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="./pretrained",
        help="Directory to cache downloaded HuggingFace models (default: ./pretrained)",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip samples whose output files already exist (default: on)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed per-file progress",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Discover samples
    # ------------------------------------------------------------------
    print(f"Scanning {args.input_dir} for audio files...")
    try:
        all_samples = discover_samples(args.input_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not all_samples:
        print("No audio files found. Nothing to do.")
        return

    n_with_lrc = sum(1 for s in all_samples if s.lrc_path is not None)
    n_instrumental = len(all_samples) - n_with_lrc
    print(
        f"Found {len(all_samples)} audio file(s) "
        f"({n_with_lrc} with lyrics, {n_instrumental} instrumental)"
    )

    # ------------------------------------------------------------------
    # 2. Filter out already-processed samples
    # ------------------------------------------------------------------
    if args.skip_existing:
        samples = [s for s in all_samples if not should_skip(s.sample_id, args.output_dir)]
        n_skipped = len(all_samples) - len(samples)
        if n_skipped:
            print(f"Skipping {n_skipped} already-processed sample(s)")
    else:
        samples = all_samples
        n_skipped = 0

    if not samples:
        print("All samples already processed. Nothing to do.")
        return

    print(f"Will process {len(samples)} sample(s)")

    # ------------------------------------------------------------------
    # 3. Create output directories
    # ------------------------------------------------------------------
    for subdir in ("latent", "style", "lrc"):
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # ------------------------------------------------------------------
    # 4. Load models
    # ------------------------------------------------------------------
    print(f"Loading VAE model (device={args.device})...")
    t0 = time.time()
    vae = load_vae(device=args.device, cache_dir=args.cache_dir)
    print(f"  VAE loaded in {time.time() - t0:.1f}s")

    print(f"Loading MuQ model (device={args.device})...")
    t0 = time.time()
    muq = load_muq(device=args.device, cache_dir=args.cache_dir)
    print(f"  MuQ loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 5. Process each sample
    # ------------------------------------------------------------------
    succeeded: List[str] = []
    failed: List[Tuple[str, str]] = []  # (sample_id, error_message)

    for sample in tqdm(samples, desc="Processing", disable=args.verbose):
        try:
            if args.verbose:
                print(f"\nProcessing: {sample.sample_id} ({sample.audio_path})")

            process_sample(
                sample,
                args.output_dir,
                vae_model=vae,
                muq_model=muq,
                device=args.device,
                verbose=args.verbose,
            )
            succeeded.append(sample.sample_id)

        except Exception as e:
            error_msg = str(e)
            failed.append((sample.sample_id, error_msg))
            print(f"\nFAILED {sample.sample_id}: {error_msg}")

    # ------------------------------------------------------------------
    # 6. Write manifest (only successful samples)
    # ------------------------------------------------------------------
    # Include previously-existing samples in the manifest too
    if args.skip_existing:
        existing_ids = [
            s.sample_id for s in all_samples
            if s.sample_id not in succeeded
            and should_skip(s.sample_id, args.output_dir)
        ]
    else:
        existing_ids = []

    all_success_ids = succeeded + existing_ids
    if all_success_ids:
        manifest_path = write_manifest(all_success_ids, args.output_dir)
        print(f"\nManifest written to: {manifest_path} ({len(all_success_ids)} entries)")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Preprocessing complete")
    print(f"  Succeeded:  {len(succeeded)}")
    print(f"  Skipped:    {n_skipped}")
    print(f"  Failed:     {len(failed)}")
    if all_success_ids:
        print(f"  Manifest:   {len(all_success_ids)} total entries")
    print("=" * 50)

    if failed:
        print("\nFailed samples:")
        for sid, err in failed:
            print(f"  {sid}: {err}")


if __name__ == "__main__":
    main()
