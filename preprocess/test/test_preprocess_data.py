"""
Tests for the Unified Preprocessing Script (preprocess/preprocess_data.py)

Tests are split into two groups:
    1. Offline tests   — No models needed, test file discovery, skip logic, manifest
    2. Integration tests — Require VAE + MuQ models and test_audio.mp3

Test data:
    preprocess/test/test_audio.mp3  — Sample audio (~35s)
    preprocess/test/eg_en.lrc       — English lyrics
    preprocess/test/eg_cn.lrc       — Chinese lyrics

Usage (recommended — run from repo root):
    python preprocess/test/test_preprocess_data.py
    python preprocess/test/test_preprocess_data.py cpu      # Force CPU
    python preprocess/test/test_preprocess_data.py cuda     # Force CUDA
    python preprocess/test/test_preprocess_data.py mps      # Force MPS
"""

import os
import shutil
import sys
import tempfile

# ---------------------------------------------------------------------------
# Path setup — ensure both the preprocess dir and repo root are importable.
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
preprocess_dir = os.path.dirname(script_dir)
repo_root = os.path.dirname(preprocess_dir)
sys.path.insert(0, preprocess_dir)
sys.path.insert(0, repo_root)

import torch
from preprocess_data import (
    AUDIO_EXTENSIONS,
    Sample,
    discover_samples,
    should_skip,
    process_sample,
    write_manifest,
)


# ===================================================================
# Helpers
# ===================================================================

def find_test_audio():
    """Locate the test audio file regardless of cwd."""
    candidates = [
        os.path.join(script_dir, "test_audio.mp3"),
        "preprocess/test/test_audio.mp3",
        "test/test_audio.mp3",
        "test_audio.mp3",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def find_test_lrc():
    """Locate the English LRC test file."""
    candidates = [
        os.path.join(script_dir, "eg_en.lrc"),
        "preprocess/test/eg_en.lrc",
        "test/eg_en.lrc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def make_temp_input_dir(files):
    """
    Create a temporary directory populated with empty placeholder files.

    Args:
        files: list of filenames to create (e.g. ["song.mp3", "song.lrc"]).
               Files are empty — this is for testing discovery logic, not
               actual audio processing.

    Returns:
        Path to the temporary directory (caller must clean up).
    """
    tmpdir = tempfile.mkdtemp(prefix="diffrhythm_test_")
    for name in files:
        filepath = os.path.join(tmpdir, name)
        # Create an empty placeholder file
        with open(filepath, "w") as f:
            pass
    return tmpdir


def make_temp_input_dir_with_audio(audio_src, lrc_src=None, stem="test_audio"):
    """
    Create a temp input directory with a real audio file (and optional LRC).

    Copies the actual files so the processing pipeline can read them.

    Returns:
        (tmpdir, sample_id)
    """
    tmpdir = tempfile.mkdtemp(prefix="diffrhythm_test_")
    ext = os.path.splitext(audio_src)[1]
    dst_audio = os.path.join(tmpdir, f"{stem}{ext}")
    shutil.copy2(audio_src, dst_audio)

    if lrc_src:
        dst_lrc = os.path.join(tmpdir, f"{stem}.lrc")
        shutil.copy2(lrc_src, dst_lrc)

    return tmpdir, stem


# ===================================================================
# Offline Tests: discover_samples
# ===================================================================

def test_discover_basic() -> bool:
    """Discover audio files and pair with LRC by stem."""
    print("Testing discover_samples (basic)...")

    tmpdir = make_temp_input_dir(["song001.mp3", "song001.lrc", "song002.wav"])
    try:
        samples = discover_samples(tmpdir)

        assert len(samples) == 2, f"Expected 2 samples, got {len(samples)}"

        by_id = {s.sample_id: s for s in samples}

        # song001 has a matching .lrc
        assert "song001" in by_id
        assert by_id["song001"].lrc_path is not None
        assert by_id["song001"].lrc_path.endswith("song001.lrc")

        # song002 has no .lrc (instrumental)
        assert "song002" in by_id
        assert by_id["song002"].lrc_path is None

        print("✓ discover_samples (basic) passed")
        return True
    finally:
        shutil.rmtree(tmpdir)


def test_discover_duplicate_stems() -> bool:
    """When two audio files share a stem, only the first (alphabetical) is kept."""
    print("Testing discover_samples (duplicate stems)...")

    tmpdir = make_temp_input_dir(["song.flac", "song.mp3", "song.wav"])
    try:
        samples = discover_samples(tmpdir)

        assert len(samples) == 1, f"Expected 1 sample, got {len(samples)}"
        # .flac comes first alphabetically
        assert samples[0].audio_path.endswith("song.flac"), \
            f"Expected .flac to win, got {samples[0].audio_path}"

        print("✓ discover_samples (duplicate stems) passed")
        return True
    finally:
        shutil.rmtree(tmpdir)


def test_discover_empty_dir() -> bool:
    """An empty directory returns an empty list."""
    print("Testing discover_samples (empty directory)...")

    tmpdir = tempfile.mkdtemp(prefix="diffrhythm_test_")
    try:
        samples = discover_samples(tmpdir)
        assert len(samples) == 0, f"Expected 0 samples, got {len(samples)}"
        print("✓ discover_samples (empty directory) passed")
        return True
    finally:
        shutil.rmtree(tmpdir)


def test_discover_nonexistent_dir() -> bool:
    """A nonexistent directory raises FileNotFoundError."""
    print("Testing discover_samples (nonexistent directory)...")

    try:
        discover_samples("/nonexistent/path/that/does/not/exist")
        print("  ✗ Should have raised FileNotFoundError")
        return False
    except FileNotFoundError:
        print("  ✓ Correctly raised FileNotFoundError")

    print("✓ discover_samples (nonexistent directory) passed")
    return True


def test_discover_ignores_non_audio() -> bool:
    """Non-audio files (.txt, .lrc, .py) are not discovered."""
    print("Testing discover_samples (ignores non-audio)...")

    tmpdir = make_temp_input_dir([
        "readme.txt", "notes.lrc", "script.py", "image.png",
        "actual_song.mp3",
    ])
    try:
        samples = discover_samples(tmpdir)
        assert len(samples) == 1, f"Expected 1 sample, got {len(samples)}"
        assert samples[0].sample_id == "actual_song"
        print("✓ discover_samples (ignores non-audio) passed")
        return True
    finally:
        shutil.rmtree(tmpdir)


def test_discover_all_extensions() -> bool:
    """All supported audio extensions are discovered."""
    print("Testing discover_samples (all extensions)...")

    files = [f"track{i}{ext}" for i, ext in enumerate(sorted(AUDIO_EXTENSIONS))]
    tmpdir = make_temp_input_dir(files)
    try:
        samples = discover_samples(tmpdir)
        assert len(samples) == len(AUDIO_EXTENSIONS), \
            f"Expected {len(AUDIO_EXTENSIONS)} samples, got {len(samples)}"
        print(f"  Found all {len(AUDIO_EXTENSIONS)} extensions: {sorted(AUDIO_EXTENSIONS)}")
        print("✓ discover_samples (all extensions) passed")
        return True
    finally:
        shutil.rmtree(tmpdir)


# ===================================================================
# Offline Tests: should_skip
# ===================================================================

def test_should_skip_all_exist() -> bool:
    """should_skip returns True when all three .pt files exist."""
    print("Testing should_skip (all exist)...")

    tmpdir = tempfile.mkdtemp(prefix="diffrhythm_test_")
    try:
        for subdir in ("latent", "style", "lrc"):
            dirpath = os.path.join(tmpdir, subdir)
            os.makedirs(dirpath)
            torch.save(torch.zeros(1), os.path.join(dirpath, "song001.pt"))

        assert should_skip("song001", tmpdir) is True
        print("✓ should_skip (all exist) passed")
        return True
    finally:
        shutil.rmtree(tmpdir)


def test_should_skip_partial() -> bool:
    """should_skip returns False if any .pt file is missing."""
    print("Testing should_skip (partial)...")

    tmpdir = tempfile.mkdtemp(prefix="diffrhythm_test_")
    tmpdir2 = tempfile.mkdtemp(prefix="diffrhythm_test_")
    try:
        # Case 1: latent + style exist, but lrc missing
        for subdir in ("latent", "style"):
            dirpath = os.path.join(tmpdir, subdir)
            os.makedirs(dirpath)
            torch.save(torch.zeros(1), os.path.join(dirpath, "song001.pt"))
        os.makedirs(os.path.join(tmpdir, "lrc"))  # dir exists but no file

        assert should_skip("song001", tmpdir) is False
        print("  ✓ Missing lrc → should not skip")

        # Case 2: style + lrc exist, but latent missing
        for subdir in ("style", "lrc"):
            dirpath = os.path.join(tmpdir2, subdir)
            os.makedirs(dirpath)
            torch.save(torch.zeros(1), os.path.join(dirpath, "song001.pt"))
        os.makedirs(os.path.join(tmpdir2, "latent"))

        assert should_skip("song001", tmpdir2) is False
        print("  ✓ Missing latent → should not skip")

        print("✓ should_skip (partial) passed")
        return True
    finally:
        shutil.rmtree(tmpdir)
        shutil.rmtree(tmpdir2)


def test_should_skip_none_exist() -> bool:
    """should_skip returns False when no output files exist."""
    print("Testing should_skip (none exist)...")

    tmpdir = tempfile.mkdtemp(prefix="diffrhythm_test_")
    try:
        assert should_skip("song001", tmpdir) is False
        print("✓ should_skip (none exist) passed")
        return True
    finally:
        shutil.rmtree(tmpdir)


# ===================================================================
# Offline Tests: write_manifest
# ===================================================================

def test_write_manifest_format() -> bool:
    """Verify manifest format matches dataset/dataset.py expectations."""
    print("Testing write_manifest (format)...")

    tmpdir = tempfile.mkdtemp(prefix="diffrhythm_test_")
    try:
        path = write_manifest(["song002", "song001"], tmpdir)

        assert os.path.exists(path), f"Manifest not written to {path}"
        assert path.endswith("train.scp")

        with open(path, "r") as f:
            lines = f.read().strip().split("\n")

        # Should be sorted
        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

        # Parse and verify format: {id}|{lrc}|{latent}|{style}
        for line in lines:
            parts = line.split("|")
            assert len(parts) == 4, f"Expected 4 pipe-delimited fields, got {len(parts)}: {line}"
            sid, lrc_path, latent_path, style_path = parts
            assert lrc_path.endswith(f"{sid}.pt")
            assert latent_path.endswith(f"{sid}.pt")
            assert style_path.endswith(f"{sid}.pt")
            assert "lrc" in lrc_path
            assert "latent" in latent_path
            assert "style" in style_path

        # Verify sorted order
        assert lines[0].startswith("song001|"), "Manifest should be sorted"
        assert lines[1].startswith("song002|"), "Manifest should be sorted"

        print("  ✓ Format: {id}|{lrc_path}|{latent_path}|{style_path}")
        print("  ✓ Entries are sorted by sample_id")
        print("✓ write_manifest (format) passed")
        return True
    finally:
        shutil.rmtree(tmpdir)


def test_write_manifest_empty() -> bool:
    """An empty sample list writes an empty manifest file."""
    print("Testing write_manifest (empty)...")

    tmpdir = tempfile.mkdtemp(prefix="diffrhythm_test_")
    try:
        path = write_manifest([], tmpdir)
        assert os.path.exists(path)

        with open(path, "r") as f:
            content = f.read().strip()
        assert content == "", "Empty manifest should be empty"

        print("✓ write_manifest (empty) passed")
        return True
    finally:
        shutil.rmtree(tmpdir)


# ===================================================================
# Integration Tests: process_sample (require models + test audio)
# ===================================================================

def test_process_sample_with_lyrics(device="cpu", cache_dir="./pretrained") -> bool:
    """
    Full pipeline test: process one sample with audio + lyrics.

    Verifies that all three .pt files are created with correct shapes.
    """
    print("Testing process_sample (with lyrics)...")

    test_audio = find_test_audio()
    test_lrc = find_test_lrc()

    if test_audio is None:
        print("  ⚠ Skipping (test_audio.mp3 not found)")
        return True

    if test_lrc is None:
        print("  ⚠ Skipping (eg_en.lrc not found)")
        return True

    print(f"  Audio: {test_audio}")
    print(f"  LRC:   {test_lrc}")

    # Set up temp directories
    input_dir, stem = make_temp_input_dir_with_audio(test_audio, test_lrc, stem="test_song")
    output_dir = tempfile.mkdtemp(prefix="diffrhythm_test_out_")

    try:
        for subdir in ("latent", "style", "lrc"):
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

        # Load models
        from audio_processing import load_vae
        from style_extraction import load_muq

        print("  Loading VAE...")
        vae = load_vae(device=device, cache_dir=cache_dir)
        print("  Loading MuQ...")
        muq = load_muq(device=device, cache_dir=cache_dir)

        # Build sample
        sample = Sample(
            sample_id=stem,
            audio_path=os.path.join(input_dir, f"{stem}.mp3"),
            lrc_path=os.path.join(input_dir, f"{stem}.lrc"),
        )

        # Process
        print("  Processing sample...")
        process_sample(sample, output_dir, vae, muq, device=device, verbose=True)

        # Verify latent
        latent_path = os.path.join(output_dir, "latent", f"{stem}.pt")
        assert os.path.exists(latent_path), "Latent file not created"
        latent = torch.load(latent_path, map_location="cpu")
        assert latent.dim() == 3, f"Latent should be 3D, got {latent.dim()}D"
        assert latent.shape[1] == 64, f"Latent dim should be 64, got {latent.shape[1]}"
        print(f"  ✓ Latent: {list(latent.shape)}")

        # Verify style
        style_path = os.path.join(output_dir, "style", f"{stem}.pt")
        assert os.path.exists(style_path), "Style file not created"
        style = torch.load(style_path, map_location="cpu")
        assert style.shape == (1, 512) or style.shape == (512,), \
            f"Style shape should be [1, 512] or [512], got {list(style.shape)}"
        print(f"  ✓ Style: {list(style.shape)}")

        # Verify lrc
        lrc_path = os.path.join(output_dir, "lrc", f"{stem}.pt")
        assert os.path.exists(lrc_path), "LRC file not created"
        lrc_data = torch.load(lrc_path, map_location="cpu")
        assert isinstance(lrc_data, dict), "LRC should be a dict"
        assert "time" in lrc_data and "lrc" in lrc_data, "LRC missing keys"
        assert len(lrc_data["time"]) > 0, "LRC should have entries"
        assert len(lrc_data["time"]) == len(lrc_data["lrc"]), "LRC time/lrc mismatch"
        # Verify all tokens are shifted (>= 1)
        for tl in lrc_data["lrc"]:
            for tok in tl:
                assert tok >= 1, f"Token {tok} should be >= 1 (0 is padding)"
        print(f"  ✓ LRC: {len(lrc_data['time'])} lines, tokens shifted")

        print("✓ process_sample (with lyrics) passed")
        return True

    except Exception as e:
        print(f"✗ process_sample (with lyrics) failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(input_dir)
        shutil.rmtree(output_dir)


def test_process_sample_instrumental(device="cpu", cache_dir="./pretrained") -> bool:
    """
    Full pipeline test: process one sample without lyrics (instrumental).

    Verifies that a dummy LRC entry is created.
    """
    print("Testing process_sample (instrumental — no LRC)...")

    test_audio = find_test_audio()
    if test_audio is None:
        print("  ⚠ Skipping (test_audio.mp3 not found)")
        return True

    # Set up temp dirs — no LRC file
    input_dir, stem = make_temp_input_dir_with_audio(test_audio, lrc_src=None, stem="instrumental")
    output_dir = tempfile.mkdtemp(prefix="diffrhythm_test_out_")

    try:
        for subdir in ("latent", "style", "lrc"):
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

        from audio_processing import load_vae
        from style_extraction import load_muq

        print("  Loading models...")
        vae = load_vae(device=device, cache_dir=cache_dir)
        muq = load_muq(device=device, cache_dir=cache_dir)

        sample = Sample(
            sample_id=stem,
            audio_path=os.path.join(input_dir, f"{stem}.mp3"),
            lrc_path=None,  # instrumental
        )

        print("  Processing sample...")
        process_sample(sample, output_dir, vae, muq, device=device, verbose=True)

        # Verify lrc is a dummy instrumental entry
        lrc_path = os.path.join(output_dir, "lrc", f"{stem}.pt")
        assert os.path.exists(lrc_path), "LRC file not created"
        lrc_data = torch.load(lrc_path, map_location="cpu")
        assert len(lrc_data["time"]) == 1, "Instrumental should have 1 dummy entry"
        assert len(lrc_data["lrc"]) == 1, "Instrumental should have 1 dummy entry"
        assert lrc_data["lrc"][0] == [2], f"Expected [2] (PERIOD_TOKEN_ID), got {lrc_data['lrc'][0]}"
        print(f"  ✓ Instrumental LRC: time={lrc_data['time']}, tokens={lrc_data['lrc']}")

        # Verify latent and style were also created
        assert os.path.exists(os.path.join(output_dir, "latent", f"{stem}.pt"))
        assert os.path.exists(os.path.join(output_dir, "style", f"{stem}.pt"))
        print("  ✓ Latent and style files also created")

        print("✓ process_sample (instrumental) passed")
        return True

    except Exception as e:
        print(f"✗ process_sample (instrumental) failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(input_dir)
        shutil.rmtree(output_dir)


def test_end_to_end_pipeline(device="cpu", cache_dir="./pretrained") -> bool:
    """
    End-to-end test: discover → process → manifest → verify skip.

    Mimics what main() does but without argparse, testing the full flow.
    """
    print("Testing end-to-end pipeline...")

    test_audio = find_test_audio()
    test_lrc = find_test_lrc()

    if test_audio is None:
        print("  ⚠ Skipping (test_audio.mp3 not found)")
        return True

    # Create input dir with one audio+lrc pair
    input_dir = tempfile.mkdtemp(prefix="diffrhythm_test_")
    output_dir = tempfile.mkdtemp(prefix="diffrhythm_test_out_")

    try:
        # Copy test audio
        shutil.copy2(test_audio, os.path.join(input_dir, "song_a.mp3"))
        if test_lrc:
            shutil.copy2(test_lrc, os.path.join(input_dir, "song_a.lrc"))

        # Step 1: Discover
        samples = discover_samples(input_dir)
        assert len(samples) == 1, f"Expected 1 sample, got {len(samples)}"
        assert samples[0].sample_id == "song_a"
        if test_lrc:
            assert samples[0].lrc_path is not None
        print("  ✓ Step 1: Discovered 1 sample")

        # Step 2: Should not skip (nothing processed yet)
        assert not should_skip("song_a", output_dir)
        print("  ✓ Step 2: Not skipped (no outputs yet)")

        # Step 3: Create output dirs
        for subdir in ("latent", "style", "lrc"):
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

        # Step 4: Load models and process
        from audio_processing import load_vae
        from style_extraction import load_muq

        print("  Loading models...")
        vae = load_vae(device=device, cache_dir=cache_dir)
        muq = load_muq(device=device, cache_dir=cache_dir)

        print("  Processing sample...")
        process_sample(samples[0], output_dir, vae, muq, device=device)
        print("  ✓ Step 4: Sample processed")

        # Step 5: Write manifest
        manifest_path = write_manifest(["song_a"], output_dir)
        assert os.path.exists(manifest_path)
        with open(manifest_path, "r") as f:
            manifest_content = f.read().strip()
        assert manifest_content.startswith("song_a|")
        parts = manifest_content.split("|")
        assert len(parts) == 4
        print(f"  ✓ Step 5: Manifest written ({manifest_path})")
        print(f"    {manifest_content}")

        # Step 6: should_skip should now return True
        assert should_skip("song_a", output_dir) is True
        print("  ✓ Step 6: Skip check works (all outputs exist)")

        # Step 7: Re-discover — sample should be filterable
        samples_again = discover_samples(input_dir)
        remaining = [s for s in samples_again if not should_skip(s.sample_id, output_dir)]
        assert len(remaining) == 0, "Should have 0 remaining after processing"
        print("  ✓ Step 7: Re-run correctly filters processed samples")

        print("✓ end-to-end pipeline passed")
        return True

    except Exception as e:
        print(f"✗ end-to-end pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(input_dir)
        shutil.rmtree(output_dir)


# ===================================================================
# Runner
# ===================================================================

def run_all_tests(device: str = "cpu"):
    """Run all tests: offline first, then integration."""
    print("=" * 60)
    print("Unified Preprocessing Script — Test Suite")
    print("=" * 60)
    print(f"Device: {device}")
    print()

    # --- Offline tests (no models needed) ---
    offline_tests = [
        ("Discover: basic", test_discover_basic),
        ("Discover: duplicate stems", test_discover_duplicate_stems),
        ("Discover: empty dir", test_discover_empty_dir),
        ("Discover: nonexistent dir", test_discover_nonexistent_dir),
        ("Discover: ignores non-audio", test_discover_ignores_non_audio),
        ("Discover: all extensions", test_discover_all_extensions),
        ("Skip: all exist", test_should_skip_all_exist),
        ("Skip: partial", test_should_skip_partial),
        ("Skip: none exist", test_should_skip_none_exist),
        ("Manifest: format", test_write_manifest_format),
        ("Manifest: empty", test_write_manifest_empty),
    ]

    # --- Integration tests (models + test audio) ---
    integration_tests = [
        ("Process: with lyrics",
         lambda: test_process_sample_with_lyrics(device=device)),
        ("Process: instrumental",
         lambda: test_process_sample_instrumental(device=device)),
        ("End-to-end pipeline",
         lambda: test_end_to_end_pipeline(device=device)),
    ]

    results = []

    print("--- Offline Tests ---")
    for name, test_func in offline_tests:
        print()
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print()
    print("--- Integration Tests (require models + test_audio.mp3) ---")
    for name, test_func in integration_tests:
        print()
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # --- Summary ---
    print()
    print("=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")

    passed = sum(1 for _, s in results if s)
    total = len(results)

    print()
    print(f"Tests passed: {passed}/{total}")
    if passed == total:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. Check the output above.")
    print("=" * 60)


if __name__ == "__main__":
    # Auto-detect best device or use command line argument
    if len(sys.argv) > 1:
        device = sys.argv[1]
        print(f"Using device: {device}")
    else:
        if torch.backends.mps.is_available():
            device = "mps"
            print("Auto-detected: Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            device = "cuda"
            print("Auto-detected: NVIDIA GPU (CUDA)")
        else:
            device = "cpu"
            print("Auto-detected: CPU")
        print()

    run_all_tests(device=device)
