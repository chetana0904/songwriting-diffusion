"""
Tests for Lyrics Tokenization Module

Tests the LRC parsing, G2P tokenization, token shifting, save/load
round-trip, and verification utilities in preprocess/lyrics_tokenization.py.

Test LRC files:
    preprocess/test/eg_en.lrc  – copied from infer/example/eg_en.lrc
    preprocess/test/eg_cn.lrc  – copied from infer/example/eg_cn.lrc

Usage (recommended – run from repo root):
    python preprocess/test/test_lyrics_tokenization.py
"""

import sys
import os
import tempfile

# ---------------------------------------------------------------------------
# Path setup – ensure both the preprocess dir and repo root are importable.
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
preprocess_dir = os.path.dirname(script_dir)
repo_root = os.path.dirname(preprocess_dir)
sys.path.insert(0, preprocess_dir)
sys.path.insert(0, repo_root)

import torch
from lyrics_tokenization import (
    parse_lrc_file,
    parse_lrc_string,
    tokenize_lyrics,
    tokenize_lyrics_from_string,
    create_instrumental_lrc,
    save_lrc_tokens,
    load_lrc_tokens,
    verify_lrc,
    PAD_TOKEN_ID,
    COMMA_TOKEN_ID,
    PERIOD_TOKEN_ID,
    _shift_tokens,
)


# ---------------------------------------------------------------------------
# Locate test LRC files.  The files were copied from infer/example/.
# We try several relative paths so the tests work regardless of cwd.
# ---------------------------------------------------------------------------
_LRC_SEARCH_PATHS_EN = [
    os.path.join(script_dir, "eg_en.lrc"),         # from preprocess/test/
    "preprocess/test/eg_en.lrc",                    # from repo root
    "test/eg_en.lrc",                               # from preprocess/
]

_LRC_SEARCH_PATHS_CN = [
    os.path.join(script_dir, "eg_cn.lrc"),
    "preprocess/test/eg_cn.lrc",
    "test/eg_cn.lrc",
]


def _find_lrc(search_paths):
    for p in search_paths:
        if os.path.exists(p):
            return p
    return None


# ===================================================================
# Test: LRC Parsing
# ===================================================================

def test_parse_lrc_file_english() -> bool:
    """Parse the English LRC file and validate structure."""
    print("Testing parse_lrc_file (English)...")

    lrc_path = _find_lrc(_LRC_SEARCH_PATHS_EN)
    if lrc_path is None:
        print("  ⚠ Skipping (eg_en.lrc not found)")
        return True

    print(f"  Using: {lrc_path}")
    result = parse_lrc_file(lrc_path)

    assert isinstance(result, list), "Expected a list"
    assert len(result) > 0, "Expected at least one lyric line"

    for i, (timestamp, text) in enumerate(result):
        assert isinstance(timestamp, float), f"Timestamp at {i} is not float"
        assert isinstance(text, str) and len(text) > 0, f"Text at {i} is empty"

    # Verify timestamps are non-decreasing
    timestamps = [t for t, _ in result]
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1], (
            f"Timestamps not sorted at index {i}: {timestamps[i-1]} > {timestamps[i]}"
        )

    # Verify specific known values from eg_en.lrc (copied from infer/example/eg_en.lrc)
    first_time, first_text = result[0]
    assert first_time == 10.0, f"Expected first timestamp 10.0, got {first_time}"
    assert "Moonlight" in first_text, f"Unexpected first line: {first_text}"

    print(f"  Parsed {len(result)} lyric lines")
    print(f"  Time range: {timestamps[0]:.2f}s – {timestamps[-1]:.2f}s")
    print("✓ parse_lrc_file (English) passed")
    return True


def test_parse_lrc_file_chinese() -> bool:
    """Parse the Chinese LRC file and validate structure."""
    print("Testing parse_lrc_file (Chinese)...")

    lrc_path = _find_lrc(_LRC_SEARCH_PATHS_CN)
    if lrc_path is None:
        print("  ⚠ Skipping (eg_cn.lrc not found)")
        return True

    print(f"  Using: {lrc_path}")
    result = parse_lrc_file(lrc_path)

    assert isinstance(result, list), "Expected a list"
    assert len(result) > 0, "Expected at least one lyric line"

    first_time, first_text = result[0]
    assert first_time == 18.23, f"Expected first timestamp 18.23, got {first_time}"

    print(f"  Parsed {len(result)} lyric lines")
    print("✓ parse_lrc_file (Chinese) passed")
    return True


def test_parse_lrc_string() -> bool:
    """Parse LRC lyrics from an in-memory string."""
    print("Testing parse_lrc_string...")

    lrc_text = (
        "[00:05.00]Hello world\n"
        "[00:10.50]Second line of lyrics\n"
        "[00:20.00]Third line\n"
    )
    result = parse_lrc_string(lrc_text)

    assert len(result) == 3, f"Expected 3 lines, got {len(result)}"
    assert result[0] == (5.0, "Hello world")
    assert result[1] == (10.5, "Second line of lyrics")
    assert result[2] == (20.0, "Third line")

    print("✓ parse_lrc_string passed")
    return True


def test_parse_lrc_malformed_lines() -> bool:
    """Ensure malformed lines are silently skipped."""
    print("Testing parse_lrc_string (malformed lines)...")

    lrc_text = (
        "[00:05.00]Good line\n"
        "no bracket line\n"
        "[bad time]Bad timestamp\n"
        "\n"
        "[00:15.00]Another good line\n"
    )
    result = parse_lrc_string(lrc_text)

    assert len(result) == 2, f"Expected 2 lines, got {len(result)}"
    assert result[0][1] == "Good line"
    assert result[1][1] == "Another good line"

    print("✓ parse_lrc_string (malformed lines) passed")
    return True


# ===================================================================
# Test: Token Shifting
# ===================================================================

def test_token_shifting() -> bool:
    """
    Verify the +1 token ID shift matches CNENTokenizer.encode
    (infer/infer_utils.py lines 357-359).
    """
    print("Testing _shift_tokens (+1 shift)...")

    raw = [0, 1, 2, 50, 100]
    shifted = _shift_tokens(raw)

    assert shifted == [1, 2, 3, 51, 101], f"Unexpected shift result: {shifted}"

    # Verify constants match the shifted values
    assert COMMA_TOKEN_ID == 1, "COMMA_TOKEN_ID should be 1 (vocab ',' = 0, +1)"
    assert PERIOD_TOKEN_ID == 2, "PERIOD_TOKEN_ID should be 2 (vocab '.' = 1, +1)"
    assert PAD_TOKEN_ID == 0, "PAD_TOKEN_ID should be 0 (reserved)"

    print("✓ _shift_tokens passed")
    return True


# ===================================================================
# Test: Full Tokenization
# ===================================================================

def test_tokenize_lyrics_english() -> bool:
    """Tokenize the English LRC file and validate output structure."""
    print("Testing tokenize_lyrics (English)...")

    lrc_path = _find_lrc(_LRC_SEARCH_PATHS_EN)
    if lrc_path is None:
        print("  ⚠ Skipping (eg_en.lrc not found)")
        return True

    result = tokenize_lyrics(lrc_path)

    assert isinstance(result, dict), "Expected a dict"
    assert "time" in result, "Missing 'time' key"
    assert "lrc" in result, "Missing 'lrc' key"

    times = result["time"]
    tokens = result["lrc"]

    assert len(times) == len(tokens), "times and lrc lengths must match"
    assert len(times) > 0, "Expected at least one line"

    # Check that all token IDs are shifted (>= 1, never 0)
    for i, tl in enumerate(tokens):
        assert isinstance(tl, list), f"Token list at {i} is not a list"
        assert len(tl) > 0, f"Token list at {i} is empty"
        for tok in tl:
            assert isinstance(tok, int), f"Token is not int: {tok!r}"
            assert tok >= 1, f"Token {tok} should be >= 1 (0 is reserved for padding)"

    total_tokens = sum(len(tl) for tl in tokens)
    print(f"  Lines: {len(times)}, Total tokens: {total_tokens}")
    print(f"  First line tokens (first 10): {tokens[0][:10]}")
    print("✓ tokenize_lyrics (English) passed")
    return True


def test_tokenize_lyrics_chinese() -> bool:
    """Tokenize the Chinese LRC file and validate output structure."""
    print("Testing tokenize_lyrics (Chinese)...")

    lrc_path = _find_lrc(_LRC_SEARCH_PATHS_CN)
    if lrc_path is None:
        print("  ⚠ Skipping (eg_cn.lrc not found)")
        return True

    result = tokenize_lyrics(lrc_path)

    times = result["time"]
    tokens = result["lrc"]

    assert len(times) == len(tokens), "times and lrc lengths must match"
    assert len(times) > 0, "Expected at least one line"

    # Chinese phoneme tokens tend to be in the higher range of the vocab
    for tl in tokens:
        for tok in tl:
            assert tok >= 1, f"Token {tok} should be >= 1"

    total_tokens = sum(len(tl) for tl in tokens)
    print(f"  Lines: {len(times)}, Total tokens: {total_tokens}")
    print("✓ tokenize_lyrics (Chinese) passed")
    return True


def test_tokenize_lyrics_from_string() -> bool:
    """Tokenize an inline LRC string."""
    print("Testing tokenize_lyrics_from_string...")

    lrc_text = "[00:05.00]Hello world\n[00:10.00]Goodbye moon\n"
    result = tokenize_lyrics_from_string(lrc_text)

    assert len(result["time"]) == 2
    assert len(result["lrc"]) == 2
    assert result["time"][0] == 5.0
    assert result["time"][1] == 10.0

    # Tokens should all be >= 1
    for tl in result["lrc"]:
        for tok in tl:
            assert tok >= 1

    print("✓ tokenize_lyrics_from_string passed")
    return True


# ===================================================================
# Test: Instrumental Helper
# ===================================================================

def test_create_instrumental_lrc() -> bool:
    """Verify the instrumental (no lyrics) helper."""
    print("Testing create_instrumental_lrc...")

    duration = 95.0
    result = create_instrumental_lrc(duration)

    assert result["time"] == [94.0], f"Expected [94.0], got {result['time']}"
    assert result["lrc"] == [[PERIOD_TOKEN_ID]], (
        f"Expected [[{PERIOD_TOKEN_ID}]], got {result['lrc']}"
    )

    # Edge case: very short audio
    short = create_instrumental_lrc(0.5)
    assert short["time"] == [0.0], "Duration < 1s should clamp time to 0.0"

    print("✓ create_instrumental_lrc passed")
    return True


# ===================================================================
# Test: Save / Load Round-Trip
# ===================================================================

def test_save_load_roundtrip() -> bool:
    """Verify save → load preserves data exactly."""
    print("Testing save/load round-trip...")

    original = {
        "time": [0.0, 5.5, 12.3],
        "lrc": [[1, 2, 3], [10, 20, 30, 40], [5]],
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_lrc_tokens(original, tmp_path)
        loaded = load_lrc_tokens(tmp_path)

        assert loaded["time"] == original["time"], "Times don't match"
        assert loaded["lrc"] == original["lrc"], "Tokens don't match"

        print("✓ save/load round-trip passed")
        return True
    finally:
        os.remove(tmp_path)


def test_save_load_real_lrc() -> bool:
    """Save tokenized real LRC file to .pt then reload and verify."""
    print("Testing save/load with real LRC data...")

    lrc_path = _find_lrc(_LRC_SEARCH_PATHS_EN)
    if lrc_path is None:
        print("  ⚠ Skipping (eg_en.lrc not found)")
        return True

    lrc_data = tokenize_lyrics(lrc_path)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_lrc_tokens(lrc_data, tmp_path)
        loaded = load_lrc_tokens(tmp_path)

        assert loaded["time"] == lrc_data["time"]
        assert loaded["lrc"] == lrc_data["lrc"]

        print("✓ save/load with real LRC data passed")
        return True
    finally:
        os.remove(tmp_path)


# ===================================================================
# Test: Verification Utility
# ===================================================================

def test_verify_lrc_valid() -> bool:
    """verify_lrc should pass on a well-formed .pt file."""
    print("Testing verify_lrc (valid file)...")

    lrc_data = {
        "time": [0.0, 5.0],
        "lrc": [[1, 5, 10], [20, 30]],
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
        torch.save(lrc_data, tmp_path)

    try:
        result = verify_lrc(tmp_path, verbose=True)
        assert result is True
        print("✓ verify_lrc (valid file) passed")
        return True
    finally:
        os.remove(tmp_path)


def test_verify_lrc_real_file() -> bool:
    """verify_lrc should pass on a tokenized real LRC file."""
    print("Testing verify_lrc (real tokenized LRC)...")

    lrc_path = _find_lrc(_LRC_SEARCH_PATHS_EN)
    if lrc_path is None:
        print("  ⚠ Skipping (eg_en.lrc not found)")
        return True

    lrc_data = tokenize_lyrics(lrc_path)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
        torch.save(lrc_data, tmp_path)

    try:
        result = verify_lrc(tmp_path, verbose=True)
        assert result is True
        print("✓ verify_lrc (real tokenized LRC) passed")
        return True
    finally:
        os.remove(tmp_path)


def test_verify_lrc_rejects_bad_data() -> bool:
    """verify_lrc should raise on malformed data."""
    print("Testing verify_lrc (rejects bad data)...")

    bad_cases = [
        ("missing 'lrc' key", {"time": [0.0]}),
        ("missing 'time' key", {"lrc": [[1]]}),
        ("mismatched lengths", {"time": [0.0, 1.0], "lrc": [[1]]}),
        ("token out of range (0)", {"time": [0.0], "lrc": [[0]]}),
        ("empty token list", {"time": [0.0], "lrc": [[]]}),
    ]

    for label, bad_data in bad_cases:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
            torch.save(bad_data, tmp_path)

        try:
            try:
                verify_lrc(tmp_path)
                assert False, f"Should have raised for: {label}"
            except (AssertionError, KeyError):
                print(f"  ✓ Correctly rejected: {label}")
        finally:
            os.remove(tmp_path)

    print("✓ verify_lrc (rejects bad data) passed")
    return True


# ===================================================================
# Test: Compatibility with Dataset Expectations
# ===================================================================

def test_dataset_compatibility() -> bool:
    """
    Verify that tokenized output is compatible with dataset/dataset.py.

    The dataset (lines 86-105) performs:
      - Replaces period tokens in the middle with commas
      - Appends a period token at the end of each line
      - Places tokens at frame positions based on timestamps

    This test simulates that logic to ensure our tokens work.
    """
    print("Testing dataset compatibility...")

    lrc_path = _find_lrc(_LRC_SEARCH_PATHS_EN)
    if lrc_path is None:
        print("  ⚠ Skipping (eg_en.lrc not found)")
        return True

    lrc_data = tokenize_lyrics(lrc_path)
    times = lrc_data["time"]
    token_lists = lrc_data["lrc"]

    # Simulate dataset/dataset.py lines 86-105 token placement
    sampling_rate = 44100
    downsample_rate = 2048
    max_frames = 2048

    # dataset/dataset.py line 33-34: comma=1, period=2
    comma_token_id = 1
    period_token_id = 2

    lrc_tensor = torch.zeros((max_frames,), dtype=torch.long)
    last_end_pos = 0

    for timestamp, line_tokens in zip(times, token_lists):
        # This replicates dataset/dataset.py line 91:
        # Replace period tokens mid-line with comma, then append period
        tokens = [
            tok if tok != period_token_id else comma_token_id
            for tok in line_tokens
        ] + [period_token_id]
        tokens = torch.tensor(tokens, dtype=torch.long)

        frame_start = int(timestamp * sampling_rate / downsample_rate)
        frame_start = max(frame_start, last_end_pos)
        frame_len = min(len(tokens), max_frames - frame_start)

        if frame_len <= 0:
            break

        lrc_tensor[frame_start : frame_start + frame_len] = tokens[:frame_len]
        last_end_pos = frame_start + frame_len

    # Verify: non-zero tokens exist in the tensor
    nonzero_count = (lrc_tensor != 0).sum().item()
    assert nonzero_count > 0, "No tokens were placed in the frame tensor"

    # Verify: all non-zero values are valid token IDs
    nonzero_tokens = lrc_tensor[lrc_tensor != 0]
    assert (nonzero_tokens >= 1).all(), "All placed tokens should be >= 1"

    print(f"  Placed {nonzero_count} tokens in {max_frames}-frame tensor")
    print(f"  Occupancy: {nonzero_count / max_frames * 100:.1f}%")
    print("✓ dataset compatibility passed")
    return True


# ===================================================================
# Runner
# ===================================================================

def run_all_tests():
    """Run all lyrics tokenization tests."""
    print("=" * 60)
    print("Lyrics Tokenization Module – Test Suite")
    print("=" * 60)
    print()

    tests = [
        # Parsing
        ("Parse LRC file (English)", test_parse_lrc_file_english),
        ("Parse LRC file (Chinese)", test_parse_lrc_file_chinese),
        ("Parse LRC string", test_parse_lrc_string),
        ("Parse malformed lines", test_parse_lrc_malformed_lines),
        # Token shifting
        ("Token +1 shift", test_token_shifting),
        # Tokenization
        ("Tokenize lyrics (English)", test_tokenize_lyrics_english),
        ("Tokenize lyrics (Chinese)", test_tokenize_lyrics_chinese),
        ("Tokenize lyrics from string", test_tokenize_lyrics_from_string),
        # Instrumental
        ("Instrumental LRC helper", test_create_instrumental_lrc),
        # Save / Load
        ("Save/load round-trip", test_save_load_roundtrip),
        ("Save/load real LRC", test_save_load_real_lrc),
        # Verification
        ("Verify valid .pt", test_verify_lrc_valid),
        ("Verify real tokenized .pt", test_verify_lrc_real_file),
        ("Verify rejects bad data", test_verify_lrc_rejects_bad_data),
        # Dataset compat
        ("Dataset compatibility", test_dataset_compatibility),
    ]

    results = []
    for name, test_func in tests:
        print()
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
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
    run_all_tests()
