"""
Tests for Style Embedding Extraction Module

Run these tests to verify the MuQ-MuLan style extraction pipeline is working
correctly before processing your full dataset.

Tests are ordered by dependency:
    1. Offline tests (no model required) - save/verify/combined math
    2. Model loading
    3. Audio extraction (requires model + test audio)
    4. Text extraction (requires model)
    5. Combined extraction (requires model + test audio) [EXPERIMENTAL]

Usage (recommended - run from repo root):
    python preprocess/test/test_style_extraction.py
    python preprocess/test/test_style_extraction.py cpu      # Force CPU
    python preprocess/test/test_style_extraction.py cuda     # Force CUDA
    python preprocess/test/test_style_extraction.py mps      # Force MPS
"""

import sys
import os

# Add preprocess directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
preprocess_dir = os.path.dirname(script_dir)
sys.path.insert(0, preprocess_dir)

import torch
from style_extraction import (
    STYLE_DIM,
    MUQ_SAMPLE_RATE,
    MIN_AUDIO_DURATION,
    load_muq,
    extract_style_from_audio,
    extract_style_from_text,
    extract_style_combined,
    save_style_file,
    verify_style_file,
)


# =============================================================================
# Test Audio Discovery
# =============================================================================

def find_test_audio():
    """
    Find a test audio file to use for audio-based tests.

    Returns:
        Path to test audio file, or None if not found
    """
    test_audio_paths = [
        "test_audio.mp3",                              # from preprocess/test/ dir
        "test/test_audio.mp3",                          # from preprocess/ dir
        "preprocess/test/test_audio.mp3",               # from repo root
        "infer/example/output-windows/en-utf8-fixed.mp3",
        "infer/example/output-windows/en-without-utf8-fix.mp3",
    ]

    for path in test_audio_paths:
        if os.path.exists(path):
            return path

    return None


# =============================================================================
# Offline Tests (no model needed)
# =============================================================================

def test_save_and_verify_style_file() -> bool:
    """
    Test save_style_file and verify_style_file with synthetic embeddings.

    No model required - uses random tensors with the correct shape.
    """
    print("Testing save/verify style file...")

    # Test 1: Save and verify [1, 512] tensor
    style_2d = torch.randn(1, STYLE_DIM)
    save_style_file(style_2d, "test_style_2d.pt")
    verify_style_file("test_style_2d.pt", verbose=True)
    os.remove("test_style_2d.pt")
    print("  ✓ [1, 512] tensor save/verify")

    # Test 2: Save and verify [512] tensor
    style_1d = torch.randn(STYLE_DIM)
    save_style_file(style_1d, "test_style_1d.pt")
    verify_style_file("test_style_1d.pt", verbose=True)
    os.remove("test_style_1d.pt")
    print("  ✓ [512] tensor save/verify")

    # Test 3: Reject wrong dimensions
    try:
        bad_style = torch.randn(1, 256)
        save_style_file(bad_style, "test_style_bad.pt")
        print("  ✗ Should have rejected [1, 256] tensor")
        os.remove("test_style_bad.pt")
        return False
    except AssertionError:
        print("  ✓ Correctly rejected [1, 256] tensor")

    # Test 4: Reject NaN values
    nan_style = torch.randn(1, STYLE_DIM)
    nan_style[0, 0] = float("nan")
    torch.save(nan_style, "test_style_nan.pt")
    try:
        verify_style_file("test_style_nan.pt", verbose=False)
        print("  ✗ Should have rejected NaN tensor")
        os.remove("test_style_nan.pt")
        return False
    except ValueError:
        print("  ✓ Correctly rejected NaN tensor")
        os.remove("test_style_nan.pt")

    print("✓ Save/verify style file tests passed")
    return True


def test_combined_math() -> bool:
    """
    Test the interpolation math used in combined mode.

    No model required - verifies the linear algebra is correct using
    synthetic embeddings.
    """
    print("Testing combined embedding math...")

    audio_emb = torch.randn(1, STYLE_DIM)
    text_emb = torch.randn(1, STYLE_DIM)

    # Test 1: alpha=1.0 should return pure audio
    combined_audio = 1.0 * audio_emb + 0.0 * text_emb
    assert torch.allclose(combined_audio, audio_emb), "alpha=1.0 should give pure audio"
    print("  ✓ alpha=1.0 gives pure audio embedding")

    # Test 2: alpha=0.0 should return pure text
    combined_text = 0.0 * audio_emb + 1.0 * text_emb
    assert torch.allclose(combined_text, text_emb), "alpha=0.0 should give pure text"
    print("  ✓ alpha=0.0 gives pure text embedding")

    # Test 3: alpha=0.5 should be the midpoint
    combined_mid = 0.5 * audio_emb + 0.5 * text_emb
    expected_mid = (audio_emb + text_emb) / 2
    assert torch.allclose(combined_mid, expected_mid), "alpha=0.5 should be average"
    print("  ✓ alpha=0.5 gives average of both")

    # Test 4: L2 normalization produces unit vector
    normalized = torch.nn.functional.normalize(combined_mid, p=2, dim=-1)
    norm = torch.norm(normalized, p=2, dim=-1).item()
    assert abs(norm - 1.0) < 1e-5, f"L2 norm should be 1.0, got {norm}"
    print("  ✓ L2 normalization produces unit vector")

    # Test 5: Interpolated embedding has correct shape
    assert combined_mid.shape == (1, STYLE_DIM), \
        f"Expected [1, {STYLE_DIM}], got {list(combined_mid.shape)}"
    print(f"  ✓ Output shape is [1, {STYLE_DIM}]")

    print("✓ Combined embedding math tests passed")
    return True


# =============================================================================
# Model Tests (require MuQ-MuLan download)
# =============================================================================

def test_muq_loading(device: str = "cuda", cache_dir: str = "./pretrained") -> bool:
    """
    Test that the MuQ-MuLan model can be loaded successfully.

    Args:
        device: torch device
        cache_dir: Model cache directory

    Returns:
        True if successful
    """
    print("Testing MuQ-MuLan model loading...")
    muq = load_muq(device=device, cache_dir=cache_dir)
    assert muq is not None, "MuQ model is None"
    print("✓ MuQ-MuLan model loaded successfully")
    return True


def test_text_extraction(device: str = "cuda", cache_dir: str = "./pretrained") -> bool:
    """
    Test text-based style extraction.

    Args:
        device: torch device
        cache_dir: Model cache directory

    Returns:
        True if successful
    """
    print("Testing text-based style extraction...")

    muq = load_muq(device=device, cache_dir=cache_dir)

    # Test 1: Single text prompt
    style = extract_style_from_text("A smooth jazz track with piano", muq, device=device)
    assert style.shape == (1, STYLE_DIM), f"Expected [1, {STYLE_DIM}], got {list(style.shape)}"
    assert not torch.isnan(style).any(), "Style contains NaN"
    print(f"  ✓ Single prompt: shape={list(style.shape)}, norm={torch.norm(style).item():.4f}")

    # Test 2: Different prompts produce different embeddings
    style_a = extract_style_from_text("Heavy metal with distorted guitars", muq, device=device)
    style_b = extract_style_from_text("Gentle classical piano sonata", muq, device=device)
    cosine_sim = torch.nn.functional.cosine_similarity(style_a, style_b, dim=-1).item()
    print(f"  ✓ Different prompts cosine similarity: {cosine_sim:.4f} (should not be ~1.0)")
    assert cosine_sim < 0.99, "Different prompts should produce different embeddings"

    # Test 3: Save and verify round-trip
    save_style_file(style, "test_style_text.pt")
    verify_style_file("test_style_text.pt", verbose=False)
    os.remove("test_style_text.pt")
    print("  ✓ Save/verify round-trip")

    print("✓ Text extraction tests passed")
    return True


def test_audio_extraction(device: str = "cuda", cache_dir: str = "./pretrained") -> bool:
    """
    Test audio-based style extraction.

    Requires a test audio file (>=10 seconds).

    Args:
        device: torch device
        cache_dir: Model cache directory

    Returns:
        True if successful
    """
    print("Testing audio-based style extraction...")

    test_audio = find_test_audio()
    if test_audio is None:
        print("  ⚠ Skipping audio extraction test (no test audio file found)")
        return True

    print(f"  Using test audio: {test_audio}")

    muq = load_muq(device=device, cache_dir=cache_dir)

    # Test 1: Extract style from audio
    style = extract_style_from_audio(test_audio, muq, device=device)
    assert style.shape == (1, STYLE_DIM), f"Expected [1, {STYLE_DIM}], got {list(style.shape)}"
    assert not torch.isnan(style).any(), "Style contains NaN"
    print(f"  ✓ Audio extraction: shape={list(style.shape)}, norm={torch.norm(style).item():.4f}")

    # Test 2: Save and verify round-trip
    save_style_file(style, "test_style_audio.pt")
    verify_style_file("test_style_audio.pt", verbose=True)
    os.remove("test_style_audio.pt")
    print("  ✓ Save/verify round-trip")

    print("✓ Audio extraction tests passed")
    return True


def test_combined_extraction(device: str = "cuda", cache_dir: str = "./pretrained") -> bool:
    """
    [EXPERIMENTAL] Test combined audio+text style extraction.

    Verifies that the interpolation produces valid embeddings and that
    the alpha parameter controls the blend as expected.

    Args:
        device: torch device
        cache_dir: Model cache directory

    Returns:
        True if successful
    """
    print("Testing combined style extraction [EXPERIMENTAL]...")

    test_audio = find_test_audio()
    if test_audio is None:
        print("  ⚠ Skipping combined extraction test (no test audio file found)")
        return True

    print(f"  Using test audio: {test_audio}")

    muq = load_muq(device=device, cache_dir=cache_dir)
    text_prompt = "Energetic rock with heavy drums"

    # Get individual embeddings for comparison
    audio_emb = extract_style_from_audio(test_audio, muq, device=device)
    text_emb = extract_style_from_text(text_prompt, muq, device=device)

    # Test 1: alpha=1.0 should match pure audio (before normalization)
    combined_audio_only = extract_style_combined(
        test_audio, text_prompt, muq, alpha=1.0, normalize=False, device=device,
    )
    assert torch.allclose(combined_audio_only, audio_emb, atol=1e-5), \
        "alpha=1.0 should match pure audio embedding"
    print("  ✓ alpha=1.0 matches pure audio embedding")

    # Test 2: alpha=0.0 should match pure text (before normalization)
    combined_text_only = extract_style_combined(
        test_audio, text_prompt, muq, alpha=0.0, normalize=False, device=device,
    )
    assert torch.allclose(combined_text_only, text_emb, atol=1e-5), \
        "alpha=0.0 should match pure text embedding"
    print("  ✓ alpha=0.0 matches pure text embedding")

    # Test 3: alpha=0.5 with normalization produces unit vector
    combined_norm = extract_style_combined(
        test_audio, text_prompt, muq, alpha=0.5, normalize=True, device=device,
    )
    norm = torch.norm(combined_norm, p=2, dim=-1).item()
    assert abs(norm - 1.0) < 1e-4, f"Normalized combined should have unit norm, got {norm}"
    print(f"  ✓ alpha=0.5 normalized: norm={norm:.6f}")

    # Test 4: Combined embedding has valid shape
    assert combined_norm.shape == (1, STYLE_DIM), \
        f"Expected [1, {STYLE_DIM}], got {list(combined_norm.shape)}"
    print(f"  ✓ Output shape: {list(combined_norm.shape)}")

    # Test 5: Show similarity between audio, text, and combined
    cos_audio_text = torch.nn.functional.cosine_similarity(audio_emb, text_emb, dim=-1).item()
    cos_audio_combined = torch.nn.functional.cosine_similarity(audio_emb, combined_norm, dim=-1).item()
    cos_text_combined = torch.nn.functional.cosine_similarity(text_emb, combined_norm, dim=-1).item()
    print(f"  ✓ Cosine similarities:")
    print(f"      audio vs text:     {cos_audio_text:.4f}")
    print(f"      audio vs combined: {cos_audio_combined:.4f}")
    print(f"      text  vs combined: {cos_text_combined:.4f}")

    # Test 6: Save and verify round-trip
    save_style_file(combined_norm, "test_style_combined.pt")
    verify_style_file("test_style_combined.pt", verbose=False)
    os.remove("test_style_combined.pt")
    print("  ✓ Save/verify round-trip")

    # Test 7: Invalid alpha should raise
    try:
        extract_style_combined(
            test_audio, text_prompt, muq, alpha=1.5, device=device,
        )
        print("  ✗ Should have rejected alpha=1.5")
        return False
    except ValueError:
        print("  ✓ Correctly rejected alpha=1.5")

    print("✓ Combined extraction tests passed [EXPERIMENTAL]")
    return True


# =============================================================================
# Test Runner
# =============================================================================

def run_all_tests(device: str = "cpu"):
    """
    Run all tests in dependency order.

    Offline tests run first (no model needed), then model-dependent tests.

    Args:
        device: torch device ('cuda', 'mps', or 'cpu')
    """
    print("=" * 60)
    print("Style Embedding Extraction Module - Test Suite")
    print("=" * 60)
    print(f"Device: {device}")
    print()

    tests = [
        # Offline tests (no model download needed)
        ("Save/Verify Style File", lambda: test_save_and_verify_style_file()),
        ("Combined Embedding Math", lambda: test_combined_math()),
        # Model-dependent tests
        ("MuQ-MuLan Loading", lambda: test_muq_loading(device=device)),
        ("Text Extraction", lambda: test_text_extraction(device=device)),
        ("Audio Extraction", lambda: test_audio_extraction(device=device)),
        ("Combined Extraction [EXPERIMENTAL]", lambda: test_combined_extraction(device=device)),
    ]

    results = []
    for name, test_func in tests:
        print()
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ {name} test failed with exception: {e}")
            results.append((name, False))

    print()
    print("=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print()
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. Check the output above.")

    print("=" * 60)


if __name__ == "__main__":
    """
    Run tests when executing this file directly.

    Usage:
        python preprocess/test/test_style_extraction.py          # Auto-detect device
        python preprocess/test/test_style_extraction.py cpu      # Force CPU
        python preprocess/test/test_style_extraction.py cuda     # Force CUDA
        python preprocess/test/test_style_extraction.py mps      # Force MPS
    """

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
