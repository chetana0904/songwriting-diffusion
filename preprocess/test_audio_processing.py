"""
Tests for VAE Audio Processing Module

Run these tests to verify the audio processing pipeline is working correctly
before processing your full dataset.

Usage:
    python preprocess/test_audio_processing.py
"""

import torch
from audio_processing import (
    load_vae,
    prepare_audio,
    normalize_audio,
    encode_audio_to_latent,
    save_latent,
    verify_latent,
)


def test_vae_loading(device: str = "cuda", cache_dir: str = "./pretrained") -> bool:
    """
    Test that the VAE model can be loaded successfully.
    
    Args:
        device: torch device
        cache_dir: Model cache directory
    
    Returns:
        True if successful
    """
    print("Testing VAE model loading...")
    vae = load_vae(device=device, cache_dir=cache_dir)
    assert vae is not None, "VAE model is None"
    print("✓ VAE model loaded successfully")
    return True


def test_audio_preparation(device: str = "cuda") -> bool:
    """
    Test audio preparation pipeline with synthetic audio.
    
    Args:
        device: torch device
    
    Returns:
        True if successful
    """
    print("Testing audio preparation...")
    
    # Test 1: Mono to stereo
    audio_mono = torch.randn(1, 44100)  # 1 second mono
    prepared = prepare_audio(audio_mono, in_sr=44100, device=device)
    assert prepared.shape == (1, 2, 44100), f"Expected [1, 2, 44100], got {prepared.shape}"
    print("  ✓ Mono to stereo conversion")
    
    # Test 2: Resampling
    audio_48k = torch.randn(2, 48000)  # 1 second stereo at 48kHz
    prepared = prepare_audio(audio_48k, in_sr=48000, device=device)
    assert prepared.shape[2] == 44100, f"Expected 44100 samples, got {prepared.shape[2]}"
    print("  ✓ Resampling 48kHz → 44.1kHz")
    
    # Test 3: Normalization
    audio = torch.randn(1, 2, 44100)
    normalized = normalize_audio(audio, target_dbfs=-6)
    max_amp = torch.max(torch.abs(normalized)).item()
    expected_amp = 10.0 ** (-6.0 / 20.0)
    assert abs(max_amp - expected_amp) < 0.01, f"Normalization failed: {max_amp:.3f}"
    print("  ✓ Normalization to -6 dBFS")
    
    print("✓ Audio preparation tests passed")
    return True


def test_encoding_pipeline(device: str = "cuda") -> bool:
    """
    Test the complete encoding pipeline with a real audio file.
    
    Note: Requires a test audio file to exist.
    
    Args:
        device: torch device
    
    Returns:
        True if successful
    """
    import os
    
    print("Testing encoding pipeline...")
    
    # Check if test audio exists
    test_audio_paths = [
        "test_audio.wav",
        "infer/example/test.wav",
        "examples/test.wav"
    ]
    
    test_audio = None
    for path in test_audio_paths:
        if os.path.exists(path):
            test_audio = path
            break
    
    if test_audio is None:
        print("  ⚠ Skipping encoding test (no test audio file found)")
        print(f"    Create one of these files to test: {test_audio_paths}")
        return True
    
    try:
        # Load VAE
        vae = load_vae(device=device)
        
        # Encode audio
        latent = encode_audio_to_latent(test_audio, vae, device=device, chunked=True)
        
        # Verify shape
        assert latent.dim() == 3, f"Expected 3D tensor, got {latent.dim()}D"
        assert latent.shape[1] == 64, f"Expected 64 latent dims, got {latent.shape[1]}"
        
        # Save and verify
        save_latent(latent, "test_latent.pt")
        verify_latent("test_latent.pt", verbose=True)
        
        # Clean up
        os.remove("test_latent.pt")
        
        print("✓ Encoding pipeline test passed")
        return True
        
    except Exception as e:
        print(f"✗ Encoding pipeline test failed: {e}")
        return False


def run_all_tests(device: str = "cpu"):
    """
    Run all tests.
    
    Args:
        device: torch device ('cuda' or 'cpu')
    """
    print("=" * 60)
    print("VAE Audio Processing Module - Test Suite")
    print("=" * 60)
    print(f"Device: {device}")
    print()
    
    tests = [
        ("VAE Loading", lambda: test_vae_loading(device=device)),
        ("Audio Preparation", lambda: test_audio_preparation(device=device)),
        ("Encoding Pipeline", lambda: test_encoding_pipeline(device=device)),
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
        python preprocess/test_audio_processing.py
    """
    import sys

    # Use CUDA if specified, fallback to MPS if available, otherwise use CPU
    # Pass 'cuda' as argument to test on GPU
    if len(sys.argv) > 1 and sys.argv[1] == "cuda":
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    run_all_tests(device=device)
