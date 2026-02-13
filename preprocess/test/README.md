# Test Data for Preprocessing Module

This directory contains test audio files used for validating the preprocessing pipeline.

## Test Files

- `test_audio.mp3` - Sample audio file for testing encoding pipeline (~35 seconds, 1.5MB)

## Usage

**Recommended:** Run the test suite from the repository root:

```bash
python preprocess/test/test_audio_processing.py
```

The test can technically run from other directories, but running from the repository root is most reliable.

## Generated Test Files

The test suite may generate temporary files:
- `test_latent.pt` - Test output from encoding pipeline (automatically cleaned up)

These are automatically removed after tests complete.

## For Standalone Package

When separating this preprocessing code into a standalone repository, this test directory ensures the module can be tested independently without dependencies on the main DiffRhythm codebase.
