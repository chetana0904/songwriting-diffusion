# VAE Latent Generation - Technical Implementation Plan

## Overview

This document provides the technical specification for implementing the VAE latent generation component of the DiffRhythm preprocessing pipeline. This is the first and most critical component, as the entire training pipeline operates in VAE latent space, not on raw audio or mel spectrograms.

**Purpose**: Encode audio files into 64-dimensional VAE latent representations that serve as the ground truth for training.

---

## 1. The VAE Model

### 1.1 Model Details

- **Source**: HuggingFace repository `ASLP-lab/DiffRhythm-vae`
- **Type**: Pre-trained TorchScript model (black-box, cannot inspect architecture)
- **File**: `vae_model.pt`

### 1.2 Model Interface

```python
# Encoding: audio → latent
vae.encode_export(audio)  # [batch, 2, samples] → [batch, 128, T]

# Decoding: latent → audio (used in inference, not preprocessing)
vae.decode_export(latent)  # [batch, 64, T] → [batch, 2, samples]
```

### 1.3 Input/Output Specifications

| Parameter | Specification |
|-----------|---------------|
| **Input Shape** | `[batch, 2, samples]` |
| **Input Channels** | 2 (stereo) - REQUIRED |
| **Input Sample Rate** | 44100 Hz - REQUIRED |
| **Output Shape** | `[batch, 128, T]` |
| **Output Channels** | 128 (64 mean + 64 scale) |
| **Compression Ratio** | 2048:1 (fixed) |

### 1.4 Why 128 Output Channels?

The VAE is a **Variational Autoencoder**, which models a distribution rather than a deterministic encoding:

- **Channels 0-63**: Mean (μ) of the latent distribution
- **Channels 64-127**: Scale (σ) of the latent distribution

For training, we use the **mean only** (deterministic encoding):

```python
latent = vae.encode_export(audio)  # [1, 128, T]
mean, scale = latent.chunk(2, dim=1)  # Each [1, 64, T]
latent_for_training = mean  # Use this!
```

**Alternative**: Sample from the distribution: `mean + scale * torch.randn_like(mean)`

### 1.5 Compression Ratio

```
downsample_rate = 2048
```

**This means**:
- 2048 audio samples → 1 latent frame
- 1 second of 44.1kHz audio → 44100/2048 ≈ 21.53 latent frames
- 95 seconds of audio → ~2048 latent frames

**Example calculation**:
```
Audio: 95 seconds at 44.1kHz
Samples: 95 × 44100 = 4,189,500
Latent frames: 4,189,500 / 2048 ≈ 2045
Output shape: [1, 64, 2045]
```

---

## 2. Audio Preparation Pipeline

Audio must be prepared to **exact specifications** before VAE encoding. Any deviation will cause dimension mismatches or quality issues.

### 2.1 Step-by-Step Pipeline

```
Raw Audio File
    ↓
[1] Load with torchaudio
    ↓
[2] Resample to 44.1kHz (if needed)
    ↓
[3] Convert to stereo (2 channels)
    ↓
[4] Normalize to -6 dBFS (optional)
    ↓
[5] Add batch dimension
    ↓
[6] Move to GPU
    ↓
Ready for VAE encoding
```

### 2.2 Implementation Details

#### Step 1: Load Audio

```python
audio, sr = torchaudio.load(audio_path)
# Output: audio shape = [channels, samples]
```

**Supported formats**: WAV, MP3, FLAC, M4A, OGG (anything torchaudio supports)

#### Step 2: Resample to 44.1kHz

```python
target_sr = 44100
if sr != target_sr:
    audio = torchaudio.functional.resample(audio, sr, target_sr)
```

**Critical**: The model expects exactly 44.1kHz. The config file specifies `mel_dim=64`, which corresponds to the VAE latent dimension at 44.1kHz sample rate.

**Reference**: `infer/infer_utils.py` lines 79-98

#### Step 3: Convert to Stereo

```python
if audio.shape[0] == 1:
    # Mono → Stereo: duplicate the channel
    audio = audio.repeat(2, 1)
elif audio.shape[0] > 2:
    # Multi-channel → Stereo: take first two
    audio = audio[:2, :]
# Result: [2, samples]
```

**Why stereo is required**: The VAE was trained on stereo audio and expects exactly 2 channels. The TorchScript model will error on other channel counts.

#### Step 4: Normalize Audio (Optional but Recommended)

```python
def normalize_audio(audio, target_dbfs=-6):
    """
    Peak normalize audio to target dBFS.
    -6 dBFS provides headroom and prevents clipping.
    """
    max_amplitude = torch.max(torch.abs(audio))
    target_amplitude = 10.0 ** (target_dbfs / 20.0)  # -6 dBFS ≈ 0.5
    scale_factor = target_amplitude / max_amplitude
    return audio * scale_factor
```

**Benefits**:
- Consistent volume levels across dataset
- Prevents clipping during processing
- Improved training stability

**Reference**: `infer/infer_utils.py` lines 43-51

#### Step 5: Add Batch Dimension

```python
audio = audio.unsqueeze(0)
# [2, samples] → [1, 2, samples]
```

#### Step 6: Move to Device

```python
audio = audio.to(device)  # GPU for faster processing
```

---

## 3. Chunked Encoding Strategy

### 3.1 Why Chunked Encoding?

Processing long audio files in one pass can cause:
- **Out of Memory (OOM) errors** on GPU
- **Excessive memory usage** even if it fits
- **Inefficient processing** of very long files

**Memory requirements** (approximate):

| Audio Length | Latent Frames | VRAM for Encoding |
|--------------|---------------|-------------------|
| 30 seconds   | ~645          | ~2 GB            |
| 95 seconds   | ~2048         | ~4 GB            |
| 285 seconds  | ~6144         | ~8 GB            |

**Recommendation**: Use chunked encoding for audio > 30 seconds.

**Reference**: `infer/infer_utils.py` lines 154-211

### 3.2 Chunking Parameters

```python
chunk_size = 128        # Latent frames per chunk
overlap = 32            # Overlapping latent frames between chunks
downsample_rate = 2048  # Audio samples per latent frame

# Derived values
chunk_samples = chunk_size * downsample_rate      # 262,144 audio samples
overlap_samples = overlap * downsample_rate       # 65,536 audio samples  
hop_samples = chunk_samples - overlap_samples     # 196,608 audio samples
```

**Visual representation**:
```
Chunk 1: [=================]
Chunk 2:            [=================]
Chunk 3:                       [=================]
         |← hop →|  |← overlap →|
```

### 3.3 Algorithm Overview

```python
def encode_audio_chunked(audio, vae, chunk_size=128, overlap=32):
    """
    Chunked VAE encoding with overlap handling.
    
    Args:
        audio: Tensor [1, 2, samples] on device
        vae: Loaded VAE model
        chunk_size: Number of latent frames per chunk
        overlap: Number of overlapping frames
    
    Returns:
        latent: Tensor [1, 128, T] on device
    """
    downsample_rate = 2048
    total_samples = audio.shape[2]
    
    chunk_samples = chunk_size * downsample_rate
    overlap_samples = overlap * downsample_rate
    hop_samples = chunk_samples - overlap_samples
    
    # 1. Calculate output dimensions
    total_frames = total_samples // downsample_rate
    output = torch.zeros((1, 128, total_frames), device=audio.device)
    
    # 2. Process chunks with sliding window
    chunks = []
    for i in range(0, total_samples - chunk_samples + 1, hop_samples):
        chunk = audio[:, :, i:i+chunk_samples]
        chunks.append(chunk)
    
    # 3. Handle final chunk (align to end)
    if len(chunks) == 0 or chunks[-1].shape[2] < chunk_samples:
        final_chunk = audio[:, :, -chunk_samples:]
        chunks.append(final_chunk)
    
    num_chunks = len(chunks)
    
    # 4. Encode each chunk and place in output
    for i, chunk in enumerate(chunks):
        chunk_latent = vae.encode_export(chunk)  # [1, 128, chunk_size]
        
        # Calculate position in output
        if i == num_chunks - 1:
            # Last chunk: align to end
            t_end = total_frames
            t_start = t_end - chunk_latent.shape[2]
        else:
            # Regular chunk
            t_start = i * (chunk_size - overlap)
            t_end = t_start + chunk_size
        
        # Handle overlap regions
        ol = overlap // 2
        chunk_start = 0
        chunk_end = chunk_latent.shape[2]
        
        if i > 0:
            # Skip first half of overlap
            t_start += ol
            chunk_start += ol
        if i < num_chunks - 1:
            # Skip second half of overlap
            t_end -= ol
            chunk_end -= ol
        
        # Place chunk in output
        output[:, :, t_start:t_end] = chunk_latent[:, :, chunk_start:chunk_end]
    
    return output
```

### 3.4 Overlap Handling Details

The overlap prevents boundary artifacts. Here's how it works:

**First Chunk**:
- Use frames: `[0 : chunk_size - overlap//2]`
- Discard: Last `overlap//2` frames

**Middle Chunks**:
- Use frames: `[overlap//2 : chunk_size - overlap//2]`
- Discard: First `overlap//2` and last `overlap//2` frames

**Last Chunk**:
- Align to the end of the audio
- Use frames: `[overlap//2 : end]`
- Discard: First `overlap//2` frames

This creates smooth transitions without discontinuities or gaps.

### 3.5 Edge Cases

**Short audio** (< chunk_size):
```python
if total_samples < chunk_samples:
    # Pad audio to minimum chunk size
    pad_size = chunk_samples - total_samples
    audio = torch.nn.functional.pad(audio, (0, pad_size))
```

**Exact multiple of chunk size**:
- The algorithm handles this correctly
- No special case needed

---

## 4. Output Format

### 4.1 Required Format

The dataset class (`dataset/dataset.py` lines 63-64) expects:

```python
latent = torch.load(latent_path, map_location='cpu')  # [b, d, t]
latent = latent.squeeze(0)  # Removes batch dim if present
```

**This means both shapes are valid**:
- `[1, 64, T]` - with batch dimension
- `[64, T]` - without batch dimension

**Recommendation**: Save with batch dimension `[1, 64, T]` for consistency.

### 4.2 Saving Latents

```python
def save_latent(latent, output_path):
    """
    Save latent tensor to disk.
    
    Args:
        latent: Tensor [1, 64, T] or [64, T]
        output_path: Path to .pt file
    """
    # Ensure it's on CPU before saving
    latent = latent.cpu()
    
    # Verify shape
    assert latent.shape[-2] == 64 or latent.shape[-1] == 64, \
        f"Expected 64-dim latent, got shape {latent.shape}"
    
    # Save
    torch.save(latent, output_path)
```

### 4.3 Validation

After saving, verify the file:

```python
def verify_latent(path):
    """Verify a latent file is correctly formatted."""
    latent = torch.load(path, map_location='cpu')
    
    print(f"Shape: {latent.shape}")
    print(f"Dtype: {latent.dtype}")
    print(f"Min: {latent.min():.4f}, Max: {latent.max():.4f}")
    
    # Check dimensions
    if latent.dim() == 3:
        assert latent.shape[0] == 1, "Batch size should be 1"
        assert latent.shape[1] == 64, "Latent dim should be 64"
    elif latent.dim() == 2:
        assert latent.shape[0] == 64, "Latent dim should be 64"
    else:
        raise ValueError(f"Unexpected dimensions: {latent.dim()}")
    
    print("✓ Latent file is valid")
```

---

## 5. Model Loading

### 5.1 Download and Load

```python
import torch
from huggingface_hub import hf_hub_download

def load_vae(device="cuda", cache_dir="./pretrained"):
    """
    Load the pre-trained VAE model from HuggingFace.
    
    Args:
        device: torch device (cuda/cpu)
        cache_dir: Directory to cache downloaded models
    
    Returns:
        vae: Loaded TorchScript VAE model
    """
    # Download from HuggingFace Hub
    vae_path = hf_hub_download(
        repo_id="ASLP-lab/DiffRhythm-vae",
        filename="vae_model.pt",
        cache_dir=cache_dir
    )
    
    # Load TorchScript model
    vae = torch.jit.load(vae_path, map_location="cpu")
    
    # Move to device and set to eval mode
    vae = vae.to(device)
    vae.eval()
    
    return vae
```

### 5.2 Important Notes

- **Load once, reuse**: Don't reload the model for each audio file
- **Eval mode**: Always call `.eval()` to disable dropout/batchnorm updates
- **No gradients needed**: Wrap all encoding in `torch.no_grad()` context

---

## 6. Complete Processing Function

```python
def encode_audio_to_latent(audio_path, vae, device="cuda", chunked=True):
    """
    Complete pipeline: audio file → VAE latent.
    
    Args:
        audio_path: Path to audio file
        vae: Loaded VAE model
        device: torch device
        chunked: Use chunked encoding (recommended for audio > 30s)
    
    Returns:
        latent: Tensor [1, 64, T] on CPU
    """
    # Step 1: Load audio
    audio, sr = torchaudio.load(audio_path)
    
    # Step 2: Prepare audio
    audio = prepare_audio(audio, sr, target_sr=44100, device=device)
    audio = normalize_audio(audio, target_dbfs=-6)
    
    # Step 3: Encode through VAE
    with torch.no_grad():
        if chunked:
            latent = encode_audio_chunked(audio, vae)
        else:
            latent = vae.encode_export(audio)
        
        # Step 4: Extract mean (discard scale)
        mean, scale = latent.chunk(2, dim=1)
        latent = mean  # [1, 64, T]
    
    # Step 5: Move to CPU for saving
    return latent.cpu()
```

---

## 7. Error Handling

### 7.1 Common Issues

```python
def encode_audio_to_latent_safe(audio_path, vae, device="cuda", chunked=True):
    """
    Safe version with comprehensive error handling.
    """
    try:
        # Load audio
        audio, sr = torchaudio.load(audio_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio {audio_path}: {e}")
    
    # Check audio is not empty
    if audio.shape[1] == 0:
        raise ValueError(f"Audio file {audio_path} is empty")
    
    # Check minimum length (at least 1 second)
    if audio.shape[1] < sr:
        raise ValueError(f"Audio file {audio_path} too short: {audio.shape[1]/sr:.2f}s")
    
    try:
        # Prepare audio
        audio = prepare_audio(audio, sr, target_sr=44100, device=device)
        audio = normalize_audio(audio, target_dbfs=-6)
    except Exception as e:
        raise RuntimeError(f"Failed to prepare audio {audio_path}: {e}")
    
    try:
        # Encode
        with torch.no_grad():
            if chunked:
                latent = encode_audio_chunked(audio, vae)
            else:
                latent = vae.encode_export(audio)
            
            mean, scale = latent.chunk(2, dim=1)
            latent = mean
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            f"GPU OOM encoding {audio_path}. "
            "Try: (1) Use chunked=True, (2) Smaller chunk_size, (3) CPU device"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to encode {audio_path}: {e}")
    
    return latent.cpu()
```

### 7.2 Error Types to Handle

| Error | Cause | Solution |
|-------|-------|----------|
| Load failure | Corrupt audio file | Skip or re-encode source |
| Empty audio | 0 samples | Skip file |
| Too short | < 1 second | Skip or pad |
| OOM error | Audio too long | Use chunked encoding |
| Sample rate error | Unsupported rate | May need manual conversion |
| Channel error | >2 channels stereo | Already handled in prepare_audio |

---

## 8. Performance Optimization

### 8.1 GPU Optimization

```python
# Keep data on GPU throughout processing
audio = audio.to(device)  # Load to GPU once

# Use no_grad context
with torch.no_grad():
    latent = vae.encode_export(audio)

# Only move to CPU when saving
latent = latent.cpu()
```

### 8.2 Mixed Precision (Optional)

```python
# If VAE supports fp16 (test first!)
with torch.cuda.amp.autocast():
    latent = vae.encode_export(audio)
```

**Warning**: TorchScript models may not support autocast. Test carefully.

### 8.3 Batch Processing (Advanced)

If you have sufficient VRAM, process multiple short files in one batch:

```python
# Stack multiple audio files
audio_batch = torch.stack([audio1, audio2, audio3])  # [3, 2, samples]

# Encode in batch
latent_batch = vae.encode_export(audio_batch)  # [3, 128, T]
```

**Note**: All files must have the same length (pad to max length).

---

## 9. Testing Strategy

### 9.1 Unit Tests

Test each component individually:

```python
# Test 1: Model loads correctly
vae = load_vae(device="cuda")
assert vae is not None

# Test 2: Audio preparation
audio = torch.randn(1, 44100)  # 1 second mono
prepared = prepare_audio(audio, 44100)
assert prepared.shape == (1, 2, 44100)

# Test 3: Normalization
normalized = normalize_audio(prepared, -6)
assert torch.max(torch.abs(normalized)) <= 0.501  # -6dB ≈ 0.5

# Test 4: Encoding short audio
latent = encode_audio_to_latent("short_test.wav", vae, chunked=False)
assert latent.shape[1] == 64

# Test 5: Chunked encoding long audio
latent = encode_audio_to_latent("long_test.wav", vae, chunked=True)
assert latent.shape[1] == 64
```

### 9.2 Integration Test

```python
# Test full pipeline
audio_path = "test_song.mp3"
output_path = "test_latent.pt"

# Encode
vae = load_vae()
latent = encode_audio_to_latent(audio_path, vae)

# Save
save_latent(latent, output_path)

# Load and verify
verify_latent(output_path)

# Check frame count
audio, sr = torchaudio.load(audio_path)
expected_frames = audio.shape[1] // 2048
assert abs(latent.shape[2] - expected_frames) <= 1
```

### 9.3 Validation Checklist

- [ ] Model loads successfully
- [ ] Handles mono, stereo, multi-channel audio
- [ ] Resamples correctly
- [ ] Normalizes audio
- [ ] Encodes short audio (< 30s) without chunking
- [ ] Encodes long audio (> 30s) with chunking
- [ ] Output shape is `[1, 64, T]`
- [ ] Output dtype is `torch.float32`
- [ ] Frame count matches expected (samples / 2048)
- [ ] Can save and reload latent
- [ ] Dataset can load the saved file

---

## 10. Function Signatures

Here's the complete API for `audio_processing.py`:

```python
# Model management
def load_vae(device: str = "cuda", cache_dir: str = "./pretrained") -> torch.jit.ScriptModule

# Audio preparation
def prepare_audio(
    audio: torch.Tensor, 
    in_sr: int, 
    target_sr: int = 44100,
    target_channels: int = 2,
    device: str = "cuda"
) -> torch.Tensor

def normalize_audio(
    audio: torch.Tensor, 
    target_dbfs: float = -6.0
) -> torch.Tensor

# Encoding
def encode_audio_to_latent(
    audio_path: str,
    vae: torch.jit.ScriptModule,
    device: str = "cuda",
    chunked: bool = True
) -> torch.Tensor

def encode_audio_chunked(
    audio: torch.Tensor,
    vae: torch.jit.ScriptModule,
    chunk_size: int = 128,
    overlap: int = 32
) -> torch.Tensor

# I/O
def save_latent(latent: torch.Tensor, output_path: str) -> None

def verify_latent(path: str) -> None
```

---

## 11. Key References

### From DATA_PREPROCESSING_GUIDE.md

- **Section 4**: Component 1: Audio Latent Generation (lines 227-435)
- **Section 10.1**: Verification and Debugging (lines 1172-1192)

### From Original Codebase

- `infer/infer_utils.py` lines 79-98: `prepare_audio()` function
- `infer/infer_utils.py` lines 43-51: `normalize_audio()` function  
- `infer/infer_utils.py` lines 154-211: `encode_audio_chunked()` function
- `infer/infer_utils.py` lines 258-275: VAE encoding in inference
- `dataset/dataset.py` lines 63-64: How training loads latents
- `config/diffrhythm-1b.json` line 10: `mel_dim=64` confirms latent dimension

---

## 12. Implementation Checklist

Before you start coding:

- [ ] Review this entire document
- [ ] Have access to HuggingFace for model download
- [ ] Have test audio files of various lengths (10s, 60s, 180s)
- [ ] Have GPU with sufficient VRAM (8GB+ recommended)
- [ ] Understand chunked encoding algorithm

During implementation:

- [ ] Implement `load_vae()` first, test model loads
- [ ] Implement `prepare_audio()`, test with various audio types
- [ ] Implement `normalize_audio()`, test normalization works
- [ ] Implement `encode_audio_to_latent()` without chunking first
- [ ] Test on short audio (< 30s)
- [ ] Implement `encode_audio_chunked()` carefully
- [ ] Test on long audio (> 60s)
- [ ] Implement `save_latent()` and `verify_latent()`
- [ ] Test full pipeline end-to-end

After implementation:

- [ ] Process a few sample files
- [ ] Verify output files with `verify_latent()`
- [ ] Test that dataset can load your files
- [ ] Profile memory usage
- [ ] Document any deviations from this plan

---

## Notes and Considerations

### Deterministic vs Stochastic Encoding

We use **mean only** (deterministic) for training data. The alternative is to sample:

```python
mean, scale = latent.chunk(2, dim=1)
latent = mean + scale * torch.randn_like(mean)  # Stochastic
```

Stick with deterministic (mean only) unless you have a specific reason to add noise.

### Chunk Size Selection

`chunk_size=128` and `overlap=32` are defaults from the reference implementation. You can adjust:

- **Smaller chunk_size**: Less memory, more chunks, possibly slower
- **Larger chunk_size**: More memory, fewer chunks, possibly faster
- **Larger overlap**: Smoother transitions, more computation
- **Smaller overlap**: Possible artifacts at boundaries, faster

Test if you change these values.

### CPU vs GPU

- **GPU**: 5-10x faster, but limited by VRAM
- **CPU**: Slower, but can handle any length audio

For preprocessing large datasets, GPU is highly recommended.

---

## Summary

The VAE latent generation is the foundation of the preprocessing pipeline. Key points:

1. **Model is black-box TorchScript** - we use the interface, not the architecture
2. **44.1kHz stereo is mandatory** - no flexibility here
3. **Chunked encoding is essential** for long audio (>30s)
4. **Output is mean only** (64 channels), not mean+scale (128 channels)
5. **Compression is 2048:1** - this is fixed and determines frame counts
6. **Overlap handling prevents artifacts** - implement carefully

Take your time implementing the chunked encoding algorithm. It's the most complex part but critical for quality and memory efficiency.
