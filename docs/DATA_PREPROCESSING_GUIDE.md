# DiffRhythm Data Preprocessing Guide for Full Finetuning

This document provides a comprehensive technical guide for preparing training data to finetune DiffRhythm. The training pipeline requires three preprocessed files per audio sample, and this guide explains exactly how to create them.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Why VAE Latents? (Proof)](#2-why-vae-latents-proof)
3. [Required Data Format](#3-required-data-format)
4. [Component 1: Audio Latent Generation](#4-component-1-audio-latent-generation)
5. [Component 2: Style Embedding Extraction](#5-component-2-style-embedding-extraction)
6. [Component 3: Lyrics Tokenization](#6-component-3-lyrics-tokenization)
7. [Creating the Training Manifest](#7-creating-the-training-manifest)
8. [Handling Instrumental Music](#8-handling-instrumental-music)
9. [Complete Preprocessing Script](#9-complete-preprocessing-script)
10. [Verification and Debugging](#10-verification-and-debugging)
11. [Technical Reference](#11-technical-reference)

---

## 1. Overview

### What You Need to Create

For each audio sample, you must generate **three `.pt` files**:

```
dataset/
├── latent/
│   └── {sample_id}.pt    # VAE-encoded audio representation
├── lrc/
│   └── {sample_id}.pt    # Tokenized lyrics with timestamps
├── style/
│   └── {sample_id}.pt    # MuQ style embedding
└── train.scp             # Manifest file listing all samples
```

### The Pipeline at a Glance

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Audio File    │     │   LRC File      │     │   Audio File    │
│   (MP3/WAV)     │     │   (Lyrics)      │     │   (MP3/WAV)     │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   VAE Encoder   │     │  G2P Tokenizer  │     │   MuQ Model     │
│   (Black-box)   │     │  (In repo)      │     │  (HuggingFace)  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  latent/*.pt    │     │   lrc/*.pt      │     │  style/*.pt     │
│  [1, 64, T]     │     │  {time, lrc}    │     │  [1, 512]       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Key Dependencies

```python
# External models (downloaded from HuggingFace)
- ASLP-lab/DiffRhythm-vae    # VAE for audio encoding
- OpenMuQ/MuQ-MuLan-large    # Style embedding model

# In-repo code
- g2p/g2p_generation.py      # Lyrics tokenization
- infer/infer_utils.py       # Audio processing utilities
```

---

## 2. Why VAE Latents? (Proof)

You might wonder: how do we know the training data must be VAE-encoded latents and not mel spectrograms or raw audio? Here's the definitive proof from the codebase.

### 2.1 Dimension Mismatch with Mel Spectrograms

The codebase contains a `MelSpec` class in `model/modules.py` (lines 53-94) that was inherited from the F5-TTS project this code is based on. However, **it's dead code that's never used**.

| Component | Dimension | Source |
|-----------|-----------|--------|
| `MelSpec` class default | `n_mel_channels=100` | `model/modules.py` line 59 |
| Model config | `mel_dim=64` | `config/diffrhythm-1b.json` line 10 |
| VAE encoder output | 128 (64 mean + 64 scale) | `infer/infer_utils.py` line 266 |
| VAE mean (used for training) | **64** | After `.chunk(2, dim=1)` |

**Key insight:** The model expects 64-dimensional input (`mel_dim=64`), which matches the VAE latent dimension, **not** the mel spectrogram dimension (100).

### 2.2 MelSpec is Dead Code

Searching the codebase reveals:

1. **`MelSpec(...)` is never instantiated** anywhere in the code
2. The `CFM` class references `self.mel_spec` in the `sample()` method (line 152), but **`self.mel_spec` is never initialized** in `__init__`
3. If you passed raw audio to `sample()`, it would crash with `AttributeError: 'CFM' object has no attribute 'mel_spec'`

```python
# model/cfm.py line 152 - This code path would crash!
if cond.ndim == 2:
    cond = self.mel_spec(cond)  # AttributeError: 'CFM' has no 'mel_spec'
```

### 2.3 Training Code Loads Pre-encoded Latents

The training loop in `model/trainer.py` (lines 294-295) directly loads latents:

```python
text_inputs = batch["lrc"]
mel_spec = batch["latent"].permute(0, 2, 1)  # Loaded directly, no encoding!
```

There is no VAE encoder call in the training pipeline. The data must already be encoded.

### 2.4 Inference Confirms Latent Space Operation

During inference (`infer/infer.py` line 73), the model output IS decoded through the VAE:

```python
output = decode_audio(latent, vae_model, chunked=chunked)
```

This confirms the model operates entirely in VAE latent space.

### 2.5 Summary

| Evidence | Conclusion |
|----------|------------|
| `mel_dim=64` ≠ `n_mel_channels=100` | Not mel spectrograms |
| `mel_dim=64` = VAE mean dimension | VAE latents confirmed |
| `MelSpec` never instantiated | Dead code from F5-TTS |
| `self.mel_spec` not in `__init__` | Would crash if called |
| No `vae.encode()` in training | Data must be pre-encoded |
| `vae.decode()` in inference | Confirms latent space |

**Conclusion:** Training data must be VAE-encoded latents with shape `[64, T]` or `[1, 64, T]`.

---

## 3. Required Data Format

### 3.1 The `train.scp` Manifest File

Each line follows this pipe-delimited format:

```
{utterance_id}|{lrc_path}|{latent_path}|{style_path}
```

**Example:**
```
song001|dataset/lrc/song001.pt|dataset/latent/song001.pt|dataset/style/song001.pt
song002|dataset/lrc/song002.pt|dataset/latent/song002.pt|dataset/style/song002.pt
```

**Relevant code:** `dataset/dataset.py` lines 55-56:
```python
def get_triple(self, item):
    utt, lrc_path, latent_path, style_path = item.split("|")
```

### 3.2 Latent File Format (`latent/*.pt`)

| Property | Value |
|----------|-------|
| Type | `torch.Tensor` |
| Shape | `[1, 64, T]` or `[64, T]` |
| Dtype | `torch.float32` (converted to fp16/bf16 during training) |
| T | Number of latent frames = `audio_samples / 2048` |

**Example:** A 95-second song at 44.1kHz:
- Audio samples: 95 × 44100 = 4,189,500
- Latent frames: 4,189,500 / 2048 ≈ 2045

**Relevant code:** `dataset/dataset.py` lines 63-64:
```python
latent = torch.load(latent_path, map_location='cpu')  # [b, d, t]
latent = latent.squeeze(0)  # Removes batch dim if present
```

### 3.3 LRC File Format (`lrc/*.pt`)

| Property | Value |
|----------|-------|
| Type | `dict` with keys `'time'` and `'lrc'` |
| `'time'` | `List[float]` - timestamps in seconds |
| `'lrc'` | `List[List[int]]` - token IDs for each line |

**Example structure:**
```python
{
    'time': [0.0, 5.2, 10.8, 15.3, ...],  # Start time of each lyric line
    'lrc': [
        [45, 28, 30, 19, ...],  # Token IDs for line 1
        [12, 55, 33, 47, ...],  # Token IDs for line 2
        ...
    ]
}
```

**Relevant code:** `dataset/dataset.py` lines 58-61:
```python
time_lrc = torch.load(lrc_path, map_location='cpu')
input_times = time_lrc['time']
input_lrcs = time_lrc['lrc']
lrc_with_time = list(zip(input_times, input_lrcs))
```

### 3.4 Style File Format (`style/*.pt`)

| Property | Value |
|----------|-------|
| Type | `torch.Tensor` |
| Shape | `[1, 512]` or `[512]` |
| Dtype | `torch.float32` (converted to fp16 during training) |

**Relevant code:** `dataset/dataset.py` lines 66-67:
```python
prompt = torch.load(style_path, map_location='cpu')  # [b, d]
prompt = prompt.squeeze(0)  # Removes batch dim if present
```

---

## 4. Component 1: Audio Latent Generation

### 4.1 The VAE Model

The VAE is a **pre-trained, black-box TorchScript model**. You cannot see its architecture, but you can use its interface.

**Download location:** HuggingFace `ASLP-lab/DiffRhythm-vae`

**Interface:**
```python
# Encoding: audio -> latent
vae.encode_export(audio)  # [batch, 2, samples] -> [batch, 128, T]

# Decoding: latent -> audio  
vae.decode_export(latent)  # [batch, 64, T] -> [batch, 2, samples]
```

**Note:** The encoder outputs 128 channels (64 for mean, 64 for scale) because it's a Variational Autoencoder. For training, you typically use just the mean (first 64 channels).

### 4.2 Audio Requirements

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sample rate | 44100 Hz | **Must be exactly 44.1kHz** |
| Channels | 2 (stereo) | Mono will be duplicated |
| Format | WAV, MP3, FLAC | torchaudio supports these |
| Normalization | -6 dB peak | Optional but recommended |

**Relevant code:** `infer/infer_utils.py` lines 79-98 (`prepare_audio` function)

### 4.3 Downsampling Ratio

The VAE compresses audio with a fixed ratio:

```
downsample_rate = 2048
```

This means:
- 2048 audio samples → 1 latent frame
- 1 second of audio → 44100/2048 ≈ 21.53 latent frames
- 95 seconds → ~2048 latent frames (matches `max_frames` in config)

### 4.4 Implementation

```python
import torch
import torchaudio
from huggingface_hub import hf_hub_download

def load_vae(device="cuda"):
    """Load the pre-trained VAE model."""
    vae_path = hf_hub_download(
        repo_id="ASLP-lab/DiffRhythm-vae",
        filename="vae_model.pt",
        cache_dir="./pretrained"
    )
    vae = torch.jit.load(vae_path, map_location="cpu").to(device)
    return vae

def prepare_audio(audio, in_sr, target_sr=44100, target_channels=2, device="cuda"):
    """
    Prepare audio for VAE encoding.
    
    Based on: infer/infer_utils.py lines 79-98
    """
    audio = audio.to(device)
    
    # Resample if necessary
    if in_sr != target_sr:
        audio = torchaudio.functional.resample(audio, in_sr, target_sr)
    
    # Handle channel conversion
    if audio.shape[0] == 1:
        # Mono to stereo: duplicate the channel
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        # More than stereo: take first two channels
        audio = audio[:2, :]
    
    # Add batch dimension: [2, samples] -> [1, 2, samples]
    if audio.dim() == 2:
        audio = audio.unsqueeze(0)
    
    return audio

def normalize_audio(audio, target_dbfs=-6):
    """
    Peak normalize audio to target dBFS.
    
    Based on: infer/infer_utils.py lines 43-51
    """
    max_amplitude = torch.max(torch.abs(audio))
    target_amplitude = 10.0 ** (target_dbfs / 20.0)
    scale_factor = target_amplitude / max_amplitude
    return audio * scale_factor

def encode_audio_to_latent(audio_path, vae, device="cuda", chunked=True):
    """
    Encode an audio file to VAE latent representation.
    
    Args:
        audio_path: Path to audio file (MP3, WAV, etc.)
        vae: Loaded VAE model
        device: torch device
        chunked: Use chunked encoding for long audio (recommended)
    
    Returns:
        latent: Tensor of shape [1, 64, T]
    """
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    
    # Prepare audio (resample, stereo, normalize)
    audio = prepare_audio(audio, sr, target_sr=44100, device=device)
    audio = normalize_audio(audio, target_dbfs=-6)
    
    # Encode through VAE
    with torch.no_grad():
        if chunked:
            # Use chunked encoding for long audio to avoid OOM
            # Based on: infer/infer_utils.py lines 154-211
            latent = encode_audio_chunked(audio, vae)
        else:
            latent = vae.encode_export(audio)  # [1, 128, T]
        
        # Split into mean and scale (VAE outputs both)
        mean, scale = latent.chunk(2, dim=1)  # Each [1, 64, T]
        
        # Use mean for training (deterministic)
        # Alternatively, sample: latent = mean + scale * torch.randn_like(mean)
        latent = mean
    
    return latent.cpu()

def encode_audio_chunked(audio, vae, chunk_size=128, overlap=32):
    """
    Chunked encoding for long audio files.
    
    Based on: infer/infer_utils.py lines 154-211
    """
    downsampling_ratio = 2048
    latent_dim = 128
    
    samples_per_latent = downsampling_ratio
    total_size = audio.shape[2]  # in samples
    batch_size = audio.shape[0]
    
    chunk_size_samples = chunk_size * samples_per_latent
    overlap_samples = overlap * samples_per_latent
    hop_size = chunk_size_samples - overlap_samples
    
    # Collect chunks
    chunks = []
    for i in range(0, total_size - chunk_size_samples + 1, hop_size):
        chunk = audio[:, :, i:i+chunk_size_samples]
        chunks.append(chunk)
    
    # Handle final chunk
    if len(chunks) == 0 or (len(chunks) > 0 and chunks[-1].shape[2] < chunk_size_samples):
        final_chunk = audio[:, :, -chunk_size_samples:]
        chunks.append(final_chunk)
    
    chunks = torch.stack(chunks)
    num_chunks = chunks.shape[0]
    
    # Output size in latent space
    y_size = total_size // samples_per_latent
    y_final = torch.zeros((batch_size, latent_dim, y_size), device=audio.device)
    
    # Encode each chunk
    for i in range(num_chunks):
        x_chunk = chunks[i]
        y_chunk = vae.encode_export(x_chunk)
        
        # Calculate position
        if i == num_chunks - 1:
            t_end = y_size
            t_start = t_end - y_chunk.shape[2]
        else:
            t_start = i * (chunk_size - overlap)
            t_end = t_start + chunk_size
        
        # Handle overlap (take center portion)
        ol = overlap // 2
        chunk_start = 0
        chunk_end = y_chunk.shape[2]
        
        if i > 0:
            t_start += ol
            chunk_start += ol
        if i < num_chunks - 1:
            t_end -= ol
            chunk_end -= ol
        
        y_final[:, :, t_start:t_end] = y_chunk[:, :, chunk_start:chunk_end]
    
    return y_final
```

### 4.5 Saving the Latent

```python
def save_latent(latent, output_path):
    """Save latent tensor to .pt file."""
    # latent should be [1, 64, T] or [64, T]
    torch.save(latent, output_path)
```

---

## 5. Component 2: Style Embedding Extraction

### 5.1 The MuQ Model

MuQ (Music-Query) is a multimodal model that encodes audio into a 512-dimensional style embedding. It can also encode text prompts to the same space.

**Model:** `OpenMuQ/MuQ-MuLan-large`  
**Library:** `muq` (install via `pip install muq`)

### 5.2 Audio Requirements for MuQ

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sample rate | 24000 Hz | **Different from VAE!** |
| Duration | ≥10 seconds | Model expects at least 10s |
| Channels | 1 (mono) | Internally converted |

**Relevant code:** `infer/infer_utils.py` lines 295-330 (`get_style_prompt` function)

### 5.3 Implementation

```python
from muq import MuQMuLan
import librosa
import torch

def load_muq(device="cuda"):
    """Load the MuQ model for style embedding extraction."""
    muq = MuQMuLan.from_pretrained(
        "OpenMuQ/MuQ-MuLan-large", 
        cache_dir="./pretrained"
    )
    return muq.to(device).eval()

def extract_style_embedding(audio_path, muq, device="cuda"):
    """
    Extract style embedding from audio using MuQ.
    
    Based on: infer/infer_utils.py lines 295-330
    
    Args:
        audio_path: Path to audio file
        muq: Loaded MuQ model
        device: torch device
    
    Returns:
        style_embedding: Tensor of shape [1, 512]
    """
    # Get audio duration
    duration = librosa.get_duration(path=audio_path)
    
    if duration < 10:
        raise ValueError(
            f"Audio file {audio_path} is too short ({duration:.2f}s). "
            "MuQ requires at least 10 seconds of audio."
        )
    
    # Extract 10 seconds from the middle of the song
    # This typically captures the most representative style
    mid_time = duration / 2
    start_time = mid_time - 5
    
    # Load audio at 24kHz (MuQ's expected sample rate)
    wav, _ = librosa.load(audio_path, sr=24000, offset=start_time, duration=10)
    
    # Convert to tensor
    wav = torch.tensor(wav).unsqueeze(0).to(device)  # [1, samples]
    
    # Extract embedding
    with torch.no_grad():
        style_embedding = muq(wavs=wav)  # [1, 512]
    
    return style_embedding.cpu()

def extract_style_from_text(text_prompt, muq, device="cuda"):
    """
    Extract style embedding from a text description.
    
    This allows training with text-based style prompts instead of audio.
    
    Args:
        text_prompt: Style description (e.g., "jazzy nightclub vibe")
        muq: Loaded MuQ model
        device: torch device
    
    Returns:
        style_embedding: Tensor of shape [1, 512]
    """
    with torch.no_grad():
        style_embedding = muq(texts=text_prompt)  # [1, 512]
    
    return style_embedding.cpu()
```

### 5.4 Saving the Style Embedding

```python
def save_style(style_embedding, output_path):
    """Save style embedding to .pt file."""
    # style_embedding should be [1, 512] or [512]
    torch.save(style_embedding, output_path)
```

---

## 6. Component 3: Lyrics Tokenization

### 6.1 The G2P (Grapheme-to-Phoneme) System

The lyrics tokenization uses a sophisticated G2P system that:
1. Detects language (Chinese, English, Korean, French, German)
2. Converts text to phonemes
3. Maps phonemes to token IDs

**Key files:**
- `g2p/g2p_generation.py` - Main entry point (`chn_eng_g2p` function)
- `g2p/g2p/__init__.py` - `PhonemeBpeTokenizer` class
- `g2p/g2p/vocab.json` - Vocabulary mapping (363 phoneme tokens)

### 6.2 LRC File Format

The input is a standard LRC (Lyric) file with timestamps:

```
[00:00.00]First line of lyrics
[00:05.23]Second line of lyrics
[00:10.45]Third line of lyrics
...
```

**Timestamp format:** `[MM:SS.xx]` where:
- `MM` = minutes (00-99)
- `SS` = seconds (00-59)
- `xx` = hundredths of a second (00-99)

### 6.3 Token Vocabulary

The vocabulary has 363 tokens (indices 0-362):

| Index | Token | Description |
|-------|-------|-------------|
| 0 | `,` | Comma |
| 1 | `.` | Period (used as line separator) |
| 2 | `?` | Question mark |
| 3 | `!` | Exclamation |
| 4 | `_` | Word separator |
| 5-74 | English phonemes | IPA symbols |
| 75-362 | Chinese phonemes | With tone markers (→↑↓↑↓) |

**Relevant file:** `g2p/g2p/vocab.json`

### 6.4 Implementation

```python
import sys
import os
import torch

# Add repo root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g2p.g2p_generation import chn_eng_g2p

def parse_lrc_file(lrc_path):
    """
    Parse an LRC file into (timestamp, text) pairs.
    
    Based on: infer/infer_utils.py lines 333-345
    
    Args:
        lrc_path: Path to .lrc file
    
    Returns:
        List of (timestamp_seconds, lyric_text) tuples
    """
    lyrics_with_time = []
    
    with open(lrc_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('['):
                continue
            
            try:
                # Extract timestamp [MM:SS.xx]
                time_str = line[1:9]  # "MM:SS.xx"
                text = line[10:].strip()  # Everything after "]"
                
                # Convert to seconds
                mins, secs = time_str.split(':')
                timestamp = int(mins) * 60 + float(secs)
                
                if text:  # Only add non-empty lines
                    lyrics_with_time.append((timestamp, text))
            except (ValueError, IndexError):
                # Skip malformed lines
                continue
    
    return lyrics_with_time

def tokenize_lyrics(lrc_path):
    """
    Convert LRC file to tokenized format for training.
    
    Args:
        lrc_path: Path to .lrc file
    
    Returns:
        dict with 'time' (list of floats) and 'lrc' (list of token lists)
    """
    lyrics_with_time = parse_lrc_file(lrc_path)
    
    times = []
    token_lists = []
    
    for timestamp, text in lyrics_with_time:
        # Convert text to phoneme tokens using G2P
        phonemes, tokens = chn_eng_g2p(text)
        
        times.append(timestamp)
        token_lists.append(tokens)
    
    return {
        'time': times,
        'lrc': token_lists
    }

def save_lrc_tokens(lrc_data, output_path):
    """Save tokenized lyrics to .pt file."""
    torch.save(lrc_data, output_path)
```

### 6.5 How the Dataset Uses LRC Data

Understanding how the dataset processes LRC data is critical. Here's what happens in `dataset/dataset.py`:

```python
# Lines 75-76: Filter lyrics by time window
lrc_with_time = [(time_start - start_time, line) 
                 for (time_start, line) in lrc_with_time 
                 if (time_start - start_time) >= 0]  # Adjust for random crop

# Lines 86-105: Place tokens in frame-aligned tensor
lrc = torch.zeros((self.max_frames,), dtype=torch.long)

for time_start, line in lrc_with_time:
    tokens = [token if token != period_token_id else comma_token_id 
              for token in line] + [period_token_id]  # Add period at end
    tokens = torch.tensor(tokens, dtype=torch.long)
    
    # Convert time to frame position
    frame_start = int(time_start * sampling_rate / downsample_rate)
    
    # Place tokens starting at frame_start
    lrc[frame_start:frame_start + len(tokens)] = tokens
```

**Key insight:** The tokens are placed at specific frame positions based on timestamps. The rest of the tensor remains zeros (padding token).

---

## 7. Creating the Training Manifest

### 7.1 The `train.scp` Format

```python
def create_train_scp(samples, output_path):
    """
    Create the training manifest file.
    
    Args:
        samples: List of (sample_id, lrc_path, latent_path, style_path) tuples
        output_path: Path to write train.scp
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample_id, lrc_path, latent_path, style_path in samples:
            line = f"{sample_id}|{lrc_path}|{latent_path}|{style_path}\n"
            f.write(line)
```

### 7.2 Example

```
song001|dataset/lrc/song001.pt|dataset/latent/song001.pt|dataset/style/song001.pt
song002|dataset/lrc/song002.pt|dataset/latent/song002.pt|dataset/style/song002.pt
```

---

## 8. Handling Instrumental Music

### 8.1 The Problem

The current dataset code **crashes** on instrumental music (no lyrics):

```python
# dataset/dataset.py lines 78-81
if len(lrc_with_time) >= 1:
    latent_end_time = lrc_with_time[-1][0]
else:
    raise  # <-- Crashes!
```

### 8.2 Workaround: Dummy LRC Entry

For instrumental tracks, create an LRC file with a single entry at the end:

```python
def create_instrumental_lrc(duration_seconds):
    """
    Create a dummy LRC data structure for instrumental music.
    
    Args:
        duration_seconds: Total length of the audio in seconds
    
    Returns:
        dict with minimal 'time' and 'lrc' entries
    """
    # Place a single timestamp near the end
    # This tells the dataset how long the audio is
    end_time = duration_seconds - 1.0  # 1 second before end
    
    return {
        'time': [end_time],
        'lrc': [[2]]  # Just a period token (index 2 in vocab)
    }
```

### 8.3 Better Solution: Modify the Dataset

To properly support instrumentals, modify `dataset/dataset.py`:

```python
# Replace lines 78-81 with:
if len(lrc_with_time) >= 1:
    latent_end_time = lrc_with_time[-1][0]
else:
    # For instrumental: use full latent length
    latent_end_time = latent.shape[-1] * self.downsample_rate / self.sampling_rate
```

---

## 9. Complete Preprocessing Script

Here's a complete, production-ready preprocessing script:

```python
#!/usr/bin/env python3
"""
DiffRhythm Data Preprocessing Script

Usage:
    python preprocess_data.py --input-dir /path/to/songs --output-dir dataset

Input directory structure:
    /path/to/songs/
    ├── song001.mp3
    ├── song001.lrc
    ├── song002.wav
    ├── song002.lrc
    └── ...

Output:
    dataset/
    ├── latent/*.pt
    ├── lrc/*.pt
    ├── style/*.pt
    └── train.scp
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torchaudio
import librosa

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import hf_hub_download
from muq import MuQMuLan
from g2p.g2p_generation import chn_eng_g2p


class DiffRhythmPreprocessor:
    """
    Preprocessor for creating DiffRhythm training data.
    """
    
    def __init__(self, device="cuda", cache_dir="./pretrained"):
        self.device = device
        self.cache_dir = cache_dir
        
        # Audio parameters (must match training config)
        self.vae_sample_rate = 44100
        self.muq_sample_rate = 24000
        self.downsample_rate = 2048
        
        # Load models
        print("Loading VAE model...")
        self.vae = self._load_vae()
        
        print("Loading MuQ model...")
        self.muq = self._load_muq()
        
        print("Models loaded successfully!")
    
    def _load_vae(self):
        """Load the VAE model from HuggingFace."""
        vae_path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-vae",
            filename="vae_model.pt",
            cache_dir=self.cache_dir
        )
        vae = torch.jit.load(vae_path, map_location="cpu").to(self.device)
        return vae
    
    def _load_muq(self):
        """Load the MuQ model from HuggingFace."""
        muq = MuQMuLan.from_pretrained(
            "OpenMuQ/MuQ-MuLan-large",
            cache_dir=self.cache_dir
        )
        return muq.to(self.device).eval()
    
    def process_audio_to_latent(self, audio_path):
        """
        Encode audio to VAE latent representation.
        
        Returns:
            Tensor of shape [1, 64, T]
        """
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Resample to 44.1kHz if needed
        if sr != self.vae_sample_rate:
            audio = torchaudio.functional.resample(
                audio, sr, self.vae_sample_rate
            )
        
        # Convert to stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2, :]
        
        # Add batch dimension and move to device
        audio = audio.unsqueeze(0).to(self.device)
        
        # Normalize to -6dB peak
        max_amp = torch.max(torch.abs(audio))
        audio = audio * (0.5 / max_amp)  # -6dB ≈ 0.5 amplitude
        
        # Encode through VAE (chunked for long audio)
        with torch.no_grad():
            latent = self._encode_chunked(audio)
            mean, scale = latent.chunk(2, dim=1)
            latent = mean  # Use mean for deterministic encoding
        
        return latent.cpu()
    
    def _encode_chunked(self, audio, chunk_size=128, overlap=32):
        """Chunked VAE encoding for long audio."""
        samples_per_frame = self.downsample_rate
        total_samples = audio.shape[2]
        
        chunk_samples = chunk_size * samples_per_frame
        overlap_samples = overlap * samples_per_frame
        hop_samples = chunk_samples - overlap_samples
        
        # Pad audio if necessary
        if total_samples < chunk_samples:
            pad_size = chunk_samples - total_samples
            audio = torch.nn.functional.pad(audio, (0, pad_size))
            total_samples = chunk_samples
        
        # Output dimensions
        total_frames = total_samples // samples_per_frame
        output = torch.zeros(1, 128, total_frames, device=self.device)
        
        # Process chunks
        pos = 0
        frame_pos = 0
        
        while pos < total_samples:
            end_pos = min(pos + chunk_samples, total_samples)
            
            # Extract chunk
            if end_pos - pos < chunk_samples:
                # Last chunk - take from end
                chunk = audio[:, :, -chunk_samples:]
                chunk_latent = self.vae.encode_export(chunk)
                output[:, :, -chunk_latent.shape[2]:] = chunk_latent
                break
            else:
                chunk = audio[:, :, pos:end_pos]
                chunk_latent = self.vae.encode_export(chunk)
                
                # Handle overlap
                if pos == 0:
                    # First chunk
                    valid_frames = chunk_size - overlap // 2
                    output[:, :, :valid_frames] = chunk_latent[:, :, :valid_frames]
                    frame_pos = valid_frames
                else:
                    # Middle chunks
                    start_offset = overlap // 2
                    valid_frames = chunk_size - overlap
                    output[:, :, frame_pos:frame_pos + valid_frames] = \
                        chunk_latent[:, :, start_offset:start_offset + valid_frames]
                    frame_pos += valid_frames
            
            pos += hop_samples
        
        return output[:, :, :total_frames]
    
    def process_audio_to_style(self, audio_path):
        """
        Extract style embedding using MuQ.
        
        Returns:
            Tensor of shape [1, 512]
        """
        # Get duration
        duration = librosa.get_duration(path=audio_path)
        
        # Extract 10 seconds from middle
        if duration < 10:
            print(f"Warning: {audio_path} is short ({duration:.1f}s), using full audio")
            start_time = 0
            dur = duration
        else:
            mid = duration / 2
            start_time = mid - 5
            dur = 10
        
        # Load at MuQ's expected sample rate
        wav, _ = librosa.load(
            audio_path, 
            sr=self.muq_sample_rate, 
            offset=start_time, 
            duration=dur
        )
        
        # Convert to tensor
        wav = torch.tensor(wav).unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            style = self.muq(wavs=wav)
        
        return style.cpu()
    
    def process_lrc(self, lrc_path, audio_duration=None):
        """
        Tokenize LRC file.
        
        Args:
            lrc_path: Path to .lrc file (or None for instrumental)
            audio_duration: Duration in seconds (required for instrumental)
        
        Returns:
            dict with 'time' and 'lrc' keys
        """
        if lrc_path is None or not os.path.exists(lrc_path):
            # Instrumental - create dummy entry
            if audio_duration is None:
                raise ValueError("audio_duration required for instrumental")
            return {
                'time': [audio_duration - 1.0],
                'lrc': [[2]]  # Period token
            }
        
        # Parse LRC file
        lyrics = []
        with open(lrc_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith('['):
                    continue
                try:
                    time_str = line[1:9]
                    text = line[10:].strip()
                    mins, secs = time_str.split(':')
                    timestamp = int(mins) * 60 + float(secs)
                    if text:
                        lyrics.append((timestamp, text))
                except:
                    continue
        
        # Tokenize
        times = []
        tokens = []
        for timestamp, text in lyrics:
            _, token_ids = chn_eng_g2p(text)
            times.append(timestamp)
            tokens.append(token_ids)
        
        return {'time': times, 'lrc': tokens}
    
    def process_sample(self, audio_path, lrc_path, output_dir, sample_id):
        """
        Process a single sample.
        
        Args:
            audio_path: Path to audio file
            lrc_path: Path to LRC file (or None for instrumental)
            output_dir: Output directory
            sample_id: Unique identifier for this sample
        
        Returns:
            Manifest line string
        """
        # Create output directories
        os.makedirs(f"{output_dir}/latent", exist_ok=True)
        os.makedirs(f"{output_dir}/lrc", exist_ok=True)
        os.makedirs(f"{output_dir}/style", exist_ok=True)
        
        # Process latent
        latent = self.process_audio_to_latent(audio_path)
        latent_path = f"{output_dir}/latent/{sample_id}.pt"
        torch.save(latent, latent_path)
        
        # Process style
        style = self.process_audio_to_style(audio_path)
        style_path = f"{output_dir}/style/{sample_id}.pt"
        torch.save(style, style_path)
        
        # Process LRC
        audio_duration = librosa.get_duration(path=audio_path)
        lrc_data = self.process_lrc(lrc_path, audio_duration)
        lrc_output_path = f"{output_dir}/lrc/{sample_id}.pt"
        torch.save(lrc_data, lrc_output_path)
        
        # Return manifest line
        return f"{sample_id}|{lrc_output_path}|{latent_path}|{style_path}"


def find_audio_lrc_pairs(input_dir):
    """
    Find matching audio and LRC file pairs in a directory.
    
    Returns:
        List of (audio_path, lrc_path, sample_id) tuples
    """
    input_dir = Path(input_dir)
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
    
    pairs = []
    for audio_file in input_dir.iterdir():
        if audio_file.suffix.lower() in audio_extensions:
            sample_id = audio_file.stem
            lrc_file = audio_file.with_suffix('.lrc')
            
            # LRC is optional (for instrumental)
            lrc_path = str(lrc_file) if lrc_file.exists() else None
            
            pairs.append((str(audio_file), lrc_path, sample_id))
    
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess audio data for DiffRhythm training"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing audio and LRC files"
    )
    parser.add_argument(
        "--output-dir", type=str, default="dataset",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="./pretrained",
        help="Directory to cache downloaded models"
    )
    args = parser.parse_args()
    
    # Find audio files
    pairs = find_audio_lrc_pairs(args.input_dir)
    print(f"Found {len(pairs)} audio files")
    
    if len(pairs) == 0:
        print("No audio files found!")
        return
    
    # Initialize preprocessor
    preprocessor = DiffRhythmPreprocessor(
        device=args.device,
        cache_dir=args.cache_dir
    )
    
    # Process all files
    manifest_lines = []
    for audio_path, lrc_path, sample_id in tqdm(pairs, desc="Processing"):
        try:
            line = preprocessor.process_sample(
                audio_path, lrc_path, args.output_dir, sample_id
            )
            manifest_lines.append(line)
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            continue
    
    # Write manifest
    manifest_path = f"{args.output_dir}/train.scp"
    with open(manifest_path, 'w') as f:
        f.write('\n'.join(manifest_lines))
    
    print(f"\nDone! Processed {len(manifest_lines)} samples")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
```

---

## 10. Verification and Debugging

### 10.1 Verify Latent Files

```python
def verify_latent(path):
    """Verify a latent file is correctly formatted."""
    latent = torch.load(path, map_location='cpu')
    
    print(f"Shape: {latent.shape}")
    print(f"Dtype: {latent.dtype}")
    print(f"Min: {latent.min():.4f}, Max: {latent.max():.4f}")
    
    # Expected: [1, 64, T] or [64, T]
    if latent.dim() == 3:
        assert latent.shape[0] == 1, "Batch size should be 1"
        assert latent.shape[1] == 64, "Latent dim should be 64"
    elif latent.dim() == 2:
        assert latent.shape[0] == 64, "Latent dim should be 64"
    else:
        raise ValueError(f"Unexpected dimensions: {latent.dim()}")
    
    print("✓ Latent file is valid")
```

### 10.2 Verify LRC Files

```python
def verify_lrc(path):
    """Verify an LRC file is correctly formatted."""
    lrc = torch.load(path, map_location='cpu')
    
    assert isinstance(lrc, dict), "LRC should be a dict"
    assert 'time' in lrc, "Missing 'time' key"
    assert 'lrc' in lrc, "Missing 'lrc' key"
    
    times = lrc['time']
    tokens = lrc['lrc']
    
    assert len(times) == len(tokens), "Mismatched lengths"
    assert all(isinstance(t, (int, float)) for t in times), "Times should be numbers"
    assert all(isinstance(t, list) for t in tokens), "Tokens should be lists"
    
    print(f"Number of lines: {len(times)}")
    print(f"Time range: {min(times):.2f}s - {max(times):.2f}s")
    print(f"Total tokens: {sum(len(t) for t in tokens)}")
    print("✓ LRC file is valid")
```

### 10.3 Verify Style Files

```python
def verify_style(path):
    """Verify a style file is correctly formatted."""
    style = torch.load(path, map_location='cpu')
    
    print(f"Shape: {style.shape}")
    print(f"Dtype: {style.dtype}")
    
    # Expected: [1, 512] or [512]
    if style.dim() == 2:
        assert style.shape == (1, 512), f"Expected [1, 512], got {style.shape}"
    elif style.dim() == 1:
        assert style.shape == (512,), f"Expected [512], got {style.shape}"
    else:
        raise ValueError(f"Unexpected dimensions: {style.dim()}")
    
    print("✓ Style file is valid")
```

### 10.4 Test Dataset Loading

```python
def test_dataset_loading(scp_path, max_frames=2048):
    """Test that the dataset can load your preprocessed data."""
    import sys
    sys.path.insert(0, '.')
    from dataset.dataset import DiffusionDataset
    
    dataset = DiffusionDataset(
        scp_path,
        max_frames=max_frames,
        min_frames=512
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Try loading a sample
    sample = dataset[0]
    
    print(f"Prompt shape: {sample['prompt'].shape}")
    print(f"LRC shape: {sample['lrc'].shape}")
    print(f"Latent shape: {sample['latent'].shape}")
    print(f"Start time: {sample['start_time']}")
    
    print("✓ Dataset loads successfully!")
```

---

## 11. Technical Reference

### 11.1 Key Constants

| Constant | Value | Used In |
|----------|-------|---------|
| `sampling_rate` | 44100 | VAE, Dataset |
| `downsample_rate` | 2048 | VAE compression ratio |
| `max_frames` | 2048 or 6144 | Maximum latent length |
| `muq_sample_rate` | 24000 | MuQ style extraction |
| `latent_dim` | 64 | VAE latent channels |
| `style_dim` | 512 | MuQ embedding size |
| `vocab_size` | 363 | G2P phoneme tokens |

### 11.2 Relevant Source Files

| File | Purpose |
|------|---------|
| `dataset/dataset.py` | Dataset class, shows expected formats |
| `infer/infer_utils.py` | Audio processing, VAE/MuQ usage |
| `g2p/g2p_generation.py` | Lyrics tokenization |
| `g2p/g2p/__init__.py` | PhonemeBpeTokenizer class |
| `g2p/g2p/vocab.json` | Token vocabulary |
| `train/train.py` | Training entry point |
| `model/trainer.py` | Training loop |
| `config/default.ini` | Training configuration |

### 11.3 Memory Considerations

| Audio Length | Latent Frames | Approx. VRAM for Encoding |
|--------------|---------------|---------------------------|
| 30 seconds | ~645 | ~2 GB |
| 95 seconds | ~2048 | ~4 GB |
| 285 seconds | ~6144 | ~8 GB |

Use chunked encoding for audio longer than 30 seconds to avoid OOM errors.

---

## Summary

To prepare data for DiffRhythm finetuning:

1. **Download models** from HuggingFace (VAE and MuQ)
2. **Encode audio** through VAE → `latent/*.pt` files
3. **Extract style** via MuQ → `style/*.pt` files  
4. **Tokenize lyrics** with G2P → `lrc/*.pt` files
5. **Create manifest** listing all samples → `train.scp`
6. **Run training** with `accelerate launch train/train.py`

The preprocessing script in Section 8 provides a complete, ready-to-use solution.
