"""
VAE Latent Generation - Audio Processing Module

This module handles encoding audio files into VAE latent representations for DiffRhythm training.
The VAE compresses audio at a fixed 2048:1 ratio into 64-dimensional latent space.

Key specifications:
- Input: Audio files (MP3, WAV, FLAC, etc.) at any sample rate
- Output: Latent tensors [1, 64, T] saved as .pt files
- Sample rate: Must be 44.1kHz (required by VAE)
- Channels: Must be stereo (2 channels, required by VAE)
- Compression: 2048 audio samples → 1 latent frame

References:
- docs/DATA_PREPROCESSING_GUIDE.md (Section 4)
- preprocess/VAE_LATENT_GENERATION.md
- infer/infer_utils.py (original inference code)
"""

import os
import torch
import torchaudio
from pathlib import Path
from typing import Optional, Tuple
from huggingface_hub import hf_hub_download


# =============================================================================
# Constants (from training config and VAE specs)
# =============================================================================

VAE_SAMPLE_RATE = 44100      # Required by VAE model
DOWNSAMPLE_RATE = 2048       # Audio samples per latent frame
TARGET_CHANNELS = 2          # Stereo required by VAE
LATENT_DIM = 128            # VAE outputs 128 (64 mean + 64 scale)
OUTPUT_LATENT_DIM = 64      # We save only the mean for training


# =============================================================================
# Model Loading
# =============================================================================

def load_vae(device: str = "cuda", cache_dir: str = "./pretrained"):
    """
    Load the pre-trained VAE model from HuggingFace.
    
    The VAE is a TorchScript black-box model that encodes stereo audio at 44.1kHz
    into a 64-dimensional latent space with 2048:1 compression.
    
    Args:
        device: torch device ('cuda' or 'cpu')
        cache_dir: Directory to cache downloaded models
    
    Returns:
        vae: Loaded TorchScript VAE model in eval mode
    
    Reference:
        - infer/infer_utils.py lines 244-249 (original implementation)
        - HuggingFace repo: ASLP-lab/DiffRhythm-vae
    """
    # Download from HuggingFace Hub (same as inference code)
    vae_ckpt_path = hf_hub_download(
        repo_id="ASLP-lab/DiffRhythm-vae",
        filename="vae_model.pt",
        cache_dir=cache_dir,
    )
    
    # Load TorchScript model
    vae = torch.jit.load(vae_ckpt_path, map_location="cpu").to(device)
    vae.eval()  # Always set to eval mode for inference
    
    return vae


# =============================================================================
# Audio Preparation Functions
# =============================================================================

def normalize_audio(audio: torch.Tensor, target_dbfs: float = -6.0) -> torch.Tensor:
    """
    Peak normalize audio to target dBFS level.
    
    This ensures consistent volume levels across the dataset and prevents clipping.
    -6 dBFS (≈0.5 amplitude) provides headroom and is the standard used in inference.
    
    Args:
        audio: Audio tensor of any shape
        target_dbfs: Target peak level in dBFS (default: -6.0)
    
    Returns:
        normalized_audio: Peak-normalized audio tensor
    
    Reference:
        - COPIED FROM: infer/infer_utils.py lines 43-51 (unchanged)
    """
    max_amplitude = torch.max(torch.abs(audio))
    target_amplitude = 10.0 ** (target_dbfs / 20.0)
    scale_factor = target_amplitude / max_amplitude
    normalized_audio = audio * scale_factor
    return normalized_audio


def set_audio_channels(audio: torch.Tensor, target_channels: int) -> torch.Tensor:
    """
    Convert audio to target number of channels.
    
    For VAE encoding, we always need stereo (2 channels):
    - Mono → Stereo: Duplicate the channel
    - Multi-channel → Stereo: Take first 2 channels
    
    Args:
        audio: Audio tensor [batch, channels, samples]
        target_channels: Target number of channels (2 for VAE)
    
    Returns:
        audio: Audio with target number of channels
    
    Reference:
        - COPIED FROM: infer/infer_utils.py lines 53-63 (unchanged)
    """
    if target_channels == 1:
        # Convert to mono
        audio = audio.mean(1, keepdim=True)
    elif target_channels == 2:
        # Convert to stereo
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
    return audio


def prepare_audio(
    audio: torch.Tensor,
    in_sr: int,
    target_sr: int = VAE_SAMPLE_RATE,
    target_channels: int = TARGET_CHANNELS,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Prepare audio for VAE encoding: resample, convert to stereo, add batch dim.
    
    This is a simplified version of infer/infer_utils.py prepare_audio() without
    the PadCrop operation (not needed for preprocessing).
    
    Args:
        audio: Input audio tensor [channels, samples]
        in_sr: Input sample rate
        target_sr: Target sample rate (must be 44100 for VAE)
        target_channels: Target number of channels (must be 2 for VAE)
        device: torch device
    
    Returns:
        audio: Prepared audio [1, 2, samples] ready for VAE encoding
    
    Reference:
        - ADAPTED FROM: infer/infer_utils.py lines 79-98
        - Changes: Removed PadCrop and target_length (not needed for preprocessing)
    """
    audio = audio.to(device)
    
    # Resample if necessary
    if in_sr != target_sr:
        resample_tf = torchaudio.transforms.Resample(in_sr, target_sr).to(device)
        audio = resample_tf(audio)
    
    # Add batch dimension if needed
    if audio.dim() == 1:
        # [samples] → [1, 1, samples]
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        # [channels, samples] → [1, channels, samples]
        audio = audio.unsqueeze(0)
    
    # Convert to target channels (stereo for VAE)
    audio = set_audio_channels(audio, target_channels)
    
    return audio


# =============================================================================
# VAE Encoding Functions
# =============================================================================

def encode_audio(
    audio: torch.Tensor,
    vae_model,
    chunked: bool = False,
    overlap: int = 32,
    chunk_size: int = 128
) -> torch.Tensor:
    """
    Encode audio through VAE, with optional chunked processing for long audio.
    
    The VAE outputs [batch, 128, T] where the 128 channels are:
    - Channels 0-63: Mean (μ) of latent distribution
    - Channels 64-127: Scale (σ) of latent distribution
    
    Chunked encoding is recommended for audio > 30 seconds to avoid OOM errors.
    
    Args:
        audio: Prepared audio [batch, 2, samples]
        vae_model: Loaded VAE model
        chunked: Use chunked encoding (recommended for long audio)
        overlap: Overlap in latent frames (default: 32)
        chunk_size: Chunk size in latent frames (default: 128)
    
    Returns:
        latent: Encoded latent [batch, 128, T] (includes both mean and scale)
    
    Reference:
        - COPIED FROM: infer/infer_utils.py lines 154-211 (unchanged)
        - This is the core chunked encoding algorithm with overlap handling
    """
    downsampling_ratio = DOWNSAMPLE_RATE
    latent_dim = LATENT_DIM
    
    if not chunked:
        # Default behavior: encode the entire audio in parallel
        return vae_model.encode_export(audio)
    else:
        # CHUNKED ENCODING
        # samples_per_latent is just the downsampling ratio
        samples_per_latent = downsampling_ratio
        total_size = audio.shape[2]  # in samples
        batch_size = audio.shape[0]
        chunk_size *= samples_per_latent  # converting metric in latents to samples
        overlap *= samples_per_latent  # converting metric in latents to samples
        hop_size = chunk_size - overlap
        
        # Collect chunks
        chunks = []
        for i in range(0, total_size - chunk_size + 1, hop_size):
            chunk = audio[:, :, i:i+chunk_size]
            chunks.append(chunk)
        
        if i + chunk_size != total_size:
            # Final chunk - align to end
            chunk = audio[:, :, -chunk_size:]
            chunks.append(chunk)
        
        chunks = torch.stack(chunks)
        num_chunks = chunks.shape[0]
        
        # Calculate output dimensions
        # Note: y_size might be a different value from the latent length used in diffusion training
        # because we can encode audio of varying lengths
        y_size = total_size // samples_per_latent
        
        # Create empty latent, populate it with chunks as we encode them
        y_final = torch.zeros((batch_size, latent_dim, y_size)).to(audio.device)
        
        for i in range(num_chunks):
            x_chunk = chunks[i, :]
            # Encode the chunk
            y_chunk = vae_model.encode_export(x_chunk)
            
            # Figure out where to put the latent along the time domain
            if i == num_chunks - 1:
                # Final chunk always goes at the end
                t_end = y_size
                t_start = t_end - y_chunk.shape[2]
            else:
                t_start = i * hop_size // samples_per_latent
                t_end = t_start + chunk_size // samples_per_latent
            
            # Remove the edges of the overlaps to prevent discontinuities
            ol = overlap // samples_per_latent // 2
            chunk_start = 0
            chunk_end = y_chunk.shape[2]
            
            if i > 0:
                # No overlap for the start of the first chunk
                t_start += ol
                chunk_start += ol
            if i < num_chunks - 1:
                # No overlap for the end of the last chunk
                t_end -= ol
                chunk_end -= ol
            
            # Paste the chunked latent into our y_final output
            y_final[:, :, t_start:t_end] = y_chunk[:, :, chunk_start:chunk_end]
        
        return y_final


# =============================================================================
# Main Processing Function
# =============================================================================

def encode_audio_to_latent(
    audio_path: str,
    vae_model,
    device: str = "cuda",
    chunked: bool = True,
    normalize: bool = True,
    target_dbfs: float = -6.0
) -> torch.Tensor:
    """
    Complete pipeline: Load audio file → Prepare → Encode → Extract mean.
    
    This is the main function for preprocessing. It handles:
    1. Loading audio from file
    2. Resampling to 44.1kHz
    3. Converting to stereo
    4. Normalizing volume (optional)
    5. Encoding through VAE
    6. Extracting mean (discarding scale) for deterministic training
    
    Args:
        audio_path: Path to audio file (MP3, WAV, FLAC, etc.)
        vae_model: Loaded VAE model
        device: torch device
        chunked: Use chunked encoding (recommended for audio > 30s)
        normalize: Apply peak normalization
        target_dbfs: Target peak level in dBFS (default: -6.0)
    
    Returns:
        latent: Tensor [1, 64, T] containing the mean of the VAE latent (on CPU)
    
    Raises:
        RuntimeError: If audio loading or encoding fails
        ValueError: If audio is empty or too short
    
    Example:
        >>> vae = load_vae(device="cuda")
        >>> latent = encode_audio_to_latent("song.mp3", vae)
        >>> print(latent.shape)  # [1, 64, 2045] for ~95 second song
    """
    # Step 1: Load audio
    try:
        audio, sr = torchaudio.load(audio_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio {audio_path}: {e}")
    
    # Validation: Check audio is not empty
    if audio.shape[-1] == 0:
        raise ValueError(f"Audio file {audio_path} is empty (0 samples)")
    
    # Validation: Check minimum length (at least 1 second)
    if audio.shape[-1] < sr:
        raise ValueError(
            f"Audio file {audio_path} too short: {audio.shape[-1]/sr:.2f}s "
            f"(minimum 1 second required)"
        )
    
    # Step 2: Prepare audio (resample, stereo, batch dim)
    try:
        audio = prepare_audio(audio, in_sr=sr, target_sr=VAE_SAMPLE_RATE, device=device)
    except Exception as e:
        raise RuntimeError(f"Failed to prepare audio {audio_path}: {e}")
    
    # Step 3: Normalize audio (optional but recommended)
    if normalize:
        audio = normalize_audio(audio, target_dbfs=target_dbfs)
    
    # Step 4: Encode through VAE
    try:
        with torch.no_grad():
            latent = encode_audio(audio, vae_model, chunked=chunked)
            
            # Step 5: Extract mean (discard scale)
            # VAE outputs [1, 128, T] with channels 0-63 = mean, 64-127 = scale
            # For training, we use deterministic encoding (mean only)
            mean, scale = latent.chunk(2, dim=1)  # Each [1, 64, T]
            latent = mean
            
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            f"GPU OOM encoding {audio_path}. Try: "
            f"(1) chunked=True, (2) Smaller chunk_size, (3) device='cpu'"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to encode {audio_path}: {e}")
    
    # Step 6: Move to CPU for saving
    return latent.cpu()


# =============================================================================
# I/O and Validation Functions
# =============================================================================

def save_latent(latent: torch.Tensor, output_path: str) -> None:
    """
    Save latent tensor to disk as .pt file.
    
    The dataset expects either [1, 64, T] or [64, T] (batch dim is optional).
    We save with batch dimension for consistency.
    
    Args:
        latent: Latent tensor [1, 64, T] or [64, T]
        output_path: Path to save .pt file
    
    Reference:
        - dataset/dataset.py lines 63-64 shows how training loads latents
    """
    # Ensure output directory exists (only if path contains a directory)
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create if there's actually a directory path
        os.makedirs(output_dir, exist_ok=True)
    
    # Verify shape before saving
    if latent.dim() == 3:
        assert latent.shape[1] == OUTPUT_LATENT_DIM, \
            f"Expected latent dim 64, got {latent.shape[1]}"
    elif latent.dim() == 2:
        assert latent.shape[0] == OUTPUT_LATENT_DIM, \
            f"Expected latent dim 64, got {latent.shape[0]}"
    else:
        raise ValueError(f"Invalid latent dimensions: {latent.shape}")
    
    # Save to disk
    torch.save(latent, output_path)


def verify_latent(path: str, verbose: bool = True) -> bool:
    """
    Verify that a saved latent file is correctly formatted.
    
    Checks:
    - File can be loaded
    - Shape is [1, 64, T] or [64, T]
    - Dtype is float32
    - Values are reasonable (not NaN/Inf)
    
    Args:
        path: Path to .pt file
        verbose: Print detailed information
    
    Returns:
        True if valid, raises exception if invalid
    
    Reference:
        - docs/DATA_PREPROCESSING_GUIDE.md section 10.1
    """
    try:
        latent = torch.load(path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to load latent from {path}: {e}")
    
    # Check dimensions
    if latent.dim() == 3:
        if latent.shape[0] != 1:
            raise ValueError(f"Batch size should be 1, got {latent.shape[0]}")
        if latent.shape[1] != OUTPUT_LATENT_DIM:
            raise ValueError(
                f"Expected latent dim {OUTPUT_LATENT_DIM}, got {latent.shape[1]}"
            )
    elif latent.dim() == 2:
        if latent.shape[0] != OUTPUT_LATENT_DIM:
            raise ValueError(
                f"Expected latent dim {OUTPUT_LATENT_DIM}, got {latent.shape[0]}"
            )
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {latent.dim()}D")
    
    # Check for NaN/Inf
    if torch.isnan(latent).any():
        raise ValueError(f"Latent contains NaN values")
    if torch.isinf(latent).any():
        raise ValueError(f"Latent contains Inf values")
    
    if verbose:
        print(f"✓ Latent file valid: {path}")
        print(f"  Shape: {latent.shape}")
        print(f"  Dtype: {latent.dtype}")
        print(f"  Range: [{latent.min():.4f}, {latent.max():.4f}]")
        print(f"  Mean: {latent.mean():.4f}, Std: {latent.std():.4f}")
    
    return True


def verify_latent_frame_count(
    audio_path: str,
    latent_path: str,
    tolerance: int = 1
) -> bool:
    """
    Verify that the latent frame count matches the expected count from audio length.
    
    Formula: expected_frames = audio_samples / DOWNSAMPLE_RATE
    
    Args:
        audio_path: Path to original audio file
        latent_path: Path to latent .pt file
        tolerance: Allowed difference in frames (default: 1)
    
    Returns:
        True if frame count matches (within tolerance)
    """
    # Load audio to get sample count
    audio, sr = torchaudio.load(audio_path)
    
    # Resample to 44.1kHz if needed (to match VAE processing)
    if sr != VAE_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, VAE_SAMPLE_RATE)
    
    audio_samples = audio.shape[-1]
    expected_frames = audio_samples // DOWNSAMPLE_RATE
    
    # Load latent to get actual frame count
    latent = torch.load(latent_path, map_location='cpu')
    actual_frames = latent.shape[-1]
    
    frame_diff = abs(actual_frames - expected_frames)
    
    if frame_diff > tolerance:
        raise ValueError(
            f"Frame count mismatch for {audio_path}:\n"
            f"  Audio samples: {audio_samples}\n"
            f"  Expected frames: {expected_frames}\n"
            f"  Actual frames: {actual_frames}\n"
            f"  Difference: {frame_diff} (tolerance: {tolerance})"
        )
    
    return True


# =============================================================================
# Batch Processing Helper
# =============================================================================

def process_audio_file(
    audio_path: str,
    output_dir: str,
    sample_id: str,
    vae_model,
    device: str = "cuda",
    chunked: bool = True,
    verbose: bool = False
) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single audio file: encode and save latent.
    
    This is a helper function for batch processing multiple files.
    
    Args:
        audio_path: Path to audio file
        output_dir: Output directory (latent saved to {output_dir}/latent/{sample_id}.pt)
        sample_id: Unique identifier for this sample
        vae_model: Loaded VAE model
        device: torch device
        chunked: Use chunked encoding
        verbose: Print progress information
    
    Returns:
        Tuple of (latent_path, success, error_message)
    
    Example:
        >>> vae = load_vae()
        >>> path, success, error = process_audio_file(
        ...     "song.mp3", "dataset", "song001", vae
        ... )
    """
    latent_output_path = os.path.join(output_dir, "latent", f"{sample_id}.pt")
    
    try:
        # Encode audio to latent
        if verbose:
            print(f"Processing: {audio_path}")
        
        latent = encode_audio_to_latent(
            audio_path, vae_model, device=device, chunked=chunked
        )
        
        # Save latent
        save_latent(latent, latent_output_path)
        
        # Verify it was saved correctly
        verify_latent(latent_output_path, verbose=False)
        
        if verbose:
            print(f"  ✓ Saved: {latent_output_path} (shape: {latent.shape})")
        
        return latent_output_path, True, None
        
    except Exception as e:
        error_msg = f"Error processing {audio_path}: {str(e)}"
        if verbose:
            print(f"  ✗ {error_msg}")
        return latent_output_path, False, error_msg
