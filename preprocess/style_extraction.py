"""
Style Embedding Extraction Module

This module handles extracting 512-dimensional style embeddings from audio or text
using the MuQ-MuLan model for DiffRhythm training.

MuQ-MuLan is a two-tower contrastive learning model (like CLIP) that maps both
audio and text into a shared 512-dimensional embedding space. This means:
- Audio of "smooth jazz piano" and the text "smooth jazz piano" produce similar vectors
- Both modalities are geometrically compatible in the same space

Extraction modes:
    1. Audio-based:  Extract style from a 10s sample of the audio (default for training)
    2. Text-based:   Extract style from a text description (e.g. "dreamy lo-fi beat")
    3. Combined:     [EXPERIMENTAL] Interpolate audio + text embeddings in the shared
                     MuLan space, inspired by CLIP-guided generation techniques.
                     This is NOT described in the MuQ paper (arXiv:2501.01108v2) and
                     should be treated as experimental.

Key specifications:
- Output: Style embedding tensors [1, 512] saved as .pt files
- MuQ sample rate: 24000 Hz (different from VAE's 44100 Hz)
- Minimum audio duration: 10 seconds

References:
- docs/DATA_PREPROCESSING_GUIDE.md (Section 5)
- infer/infer_utils.py lines 295-330 (get_style_prompt)
- MuQ paper: arXiv:2501.01108v2
"""

import os
import torch
import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, List


# =============================================================================
# Constants
# =============================================================================

MUQ_SAMPLE_RATE = 24000       # MuQ expects 24kHz audio (NOT 44.1kHz like VAE)
STYLE_DIM = 512               # MuQ-MuLan embedding dimension
MIN_AUDIO_DURATION = 10.0     # MuQ requires at least 10 seconds
STYLE_SAMPLE_DURATION = 10.0  # We extract a 10-second sample for embedding


# =============================================================================
# Model Loading
# =============================================================================

def load_muq(device: str = "cuda", cache_dir: str = "./pretrained"):
    """
    Load the MuQ-MuLan model from HuggingFace.

    MuQ-MuLan is a joint music-text embedding model that maps both audio and
    text descriptions into a shared 512-dimensional space via contrastive learning.

    Args:
        device: torch device ('cuda' or 'cpu')
        cache_dir: Directory to cache downloaded models

    Returns:
        muq: Loaded MuQ-MuLan model in eval mode

    Reference:
        - HuggingFace repo: OpenMuQ/MuQ-MuLan-large
        - infer/infer_utils.py (model loading pattern)
    """
    from muq import MuQMuLan

    muq = MuQMuLan.from_pretrained(
        "OpenMuQ/MuQ-MuLan-large",
        cache_dir=cache_dir,
    )
    muq = muq.to(device).eval()

    return muq


# =============================================================================
# Audio-Based Style Extraction
# =============================================================================

def extract_style_from_audio(
    audio_path: str,
    muq_model,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Extract a style embedding from a 10-second audio sample using MuQ-MuLan.

    Takes a 10-second clip from the middle of the track, which typically captures
    the most representative style (avoids intros/outros).

    Args:
        audio_path: Path to audio file (MP3, WAV, FLAC)
        muq_model: Loaded MuQ-MuLan model
        device: torch device

    Returns:
        style_embedding: Tensor of shape [1, 512]

    Raises:
        ValueError: If audio is shorter than 10 seconds

    Reference:
        - ADAPTED FROM: infer/infer_utils.py lines 295-330 (get_style_prompt)
        - Changes: Uses librosa.get_duration for all formats instead of MP3-specific
          mutagen path. Otherwise functionally identical.
    """
    # Get audio duration
    duration = librosa.get_duration(path=audio_path)

    if duration < MIN_AUDIO_DURATION:
        raise ValueError(
            f"Audio file {audio_path} is too short ({duration:.2f}s). "
            f"MuQ requires at least {MIN_AUDIO_DURATION} seconds of audio."
        )

    # Extract 10 seconds from the middle of the song
    mid_time = duration / 2
    start_time = mid_time - (STYLE_SAMPLE_DURATION / 2)

    # Load audio at 24kHz (MuQ's expected sample rate)
    wav, _ = librosa.load(
        audio_path,
        sr=MUQ_SAMPLE_RATE,
        offset=start_time,
        duration=STYLE_SAMPLE_DURATION,
    )

    # Convert to tensor: [samples] → [1, samples]
    wav = torch.tensor(wav).unsqueeze(0).to(device)

    # Extract embedding
    with torch.no_grad():
        style_embedding = muq_model(wavs=wav)  # [1, 512]

    return style_embedding.cpu()


# =============================================================================
# Text-Based Style Extraction
# =============================================================================

def extract_style_from_text(
    text_prompt: Union[str, List[str]],
    muq_model,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Extract a style embedding from a text description using MuQ-MuLan.

    The text encoder (RoBERTa + Transformer) maps natural language descriptions
    into the same 512-dim space as audio embeddings.

    Good text prompts describe genre, mood, instrumentation, and tempo:
        - "A smooth jazz track with piano and upright bass"
        - "Energetic electronic dance music with heavy bass drops"
        - "Gentle acoustic folk song with fingerpicked guitar"

    Args:
        text_prompt: Style description string, or list of strings for batch processing
        muq_model: Loaded MuQ-MuLan model
        device: torch device

    Returns:
        style_embedding: Tensor of shape [1, 512] (or [N, 512] for batch)

    Reference:
        - infer/infer_utils.py lines 299-300 (text prompt path in get_style_prompt)
        - MuQ paper Section "Music-Text Contrastive Learning"
    """
    with torch.no_grad():
        style_embedding = muq_model(texts=text_prompt)  # [1, 512] or [N, 512]

    return style_embedding.cpu()


# =============================================================================
# Combined Style Extraction (EXPERIMENTAL)
# =============================================================================

def extract_style_combined(
    audio_path: str,
    text_prompt: str,
    muq_model,
    alpha: float = 0.5,
    normalize: bool = True,
    device: str = "cuda",
) -> torch.Tensor:
    """
    [EXPERIMENTAL] Extract a blended style embedding from both audio and text.

    This combines audio-derived and text-derived style embeddings via linear
    interpolation in MuQ-MuLan's shared embedding space. This technique is
    borrowed from CLIP-guided image generation and is NOT described in the
    MuQ paper (arXiv:2501.01108v2).

    The idea: MuQ-MuLan is trained with contrastive learning to align audio
    and text in the same 512-dim space. Because both modalities share geometry,
    interpolating between them produces a meaningful blend:
        - The audio embedding anchors the result to the track's actual acoustics
        - The text embedding steers toward qualities described in words

    Example use case:
        You have a reference track and want to train with its style but nudge
        it toward "more aggressive" or "lo-fi aesthetic". Use alpha=0.7 to keep
        mostly the audio character while pulling toward the text description.

    WARNING: This is experimental. The MuQ authors validated audio and text
    embeddings separately, not blended. Results may vary. Recommended to A/B
    test against pure audio embeddings on your specific use case.

    Args:
        audio_path: Path to audio file (MP3, WAV, FLAC)
        text_prompt: Style description to blend with audio
        muq_model: Loaded MuQ-MuLan model
        alpha: Blending weight for audio vs text (default: 0.5)
                 - alpha=1.0: Pure audio embedding (same as extract_style_from_audio)
                 - alpha=0.0: Pure text embedding (same as extract_style_from_text)
                 - alpha=0.5: Equal blend of both
        normalize: L2-normalize the result to keep it on the unit sphere (recommended).
                   Contrastive models typically operate on normalized embeddings,
                   so renormalizing after interpolation preserves expected magnitude.
        device: torch device

    Returns:
        style_embedding: Tensor of shape [1, 512]

    Raises:
        ValueError: If alpha is not in [0, 1] or audio is too short
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    # Get both embeddings
    audio_emb = extract_style_from_audio(audio_path, muq_model, device=device)
    text_emb = extract_style_from_text(text_prompt, muq_model, device=device)

    # Linear interpolation in shared MuLan space
    combined = alpha * audio_emb + (1.0 - alpha) * text_emb

    # L2 normalize to keep on the unit sphere
    # Contrastive models use cosine similarity (i.e., normalized dot product),
    # so the learned embedding space lives on/near the unit hypersphere.
    # After interpolation the vector magnitude shrinks; renormalizing restores it.
    if normalize:
        combined = torch.nn.functional.normalize(combined, p=2, dim=-1)

    return combined


# =============================================================================
# I/O and Validation Functions
# =============================================================================

def save_style_file(style_embedding: torch.Tensor, output_path: str) -> None:
    """
    Save style embedding to disk as .pt file.

    The dataset expects either [1, 512] or [512] (batch dim is optional).
    We save with batch dimension for consistency.

    Args:
        style_embedding: Style tensor [1, 512] or [512]
        output_path: Path to save .pt file

    Reference:
        - dataset/dataset.py lines 66-67 shows how training loads style embeddings
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Verify shape before saving
    if style_embedding.dim() == 2:
        assert style_embedding.shape[-1] == STYLE_DIM, \
            f"Expected style dim {STYLE_DIM}, got {style_embedding.shape[-1]}"
    elif style_embedding.dim() == 1:
        assert style_embedding.shape[0] == STYLE_DIM, \
            f"Expected style dim {STYLE_DIM}, got {style_embedding.shape[0]}"
    else:
        raise ValueError(f"Invalid style dimensions: {style_embedding.shape}")

    torch.save(style_embedding, output_path)


def verify_style_file(path: str, verbose: bool = True) -> bool:
    """
    Verify that a saved style embedding file is correctly formatted.

    Checks:
    - File can be loaded
    - Shape is [1, 512] or [512]
    - Values are reasonable (not NaN/Inf)

    Args:
        path: Path to .pt file
        verbose: Print detailed information

    Returns:
        True if valid, raises exception if invalid

    Reference:
        - docs/DATA_PREPROCESSING_GUIDE.md section 10.3
    """
    try:
        style = torch.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load style from {path}: {e}")

    # Check dimensions
    if style.dim() == 2:
        if style.shape != (1, STYLE_DIM):
            raise ValueError(f"Expected [1, {STYLE_DIM}], got {list(style.shape)}")
    elif style.dim() == 1:
        if style.shape[0] != STYLE_DIM:
            raise ValueError(f"Expected [{STYLE_DIM}], got {list(style.shape)}")
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {style.dim()}D")

    # Check for NaN/Inf
    if torch.isnan(style).any():
        raise ValueError("Style embedding contains NaN values")
    if torch.isinf(style).any():
        raise ValueError("Style embedding contains Inf values")

    if verbose:
        norm = torch.norm(style.float(), p=2, dim=-1).item()
        print(f"✓ Style file valid: {path}")
        print(f"  Shape: {list(style.shape)}")
        print(f"  Dtype: {style.dtype}")
        print(f"  Range: [{style.min():.4f}, {style.max():.4f}]")
        print(f"  L2 norm: {norm:.4f}")

    return True


# =============================================================================
# Batch Processing Helper
# =============================================================================

def process_style(
    audio_path: str,
    output_dir: str,
    sample_id: str,
    muq_model,
    device: str = "cuda",
    text_prompt: Optional[str] = None,
    alpha: float = 0.5,
    normalize_combined: bool = True,
    verbose: bool = False,
) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single sample: extract style embedding and save.

    Supports three modes depending on arguments:
    - Audio only (default): pass audio_path, no text_prompt
    - Text only:            pass text_prompt, no audio_path
    - Combined:             pass both audio_path and text_prompt (EXPERIMENTAL)

    Args:
        audio_path: Path to audio file (None for text-only mode)
        output_dir: Output directory (style saved to {output_dir}/style/{sample_id}.pt)
        sample_id: Unique identifier for this sample
        muq_model: Loaded MuQ-MuLan model
        device: torch device
        text_prompt: Optional text description for text or combined mode
        alpha: Blending weight for combined mode (1.0=audio, 0.0=text)
        normalize_combined: L2-normalize combined embeddings (recommended)
        verbose: Print progress information

    Returns:
        Tuple of (style_path, success, error_message)
    """
    style_output_path = os.path.join(output_dir, "style", f"{sample_id}.pt")

    try:
        if verbose:
            print(f"Extracting style: {sample_id}")

        # Determine extraction mode
        has_audio = audio_path is not None
        has_text = text_prompt is not None

        if has_audio and has_text:
            # Combined mode (EXPERIMENTAL)
            if verbose:
                print(f"  Mode: combined (alpha={alpha})")
            style = extract_style_combined(
                audio_path, text_prompt, muq_model,
                alpha=alpha, normalize=normalize_combined, device=device,
            )
        elif has_audio:
            # Audio-only mode (standard)
            if verbose:
                print(f"  Mode: audio")
            style = extract_style_from_audio(audio_path, muq_model, device=device)
        elif has_text:
            # Text-only mode
            if verbose:
                print(f"  Mode: text")
            style = extract_style_from_text(text_prompt, muq_model, device=device)
        else:
            raise ValueError("Must provide at least one of audio_path or text_prompt")

        # Save
        save_style_file(style, style_output_path)

        # Verify
        verify_style_file(style_output_path, verbose=False)

        if verbose:
            print(f"  ✓ Saved: {style_output_path} (shape: {list(style.shape)})")

        return style_output_path, True, None

    except Exception as e:
        error_msg = f"Error extracting style for {sample_id}: {str(e)}"
        if verbose:
            print(f"  ✗ {error_msg}")
        return style_output_path, False, error_msg
