"""
Lyrics Tokenization Module for DiffRhythm Data Preprocessing

Parses .lrc (Lyric) files and converts them to tokenized format (.pt)
that the DiffRhythm training pipeline expects.

The tokenization pipeline:
  1. Parse LRC file into (timestamp, text) pairs
  2. Convert each lyric line to phoneme tokens via G2P
  3. Shift all token IDs by +1 (reserving 0 as padding token)
  4. Save as dict with 'time' (list of floats) and 'lrc' (list of token ID lists)

Code origins:
  - parse_lrc_file():    Based on infer/infer_utils.py lines 333-345 (parse_lyrics)
  - tokenize_lyrics():   Based on infer/infer_utils.py lines 348-383
                          (CNENTokenizer + get_lrc_token token shifting)
  - Token ID shifting:   Based on infer/infer_utils.py lines 357-359
                          (CNENTokenizer.encode applies +1 to all token IDs)
  - Token ID constants:  Based on dataset/dataset.py lines 32-35
                          (pad=0, comma=1, period=2 after the +1 shift)

Output format (consumed by dataset/dataset.py lines 58-61):
  {
      'time': [float, ...],          # Start time of each lyric line in seconds
      'lrc':  [[int, ...], ...],     # Token IDs for each line (+1 shifted)
  }

Usage (from repo root):
    from preprocess.lyrics_tokenization import tokenize_lyrics, save_lrc_tokens

    lrc_data = tokenize_lyrics("path/to/song.lrc")
    save_lrc_tokens(lrc_data, "dataset/lrc/song001.pt")
"""

import os
import sys
from typing import List, Tuple, Dict, Optional

import torch

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so g2p imports work regardless of cwd.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from g2p.g2p_generation import chn_eng_g2p


# ---------------------------------------------------------------------------
# Token ID constants (after the +1 shift applied in this module).
# These match dataset/dataset.py lines 32-35.
# ---------------------------------------------------------------------------
PAD_TOKEN_ID = 0       # Reserved padding (0 is never a real token after shift)
COMMA_TOKEN_ID = 1     # vocab.json "," = 0  ->  0 + 1 = 1
PERIOD_TOKEN_ID = 2    # vocab.json "." = 1  ->  1 + 1 = 2


# ---------------------------------------------------------------------------
# LRC Parsing
# ---------------------------------------------------------------------------

def parse_lrc_file(lrc_path: str) -> List[Tuple[float, str]]:
    """
    Parse an LRC file into (timestamp_seconds, lyric_text) pairs.

    Based on: infer/infer_utils.py lines 333-345 (parse_lyrics function).

    The expected LRC timestamp format is ``[MM:SS.xx]`` where *MM* is minutes,
    *SS* is seconds, and *xx* is hundredths of a second.

    Args:
        lrc_path: Path to a ``.lrc`` file.

    Returns:
        Sorted list of ``(timestamp_in_seconds, lyric_text)`` tuples.
        Lines with empty text are skipped.
    """
    lyrics_with_time: List[Tuple[float, str]] = []

    with open(lrc_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("["):
                continue

            try:
                # Extract timestamp [MM:SS.xx]  (positions 1-8 inside brackets)
                time_str = line[1:9]   # e.g. "00:10.00"
                text = line[10:].strip()  # everything after the ']'

                mins, secs = time_str.split(":")
                timestamp = int(mins) * 60 + float(secs)

                if text:  # skip empty lines
                    lyrics_with_time.append((timestamp, text))
            except (ValueError, IndexError):
                # Skip malformed lines
                continue

    return lyrics_with_time


def parse_lrc_string(lyrics: str) -> List[Tuple[float, str]]:
    """
    Parse an LRC-formatted *string* into (timestamp, text) pairs.

    This is a convenience wrapper matching the exact behaviour of
    ``infer/infer_utils.py:parse_lyrics`` (lines 333-345) which operates
    on a string rather than a file path.

    Args:
        lyrics: Multi-line string in LRC format.

    Returns:
        List of ``(timestamp_in_seconds, lyric_text)`` tuples.
    """
    lyrics_with_time: List[Tuple[float, str]] = []
    lyrics = lyrics.strip()

    for line in lyrics.split("\n"):
        try:
            time_str = line[1:9]
            text = line[10:].strip()

            mins, secs = time_str.split(":")
            timestamp = int(mins) * 60 + float(secs)

            lyrics_with_time.append((timestamp, text))
        except (ValueError, IndexError):
            continue

    return lyrics_with_time


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _shift_tokens(tokens: List[int]) -> List[int]:
    """
    Apply the +1 shift to raw G2P token IDs.

    The ``chn_eng_g2p`` function returns 0-based IDs that map directly to
    ``g2p/g2p/vocab.json``.  The training pipeline reserves **0 as a padding
    token**, so every real token is shifted by +1.

    This mirrors ``CNENTokenizer.encode`` in
    ``infer/infer_utils.py`` lines 357-359:

        token = [x + 1 for x in token]

    Args:
        tokens: Raw 0-based token IDs from ``chn_eng_g2p``.

    Returns:
        Shifted token IDs (each value incremented by 1).
    """
    return [t + 1 for t in tokens]


def tokenize_lyrics(lrc_path: str) -> Dict[str, list]:
    """
    Convert an LRC file to the tokenized format expected by the training
    dataset (``dataset/dataset.py`` lines 58-61).

    For each lyric line the function:
      1. Runs ``chn_eng_g2p`` (from ``g2p/g2p_generation.py``) to get phoneme
         token IDs.
      2. Shifts every token ID by +1 so that 0 is reserved for padding, matching
         ``CNENTokenizer.encode`` (``infer/infer_utils.py`` lines 357-359).

    Args:
        lrc_path: Path to a ``.lrc`` file.

    Returns:
        ``dict`` with keys:
          - ``'time'``: ``List[float]``  – start time of each lyric line (seconds).
          - ``'lrc'``:  ``List[List[int]]`` – shifted token IDs per line.
    """
    lyrics_with_time = parse_lrc_file(lrc_path)

    times: List[float] = []
    token_lists: List[List[int]] = []

    for timestamp, text in lyrics_with_time:
        _phonemes, raw_tokens = chn_eng_g2p(text)
        shifted = _shift_tokens(raw_tokens)

        times.append(timestamp)
        token_lists.append(shifted)

    return {"time": times, "lrc": token_lists}


def tokenize_lyrics_from_string(lyrics: str) -> Dict[str, list]:
    """
    Tokenize LRC-formatted lyrics provided as a string (not a file).

    Behaves identically to :func:`tokenize_lyrics` but accepts a raw string
    instead of a file path.  Useful for inline / on-the-fly tokenization during
    inference.

    Args:
        lyrics: Multi-line string in LRC format.

    Returns:
        ``dict`` with ``'time'`` and ``'lrc'`` keys (same as :func:`tokenize_lyrics`).
    """
    lyrics_with_time = parse_lrc_string(lyrics)

    times: List[float] = []
    token_lists: List[List[int]] = []

    for timestamp, text in lyrics_with_time:
        _phonemes, raw_tokens = chn_eng_g2p(text)
        shifted = _shift_tokens(raw_tokens)

        times.append(timestamp)
        token_lists.append(shifted)

    return {"time": times, "lrc": token_lists}


# ---------------------------------------------------------------------------
# Instrumental (no lyrics) helper
# ---------------------------------------------------------------------------

def create_instrumental_lrc(duration_seconds: float) -> Dict[str, list]:
    """
    Create a minimal tokenized LRC structure for instrumental music.

    The training ``dataset/dataset.py`` (lines 78-81) will raise on an empty
    LRC, so we provide a single dummy entry near the end of the track.

    The dummy token is ``PERIOD_TOKEN_ID`` (shifted ``"."`` = 2), following the
    convention described in docs/DATA_PREPROCESSING_GUIDE.md §8.2.

    Args:
        duration_seconds: Total length of the audio in seconds.

    Returns:
        ``dict`` with a single-entry ``'time'`` and ``'lrc'``.
    """
    end_time = max(0.0, duration_seconds - 1.0)
    return {
        "time": [end_time],
        "lrc": [[PERIOD_TOKEN_ID]],  # single period token (already shifted)
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_lrc_tokens(lrc_data: Dict[str, list], output_path: str) -> None:
    """
    Save tokenized lyrics to a ``.pt`` file.

    Args:
        lrc_data: Dict with ``'time'`` and ``'lrc'`` keys (as returned by
                  :func:`tokenize_lyrics`).
        output_path: Destination ``.pt`` file path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(lrc_data, output_path)


def load_lrc_tokens(pt_path: str) -> Dict[str, list]:
    """
    Load a tokenized LRC ``.pt`` file.

    Args:
        pt_path: Path to the ``.pt`` file.

    Returns:
        Dict with ``'time'`` and ``'lrc'`` keys.
    """
    return torch.load(pt_path, map_location="cpu")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_lrc(
    path: str,
    *,
    verbose: bool = False,
    max_token_id: int = 363,
) -> bool:
    """
    Verify that a tokenized LRC ``.pt`` file is correctly formatted.

    Checks performed:
      - File loads successfully.
      - Contains ``'time'`` and ``'lrc'`` keys.
      - Lengths match.
      - All timestamps are numeric.
      - All token lists are non-empty lists of ints.
      - All token IDs are in range ``[1, max_token_id]`` (0 is padding, never
        stored as a real token after the +1 shift).

    Args:
        path: Path to the ``.pt`` file.
        verbose: If ``True``, print summary statistics.
        max_token_id: Upper bound for valid token IDs (inclusive).
                      Default 363 = vocab size (362) + 1 shift.

    Returns:
        ``True`` if the file passes all checks.

    Raises:
        AssertionError: On any validation failure.
    """
    lrc = torch.load(path, map_location="cpu")

    assert isinstance(lrc, dict), "LRC data should be a dict"
    assert "time" in lrc, "Missing 'time' key"
    assert "lrc" in lrc, "Missing 'lrc' key"

    times = lrc["time"]
    tokens = lrc["lrc"]

    assert len(times) == len(tokens), (
        f"Mismatched lengths: {len(times)} times vs {len(tokens)} token lists"
    )
    assert all(isinstance(t, (int, float)) for t in times), (
        "All timestamps must be numeric"
    )
    assert all(isinstance(tl, list) for tl in tokens), (
        "Each entry in 'lrc' must be a list"
    )

    for i, tl in enumerate(tokens):
        assert len(tl) > 0, f"Token list at index {i} is empty"
        for tok in tl:
            assert isinstance(tok, int), (
                f"Token at line {i} is not an int: {tok!r}"
            )
            assert 1 <= tok <= max_token_id, (
                f"Token {tok} at line {i} out of range [1, {max_token_id}]"
            )

    if verbose:
        print(f"  Number of lines:  {len(times)}")
        print(f"  Time range:       {min(times):.2f}s – {max(times):.2f}s")
        print(f"  Total tokens:     {sum(len(tl) for tl in tokens)}")
        all_ids = [tok for tl in tokens for tok in tl]
        print(f"  Token ID range:   [{min(all_ids)}, {max(all_ids)}]")

    return True
