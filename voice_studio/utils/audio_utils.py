"""
Audio utilities for Mira Voice Studio.

Includes audio level metering, normalization, and format conversion.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import soundfile as sf


def calculate_db_level(audio_data: np.ndarray) -> float:
    """
    Calculate the dB level of audio data.

    Args:
        audio_data: Audio samples as numpy array.

    Returns:
        RMS level in dB (relative to full scale).
    """
    if len(audio_data) == 0:
        return -np.inf

    # Calculate RMS
    rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))

    if rms == 0:
        return -np.inf

    # Convert to dB (0 dB = full scale)
    db = 20 * np.log10(rms)

    return db


def get_audio_quality_indicator(db_level: float) -> Tuple[str, str]:
    """
    Get a quality indicator based on audio level.

    Args:
        db_level: Audio level in dB.

    Returns:
        Tuple of (status, description):
        - "too_quiet": Below -40 dB
        - "good": Between -40 dB and -3 dB
        - "clipping": Above -3 dB
    """
    if db_level < -40:
        return "too_quiet", "Signal too quiet. Move closer to mic."
    elif db_level > -3:
        return "clipping", "Signal too loud! Move away from mic."
    else:
        return "good", "Good level"


def normalize_audio(
    audio_data: np.ndarray,
    target_db: float = -3.0,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Normalize audio to a target dB level.

    Args:
        audio_data: Audio samples as numpy array.
        target_db: Target peak level in dB.
        sample_rate: Sample rate (not used, for consistency).

    Returns:
        Normalized audio data.
    """
    if len(audio_data) == 0:
        return audio_data

    # Find current peak
    peak = np.max(np.abs(audio_data))

    if peak == 0:
        return audio_data

    # Calculate gain needed
    target_linear = 10 ** (target_db / 20)
    gain = target_linear / peak

    # Apply gain
    return audio_data * gain


def get_audio_duration(file_path: Path) -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        file_path: Path to the audio file.

    Returns:
        Duration in seconds.
    """
    try:
        info = sf.info(str(file_path))
        return info.duration
    except Exception:
        return 0.0


def get_audio_info(file_path: Path) -> dict:
    """
    Get detailed information about an audio file.

    Args:
        file_path: Path to the audio file.

    Returns:
        Dictionary with audio properties.
    """
    try:
        info = sf.info(str(file_path))
        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
            "frames": info.frames,
        }
    except Exception as e:
        return {"error": str(e)}


def load_audio(
    file_path: Path,
    target_sample_rate: Optional[int] = None,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file.

    Args:
        file_path: Path to the audio file.
        target_sample_rate: Resample to this rate if specified.
        mono: Convert to mono if True.

    Returns:
        Tuple of (audio_data, sample_rate).
    """
    audio_data, sample_rate = sf.read(str(file_path))

    # Convert to mono if needed
    if mono and len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample if needed
    if target_sample_rate and target_sample_rate != sample_rate:
        try:
            import torchaudio
            import torch

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_data).float()
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Resample
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sample_rate
            )
            audio_tensor = resampler(audio_tensor)
            audio_data = audio_tensor.squeeze().numpy()
            sample_rate = target_sample_rate
        except ImportError:
            # Fall back to scipy if torchaudio not available
            from scipy import signal
            num_samples = int(len(audio_data) * target_sample_rate / sample_rate)
            audio_data = signal.resample(audio_data, num_samples)
            sample_rate = target_sample_rate

    return audio_data, sample_rate


def save_audio(
    file_path: Path,
    audio_data: np.ndarray,
    sample_rate: int = 44100,
    subtype: str = "PCM_16"
) -> bool:
    """
    Save audio data to a file.

    Args:
        file_path: Output file path.
        audio_data: Audio samples as numpy array.
        sample_rate: Sample rate.
        subtype: Audio subtype (PCM_16 for 16-bit WAV).

    Returns:
        True if successful.
    """
    try:
        # Ensure parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Normalize to prevent clipping
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.99

        sf.write(str(file_path), audio_data, sample_rate, subtype=subtype)
        return True
    except Exception:
        return False


def generate_silence(duration_ms: int, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate silence of specified duration.

    Args:
        duration_ms: Duration in milliseconds.
        sample_rate: Sample rate.

    Returns:
        Array of zeros.
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(num_samples, dtype=np.float32)


def format_duration(seconds: float) -> str:
    """
    Format duration as HH:MM:SS or MM:SS.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def format_timecode(seconds: float, include_ms: bool = True) -> str:
    """
    Format time as SRT timecode (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds.
        include_ms: Include milliseconds.

    Returns:
        SRT-formatted timecode.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)

    if include_ms:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
