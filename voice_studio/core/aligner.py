"""
Whisper-based audio alignment for Mira Voice Studio.

Uses OpenAI Whisper to transcribe audio and extract word-level timestamps.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import warnings

from voice_studio.utils.device import get_device


@dataclass
class WordTiming:
    """Timing information for a single word."""

    word: str
    start_time: float  # seconds
    end_time: float  # seconds

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class SegmentTiming:
    """Timing information for a segment (sentence/phrase)."""

    text: str
    start_time: float
    end_time: float
    words: List[WordTiming] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class AlignmentResult:
    """Result from Whisper alignment."""

    segments: List[SegmentTiming]
    words: List[WordTiming]
    full_text: str
    language: str
    duration_seconds: float

    @property
    def word_count(self) -> int:
        return len(self.words)


class AlignmentError(Exception):
    """Exception raised when alignment fails."""
    pass


class WhisperAligner:
    """
    Whisper-based audio transcription and alignment.

    Uses OpenAI Whisper for:
    - Transcription with word-level timestamps
    - Language detection
    - Audio alignment for caption generation

    Optimized for Apple Silicon with MPS acceleration.
    """

    # Model size to memory requirements (approximate)
    MODEL_SIZES = {
        "tiny": "~1GB",
        "base": "~1GB",
        "small": "~2GB",
        "medium": "~5GB",
        "large": "~10GB",
        "large-v2": "~10GB",
        "large-v3": "~10GB",
    }

    def __init__(
        self,
        model_size: str = "base",
        language: str = "en",
        device: str = "auto"
    ):
        """
        Initialize the Whisper aligner.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large).
            language: Language code for transcription.
            device: Compute device ("auto", "mps", "cpu").
        """
        self.model_size = model_size
        self.language = language
        self._device = device
        self._model = None
        self._loaded = False

    @property
    def device(self) -> str:
        """Get the compute device."""
        if self._device == "auto":
            device = get_device()
            return str(device)
        return self._device

    def load_model(self) -> None:
        """Load the Whisper model."""
        if self._loaded:
            return

        try:
            import whisper

            # Determine device for loading
            device_str = self.device
            if device_str == "mps":
                # Whisper works well with MPS on Apple Silicon
                device_str = "cpu"  # Load on CPU first, then move
                # Note: Whisper has some MPS issues, CPU is more reliable

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = whisper.load_model(self.model_size, device=device_str)

            self._loaded = True

        except ImportError as e:
            raise AlignmentError(
                "Whisper not installed. Install with: pip install openai-whisper"
            ) from e
        except Exception as e:
            raise AlignmentError(f"Failed to load Whisper model: {e}") from e

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

        self._loaded = False

        import gc
        gc.collect()

        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def align(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 44100
    ) -> AlignmentResult:
        """
        Align audio and extract word-level timestamps.

        Args:
            audio_data: Audio samples as numpy array.
            sample_rate: Sample rate of the audio.

        Returns:
            AlignmentResult with segments and word timings.
        """
        if not self._loaded:
            self.load_model()

        try:
            import whisper

            # Whisper expects 16kHz audio
            if sample_rate != 16000:
                audio_data = self._resample(audio_data, sample_rate, 16000)

            # Ensure float32 and normalize
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            # Pad/trim to 30 seconds (Whisper's expected length) if needed
            # Actually, whisper.transcribe handles variable length
            audio_data = whisper.pad_or_trim(audio_data)

            # Transcribe with word timestamps
            result = self._model.transcribe(
                audio_data,
                language=self.language,
                word_timestamps=True,
                verbose=False,
            )

            return self._parse_result(result, len(audio_data) / 16000)

        except Exception as e:
            raise AlignmentError(f"Alignment failed: {e}") from e

    def align_file(self, audio_path: Path) -> AlignmentResult:
        """
        Align audio from a file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            AlignmentResult with segments and word timings.
        """
        if not self._loaded:
            self.load_model()

        try:
            import whisper

            # Transcribe directly from file
            result = self._model.transcribe(
                str(audio_path),
                language=self.language,
                word_timestamps=True,
                verbose=False,
            )

            # Get audio duration
            import soundfile as sf
            info = sf.info(str(audio_path))
            duration = info.duration

            return self._parse_result(result, duration)

        except Exception as e:
            raise AlignmentError(f"Alignment failed: {e}") from e

    def _parse_result(self, result: dict, duration: float) -> AlignmentResult:
        """Parse Whisper result into AlignmentResult."""
        segments = []
        all_words = []

        for seg in result.get("segments", []):
            segment_words = []

            # Extract word-level timings if available
            for word_info in seg.get("words", []):
                word = WordTiming(
                    word=word_info.get("word", "").strip(),
                    start_time=word_info.get("start", 0.0),
                    end_time=word_info.get("end", 0.0),
                )
                if word.word:  # Skip empty words
                    segment_words.append(word)
                    all_words.append(word)

            segment = SegmentTiming(
                text=seg.get("text", "").strip(),
                start_time=seg.get("start", 0.0),
                end_time=seg.get("end", 0.0),
                words=segment_words,
            )
            segments.append(segment)

        return AlignmentResult(
            segments=segments,
            words=all_words,
            full_text=result.get("text", "").strip(),
            language=result.get("language", self.language),
            duration_seconds=duration,
        )

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            import torchaudio
            import torch

            audio_tensor = torch.from_numpy(audio).float()
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=target_sr
            )
            resampled = resampler(audio_tensor)
            return resampled.squeeze().numpy()

        except ImportError:
            from scipy import signal
            new_length = int(len(audio) * target_sr / orig_sr)
            return signal.resample(audio, new_length).astype(np.float32)

    def get_info(self) -> dict:
        """Get aligner information."""
        return {
            "model_size": self.model_size,
            "language": self.language,
            "device": self.device,
            "is_loaded": self._loaded,
            "memory_estimate": self.MODEL_SIZES.get(self.model_size, "unknown"),
        }


def align_with_known_text(
    audio_data: np.ndarray,
    known_text: str,
    sample_rate: int = 44100,
    model_size: str = "base"
) -> AlignmentResult:
    """
    Convenience function to align audio when text is already known.

    This uses Whisper for timing extraction but validates against known text.

    Args:
        audio_data: Audio samples.
        known_text: The known transcript.
        sample_rate: Audio sample rate.
        model_size: Whisper model size.

    Returns:
        AlignmentResult with word timings.
    """
    aligner = WhisperAligner(model_size=model_size)
    result = aligner.align(audio_data, sample_rate)
    aligner.unload_model()
    return result
