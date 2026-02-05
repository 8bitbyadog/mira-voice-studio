"""
TTS Engine base class for Mira Voice Studio.

Provides an abstract interface for TTS backends (Coqui, GPT-SoVITS, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Callable
import numpy as np

from voice_studio.core.text_processor import Sentence


@dataclass
class TTSResult:
    """Result from TTS generation for a single sentence."""

    sentence: Sentence
    audio_data: Optional[np.ndarray] = None
    sample_rate: int = 44100
    duration_seconds: float = 0.0
    success: bool = True
    error_message: str = ""

    @property
    def failed(self) -> bool:
        return not self.success

    def __post_init__(self):
        if self.audio_data is not None and self.duration_seconds == 0:
            self.duration_seconds = len(self.audio_data) / self.sample_rate


class TTSError(Exception):
    """Exception raised when TTS generation fails."""
    pass


class ModelNotFoundError(TTSError):
    """Exception raised when a voice model is not found."""
    pass


class TTSEngine(ABC):
    """
    Abstract base class for TTS engines.

    Subclasses must implement:
    - synthesize(): Generate audio for a single sentence
    - list_voices(): Return available voice models
    - load_voice(): Load a specific voice model
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        device: str = "auto"
    ):
        """
        Initialize the TTS engine.

        Args:
            sample_rate: Output sample rate.
            device: Compute device ("auto", "mps", "cpu").
        """
        self.sample_rate = sample_rate
        self._device = device
        self._current_voice: Optional[str] = None
        self._loaded = False

    @property
    def device(self) -> str:
        """Get the compute device."""
        if self._device == "auto":
            from voice_studio.utils.device import get_device
            return str(get_device())
        return self._device

    @property
    def current_voice(self) -> Optional[str]:
        """Get the currently loaded voice."""
        return self._current_voice

    @property
    def is_loaded(self) -> bool:
        """Check if a voice model is loaded."""
        return self._loaded

    @abstractmethod
    def synthesize(
        self,
        text: str,
        speed: float = 1.0
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech for a single text string.

        Args:
            text: The text to synthesize.
            speed: Speaking speed multiplier (0.5 to 2.0).

        Returns:
            Tuple of (audio_data, sample_rate).

        Raises:
            TTSError: If synthesis fails.
        """
        pass

    @abstractmethod
    def list_voices(self) -> List[str]:
        """
        List available voice models.

        Returns:
            List of voice model names.
        """
        pass

    @abstractmethod
    def load_voice(self, voice_name: str) -> None:
        """
        Load a voice model.

        Args:
            voice_name: Name of the voice to load.

        Raises:
            ModelNotFoundError: If the voice is not found.
        """
        pass

    def generate(
        self,
        sentence: Sentence,
        speed: float = 1.0
    ) -> TTSResult:
        """
        Generate audio for a Sentence object.

        Args:
            sentence: The Sentence to synthesize.
            speed: Speaking speed multiplier.

        Returns:
            TTSResult with audio data and metadata.
        """
        try:
            audio_data, sample_rate = self.synthesize(sentence.text, speed)

            return TTSResult(
                sentence=sentence,
                audio_data=audio_data,
                sample_rate=sample_rate,
                duration_seconds=len(audio_data) / sample_rate,
                success=True,
            )

        except Exception as e:
            return TTSResult(
                sentence=sentence,
                audio_data=None,
                sample_rate=self.sample_rate,
                success=False,
                error_message=str(e),
            )

    def generate_batch(
        self,
        sentences: List[Sentence],
        speed: float = 1.0,
        progress_callback: Optional[Callable[[int, int, Sentence], None]] = None
    ) -> List[TTSResult]:
        """
        Generate audio for multiple sentences.

        Args:
            sentences: List of sentences to synthesize.
            speed: Speaking speed multiplier.
            progress_callback: Optional callback(current, total, sentence).

        Returns:
            List of TTSResult objects.
        """
        results = []

        for i, sentence in enumerate(sentences):
            if progress_callback:
                progress_callback(i + 1, len(sentences), sentence)

            result = self.generate(sentence, speed)
            results.append(result)

        return results

    def get_info(self) -> dict:
        """
        Get information about the TTS engine.

        Returns:
            Dictionary with engine info.
        """
        return {
            "engine": self.__class__.__name__,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "current_voice": self._current_voice,
            "is_loaded": self._loaded,
            "available_voices": self.list_voices(),
        }

    def unload(self) -> None:
        """Unload the current voice model to free memory."""
        self._current_voice = None
        self._loaded = False
