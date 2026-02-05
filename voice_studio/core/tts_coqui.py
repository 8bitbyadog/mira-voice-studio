"""
Coqui TTS implementation for Mira Voice Studio.

Uses Coqui TTS (https://github.com/coqui-ai/TTS) for speech synthesis.
This is the Phase 1 TTS engine - simple and reliable.
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import warnings

from voice_studio.core.tts_engine import TTSEngine, TTSError, ModelNotFoundError
from voice_studio.utils.device import get_device


class CoquiTTS(TTSEngine):
    """
    Coqui TTS engine implementation.

    Uses the Coqui TTS library with support for various pre-trained models.
    Optimized for Apple Silicon using MPS acceleration.
    """

    # Default models that work well on macOS
    DEFAULT_MODELS = {
        "default": "tts_models/en/ljspeech/tacotron2-DDC",
        "fast": "tts_models/en/ljspeech/speedy-speech",
        "vits": "tts_models/en/ljspeech/vits",
        "vits_neon": "tts_models/en/ljspeech/vits--neon",
    }

    def __init__(
        self,
        sample_rate: int = 44100,
        device: str = "auto",
        model_name: Optional[str] = None
    ):
        """
        Initialize Coqui TTS.

        Args:
            sample_rate: Output sample rate.
            device: Compute device ("auto", "mps", "cpu").
            model_name: Model to load initially (uses default if None).
        """
        super().__init__(sample_rate=sample_rate, device=device)

        self._tts = None
        self._model_name = None
        self._available_models: List[str] = []

        # Suppress warnings during import
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._init_tts_library()

        if model_name:
            self.load_voice(model_name)

    def _init_tts_library(self) -> None:
        """Initialize the TTS library and discover available models."""
        try:
            from TTS.api import TTS
            from TTS.utils.manage import ModelManager

            # Get available models
            manager = ModelManager()
            self._available_models = list(self.DEFAULT_MODELS.keys())

            # Also add the raw model names
            for model_name in self.DEFAULT_MODELS.values():
                if model_name not in self._available_models:
                    self._available_models.append(model_name)

        except ImportError as e:
            raise TTSError(
                "Coqui TTS not installed. Install with: pip install TTS"
            ) from e

    def list_voices(self) -> List[str]:
        """
        List available voice models.

        Returns:
            List of model names (friendly names and raw model paths).
        """
        return self._available_models.copy()

    def load_voice(self, voice_name: str) -> None:
        """
        Load a TTS model.

        Args:
            voice_name: Model name (friendly name or full path).

        Raises:
            ModelNotFoundError: If the model doesn't exist.
        """
        from TTS.api import TTS

        # Resolve friendly names
        model_path = self.DEFAULT_MODELS.get(voice_name, voice_name)

        # Determine device
        device_str = self.device
        if device_str == "mps":
            # Coqui TTS uses "mps" directly
            device_str = "mps"
        elif device_str.startswith("cuda"):
            device_str = device_str
        else:
            device_str = "cpu"

        try:
            # Initialize TTS with the model
            self._tts = TTS(model_name=model_path, progress_bar=False)

            # Move to device if supported
            try:
                self._tts.to(device_str)
            except Exception:
                # Some models don't support .to() - fall back to CPU
                pass

            self._model_name = voice_name
            self._current_voice = voice_name
            self._loaded = True

        except Exception as e:
            self._loaded = False
            raise ModelNotFoundError(f"Failed to load model '{voice_name}': {e}") from e

    def synthesize(
        self,
        text: str,
        speed: float = 1.0
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech for text.

        Args:
            text: Text to synthesize.
            speed: Speaking speed (0.5 to 2.0).

        Returns:
            Tuple of (audio_data, sample_rate).
        """
        if not self._loaded or self._tts is None:
            # Auto-load default model
            self.load_voice("default")

        if not text.strip():
            # Return silence for empty text
            silence = np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)
            return silence, self.sample_rate

        try:
            # Generate audio
            # Note: Coqui TTS returns audio at its native sample rate
            wav = self._tts.tts(text=text)

            # Convert to numpy array if needed
            if isinstance(wav, list):
                audio_data = np.array(wav, dtype=np.float32)
            else:
                audio_data = wav.astype(np.float32)

            # Get native sample rate from the model
            native_sr = getattr(self._tts, "sample_rate", 22050)
            if hasattr(self._tts, "synthesizer") and hasattr(self._tts.synthesizer, "output_sample_rate"):
                native_sr = self._tts.synthesizer.output_sample_rate

            # Apply speed adjustment if needed
            if speed != 1.0 and 0.5 <= speed <= 2.0:
                audio_data = self._adjust_speed(audio_data, native_sr, speed)

            # Resample to target sample rate if needed
            if native_sr != self.sample_rate:
                audio_data = self._resample(audio_data, native_sr, self.sample_rate)

            return audio_data, self.sample_rate

        except Exception as e:
            raise TTSError(f"Synthesis failed: {e}") from e

    def _adjust_speed(
        self,
        audio: np.ndarray,
        sample_rate: int,
        speed: float
    ) -> np.ndarray:
        """
        Adjust audio playback speed without changing pitch.

        Uses time stretching via resampling (simple method).
        """
        try:
            import torchaudio
            import torch

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

            # Time stretch by resampling
            # Speed > 1 = shorter audio, Speed < 1 = longer audio
            stretched_length = int(len(audio) / speed)

            # Use interpolation for simple time stretching
            audio_tensor = torch.nn.functional.interpolate(
                audio_tensor.unsqueeze(0),
                size=stretched_length,
                mode='linear',
                align_corners=False
            ).squeeze()

            return audio_tensor.numpy()

        except ImportError:
            # Fallback: simple resampling
            from scipy import signal
            new_length = int(len(audio) / speed)
            return signal.resample(audio, new_length).astype(np.float32)

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio

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

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._tts is not None:
            del self._tts
            self._tts = None

        self._model_name = None
        self._current_voice = None
        self._loaded = False

        # Force garbage collection
        import gc
        gc.collect()

        # Clear MPS cache if available
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def get_info(self) -> dict:
        """Get engine information."""
        info = super().get_info()
        info.update({
            "model_name": self._model_name,
            "default_models": list(self.DEFAULT_MODELS.keys()),
        })
        return info
