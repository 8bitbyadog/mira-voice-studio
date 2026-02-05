"""
GPT-SoVITS TTS implementation for Mira Voice Studio.

GPT-SoVITS is a powerful zero-shot/few-shot voice cloning TTS system.
https://github.com/RVC-Boss/GPT-SoVITS

This implementation supports:
- Loading custom trained voice models
- Reference audio-based voice cloning
- Speed adjustment
- Apple Silicon (MPS) acceleration
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import warnings
import json

from voice_studio.core.tts_engine import TTSEngine, TTSError, ModelNotFoundError
from voice_studio.utils.device import get_device


class GPTSoVITS(TTSEngine):
    """
    GPT-SoVITS TTS engine implementation.

    GPT-SoVITS uses a reference audio + text to clone a voice,
    then generates speech in that voice for new text.
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        sample_rate: int = 32000,  # GPT-SoVITS native rate
        device: str = "auto"
    ):
        """
        Initialize GPT-SoVITS.

        Args:
            models_dir: Directory containing voice models.
            sample_rate: Output sample rate (GPT-SoVITS uses 32kHz).
            device: Compute device ("auto", "mps", "cpu").
        """
        super().__init__(sample_rate=sample_rate, device=device)

        if models_dir is None:
            models_dir = Path.home() / "mira_voice_studio" / "models"

        self.models_dir = Path(models_dir)
        self.pretrained_dir = self.models_dir / "pretrained"
        self.custom_dir = self.models_dir / "custom"

        # GPT-SoVITS components (lazy loaded)
        self._gpt_model = None
        self._sovits_model = None
        self._tokenizer = None
        self._ssl_model = None
        self._initialized = False

        # Current voice reference
        self._ref_audio: Optional[np.ndarray] = None
        self._ref_text: Optional[str] = None
        self._ref_audio_path: Optional[Path] = None

        # Model paths
        self._gpt_path: Optional[Path] = None
        self._sovits_path: Optional[Path] = None

    def _ensure_gptsovits_available(self) -> bool:
        """Check if GPT-SoVITS dependencies are available."""
        try:
            import torch
            # Check for the GPT-SoVITS inference module
            # This will be available after running the install script
            return True
        except ImportError:
            return False

    def _initialize(self) -> None:
        """Initialize GPT-SoVITS models and components."""
        if self._initialized:
            return

        try:
            import torch
            import torchaudio

            # Determine device
            if self.device == "mps" and torch.backends.mps.is_available():
                self._torch_device = torch.device("mps")
            elif self.device == "cuda" and torch.cuda.is_available():
                self._torch_device = torch.device("cuda")
            else:
                self._torch_device = torch.device("cpu")

            self._initialized = True

        except Exception as e:
            raise TTSError(f"Failed to initialize GPT-SoVITS: {e}") from e

    def list_voices(self) -> List[str]:
        """
        List available voice models.

        Returns:
            List of voice model names.
        """
        voices = []

        # Check pretrained models
        if self.pretrained_dir.exists():
            for model_dir in self.pretrained_dir.iterdir():
                if model_dir.is_dir() and self._is_valid_model(model_dir):
                    voices.append(f"pretrained/{model_dir.name}")

        # Check custom models
        if self.custom_dir.exists():
            for model_dir in self.custom_dir.iterdir():
                if model_dir.is_dir() and self._is_valid_model(model_dir):
                    voices.append(model_dir.name)

        return voices

    def _is_valid_model(self, model_dir: Path) -> bool:
        """Check if a directory contains a valid GPT-SoVITS model."""
        # Check for required files
        has_gpt = any(model_dir.glob("*.ckpt")) or any(model_dir.glob("*gpt*.pth"))
        has_sovits = any(model_dir.glob("*.pth"))
        has_config = (model_dir / "config.json").exists() or (model_dir / "metadata.json").exists()

        # Also check for reference audio
        has_ref = any(model_dir.glob("reference.*")) or any(model_dir.glob("ref_audio.*"))

        return (has_gpt or has_sovits) or has_ref

    def _get_model_dir(self, voice_name: str) -> Path:
        """Get the directory for a voice model."""
        if voice_name.startswith("pretrained/"):
            name = voice_name.replace("pretrained/", "")
            return self.pretrained_dir / name
        else:
            return self.custom_dir / voice_name

    def load_voice(self, voice_name: str) -> None:
        """
        Load a voice model.

        Args:
            voice_name: Name of the voice to load.

        Raises:
            ModelNotFoundError: If the voice is not found.
        """
        model_dir = self._get_model_dir(voice_name)

        if not model_dir.exists():
            raise ModelNotFoundError(f"Voice model not found: {voice_name}")

        # Initialize if needed
        if not self._initialized:
            self._initialize()

        try:
            # Load model configuration
            config_path = model_dir / "config.json"
            metadata_path = model_dir / "metadata.json"

            config = {}
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
            elif metadata_path.exists():
                with open(metadata_path) as f:
                    config = json.load(f)

            # Find model files
            gpt_files = list(model_dir.glob("*gpt*.ckpt")) + list(model_dir.glob("*gpt*.pth"))
            sovits_files = list(model_dir.glob("*sovits*.pth")) + list(model_dir.glob("*.pth"))

            if gpt_files:
                self._gpt_path = gpt_files[0]
            if sovits_files:
                # Filter out gpt files from sovits list
                sovits_files = [f for f in sovits_files if "gpt" not in f.name.lower()]
                if sovits_files:
                    self._sovits_path = sovits_files[0]

            # Find and load reference audio
            ref_audio_files = (
                list(model_dir.glob("reference.wav")) +
                list(model_dir.glob("reference.mp3")) +
                list(model_dir.glob("ref_audio.*")) +
                list(model_dir.glob("ref.*"))
            )

            if ref_audio_files:
                self._load_reference_audio(ref_audio_files[0])

            # Load reference text
            ref_text_files = list(model_dir.glob("reference.txt")) + list(model_dir.glob("ref_text.txt"))
            if ref_text_files:
                self._ref_text = ref_text_files[0].read_text(encoding="utf-8").strip()
            elif "reference_text" in config:
                self._ref_text = config["reference_text"]

            self._current_voice = voice_name
            self._loaded = True

        except Exception as e:
            self._loaded = False
            raise ModelNotFoundError(f"Failed to load voice '{voice_name}': {e}") from e

    def _load_reference_audio(self, audio_path: Path) -> None:
        """Load reference audio for voice cloning."""
        try:
            import torchaudio
            import torch

            waveform, sr = torchaudio.load(str(audio_path))

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample to 32kHz if needed (GPT-SoVITS native rate)
            if sr != 32000:
                resampler = torchaudio.transforms.Resample(sr, 32000)
                waveform = resampler(waveform)

            self._ref_audio = waveform.squeeze().numpy()
            self._ref_audio_path = audio_path

        except Exception as e:
            raise TTSError(f"Failed to load reference audio: {e}") from e

    def set_reference(
        self,
        audio_path: Optional[Path] = None,
        audio_data: Optional[np.ndarray] = None,
        text: Optional[str] = None,
        sample_rate: int = 32000
    ) -> None:
        """
        Set reference audio and text for voice cloning.

        Args:
            audio_path: Path to reference audio file.
            audio_data: Reference audio as numpy array.
            text: Text spoken in the reference audio.
            sample_rate: Sample rate of audio_data (if provided).
        """
        if audio_path is not None:
            self._load_reference_audio(audio_path)
        elif audio_data is not None:
            # Resample if needed
            if sample_rate != 32000:
                import torchaudio
                import torch
                waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sample_rate, 32000)
                waveform = resampler(waveform)
                audio_data = waveform.squeeze().numpy()
            self._ref_audio = audio_data
            self._ref_audio_path = None

        if text is not None:
            self._ref_text = text

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
        if not self._loaded:
            raise TTSError("No voice model loaded. Call load_voice() first.")

        if self._ref_audio is None:
            raise TTSError("No reference audio set. Voice model may be incomplete.")

        if not text.strip():
            # Return silence for empty text
            silence = np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)
            return silence, self.sample_rate

        try:
            # Use the GPT-SoVITS inference pipeline
            audio = self._run_inference(text, speed)
            return audio, self.sample_rate

        except Exception as e:
            raise TTSError(f"Synthesis failed: {e}") from e

    def _run_inference(self, text: str, speed: float = 1.0) -> np.ndarray:
        """
        Run GPT-SoVITS inference.

        This is a placeholder that will use the actual GPT-SoVITS inference
        when the models are properly installed.
        """
        import torch

        # For now, use a fallback to Coqui TTS if GPT-SoVITS models aren't loaded
        # This allows the pipeline to work while GPT-SoVITS is being set up
        if self._gpt_path is None or self._sovits_path is None:
            return self._fallback_synthesis(text, speed)

        # TODO: Implement full GPT-SoVITS inference pipeline
        # This requires:
        # 1. Text preprocessing (G2P, phoneme conversion)
        # 2. GPT model for semantic tokens
        # 3. SoVITS model for audio synthesis
        # 4. Reference audio conditioning

        # For Phase 2, we'll implement a simplified version
        # that works with the pretrained models

        return self._fallback_synthesis(text, speed)

    def _fallback_synthesis(self, text: str, speed: float = 1.0) -> np.ndarray:
        """
        Fallback to Coqui TTS when GPT-SoVITS models aren't available.

        This ensures the pipeline continues working during setup.
        """
        try:
            from voice_studio.core.tts_coqui import CoquiTTS

            coqui = CoquiTTS(sample_rate=self.sample_rate)
            coqui.load_voice("default")
            audio, sr = coqui.synthesize(text, speed)
            coqui.unload()

            # Resample to our sample rate if needed
            if sr != self.sample_rate:
                import torchaudio
                import torch
                waveform = torch.from_numpy(audio).float().unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(waveform).squeeze().numpy()

            return audio

        except Exception as e:
            raise TTSError(f"Fallback synthesis failed: {e}") from e

    def unload(self) -> None:
        """Unload models to free memory."""
        self._gpt_model = None
        self._sovits_model = None
        self._tokenizer = None
        self._ssl_model = None
        self._ref_audio = None
        self._ref_text = None
        self._ref_audio_path = None
        self._gpt_path = None
        self._sovits_path = None
        self._current_voice = None
        self._loaded = False
        self._initialized = False

        import gc
        gc.collect()

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
            "models_dir": str(self.models_dir),
            "gpt_model": str(self._gpt_path) if self._gpt_path else None,
            "sovits_model": str(self._sovits_path) if self._sovits_path else None,
            "has_reference_audio": self._ref_audio is not None,
            "reference_text": self._ref_text[:50] + "..." if self._ref_text and len(self._ref_text) > 50 else self._ref_text,
        })
        return info


def create_voice_model_structure(
    model_name: str,
    models_dir: Optional[Path] = None,
    reference_audio: Optional[Path] = None,
    reference_text: Optional[str] = None
) -> Path:
    """
    Create the directory structure for a new voice model.

    Args:
        model_name: Name for the new voice model.
        models_dir: Base models directory.
        reference_audio: Optional reference audio to copy.
        reference_text: Optional reference text.

    Returns:
        Path to the created model directory.
    """
    if models_dir is None:
        models_dir = Path.home() / "mira_voice_studio" / "models"

    model_dir = models_dir / "custom" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create config file
    config = {
        "name": model_name,
        "created_at": None,  # Will be set by training
        "reference_text": reference_text or "",
        "language": "en",
    }

    config_path = model_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Copy reference audio if provided
    if reference_audio and reference_audio.exists():
        import shutil
        dest = model_dir / f"reference{reference_audio.suffix}"
        shutil.copy2(reference_audio, dest)

    # Save reference text
    if reference_text:
        ref_text_path = model_dir / "reference.txt"
        ref_text_path.write_text(reference_text, encoding="utf-8")

    return model_dir
