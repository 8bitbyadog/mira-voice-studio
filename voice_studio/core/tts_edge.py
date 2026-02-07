"""
Edge TTS engine for Auto Voice.

Uses Microsoft Edge's free TTS service via edge-tts library.
Works with Python 3.12+ (unlike Coqui TTS).
"""

import asyncio
import numpy as np
from typing import List, Tuple
import io

from voice_studio.core.tts_engine import TTSEngine


# Common Edge TTS voices
EDGE_VOICES = {
    # English US
    "en-US-AriaNeural": "Aria (US Female)",
    "en-US-GuyNeural": "Guy (US Male)",
    "en-US-JennyNeural": "Jenny (US Female)",
    "en-US-ChristopherNeural": "Christopher (US Male)",
    "en-US-EricNeural": "Eric (US Male)",
    "en-US-MichelleNeural": "Michelle (US Female)",
    "en-US-RogerNeural": "Roger (US Male)",
    "en-US-SteffanNeural": "Steffan (US Male)",
    # English UK
    "en-GB-SoniaNeural": "Sonia (UK Female)",
    "en-GB-RyanNeural": "Ryan (UK Male)",
    # English Australia
    "en-AU-NatashaNeural": "Natasha (AU Female)",
    "en-AU-WilliamNeural": "William (AU Male)",
}

DEFAULT_VOICE = "en-US-AriaNeural"


class EdgeTTS(TTSEngine):
    """
    Text-to-speech using Microsoft Edge TTS.

    Features:
    - High quality neural voices
    - Multiple languages and accents
    - Speed and pitch control
    - Free to use (no API key needed)
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the Edge TTS engine.

        Args:
            sample_rate: Output sample rate (will resample if needed).
        """
        self.sample_rate = sample_rate
        self.voice = DEFAULT_VOICE
        self._loaded = False

    def load_voice(self, voice_name: str = "default") -> None:
        """
        Load a voice by name.

        Args:
            voice_name: Voice name or "default" for Aria.
        """
        if voice_name == "default":
            self.voice = DEFAULT_VOICE
        elif voice_name in EDGE_VOICES:
            self.voice = voice_name
        else:
            # Try to find a matching voice
            for voice_id, display_name in EDGE_VOICES.items():
                if voice_name.lower() in display_name.lower() or voice_name.lower() in voice_id.lower():
                    self.voice = voice_id
                    break
            else:
                # Default fallback
                self.voice = DEFAULT_VOICE

        self._loaded = True
        self._current_voice = self.voice

    def unload(self) -> None:
        """Unload the current voice (no-op for Edge TTS)."""
        self._loaded = False
        self._current_voice = None

    def synthesize(self, text: str, speed: float = 1.0) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech for text.

        Args:
            text: Text to synthesize.
            speed: Speaking speed (0.5 to 2.0).

        Returns:
            Tuple of (audio_data, sample_rate).
        """
        # Run async synthesis
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        audio_data, sr = loop.run_until_complete(
            self._synthesize_async(text, speed)
        )

        # Resample if needed
        if sr != self.sample_rate:
            audio_data = self._resample(audio_data, sr, self.sample_rate)
            sr = self.sample_rate

        return audio_data, sr

    async def _synthesize_async(self, text: str, speed: float) -> Tuple[np.ndarray, int]:
        """Async synthesis using edge-tts."""
        import edge_tts
        import soundfile as sf

        # Convert speed to rate string (e.g., "+20%" or "-10%")
        rate_percent = int((speed - 1.0) * 100)
        if rate_percent >= 0:
            rate_str = f"+{rate_percent}%"
        else:
            rate_str = f"{rate_percent}%"

        # Create communicate object
        communicate = edge_tts.Communicate(text, self.voice, rate=rate_str)

        # Collect audio bytes
        audio_bytes = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes.write(chunk["data"])

        # Convert MP3 to numpy array
        audio_bytes.seek(0)

        # Edge TTS outputs MP3, need to decode it
        from pydub import AudioSegment

        audio_segment = AudioSegment.from_mp3(audio_bytes)

        # Convert to numpy
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0  # Normalize to -1 to 1

        # Handle stereo
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)

        return samples, audio_segment.frame_rate

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
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

    def list_voices(self) -> List[str]:
        """List available voices."""
        return list(EDGE_VOICES.keys())

    def get_voice_info(self, voice_name: str) -> dict:
        """Get information about a voice."""
        display_name = EDGE_VOICES.get(voice_name, voice_name)
        return {
            "name": voice_name,
            "display_name": display_name,
            "language": voice_name.split("-")[0] + "-" + voice_name.split("-")[1] if "-" in voice_name else "en-US",
        }

    @property
    def is_loaded(self) -> bool:
        """Check if a voice is loaded."""
        return self._loaded

    @staticmethod
    def get_all_voices() -> dict:
        """Get all available Edge TTS voices."""
        return EDGE_VOICES.copy()
