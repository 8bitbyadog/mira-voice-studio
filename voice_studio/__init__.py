"""
Mira Voice Studio

A local TTS application that generates voiceovers with synced SRT captions.
Optimized for Apple Silicon (M1/M2/M3) with MPS acceleration.
"""

__version__ = "0.1.0"
__author__ = "Mira Voice Studio"

from voice_studio.config import Config
from voice_studio.core.text_processor import TextProcessor
from voice_studio.core.tts_engine import TTSEngine
from voice_studio.core.audio_stitcher import AudioStitcher
from voice_studio.core.aligner import WhisperAligner
from voice_studio.core.srt_generator import SRTGenerator
from voice_studio.core.manifest import ManifestGenerator
from voice_studio.core.output_manager import OutputManager

__all__ = [
    "Config",
    "TextProcessor",
    "TTSEngine",
    "AudioStitcher",
    "WhisperAligner",
    "SRTGenerator",
    "ManifestGenerator",
    "OutputManager",
]
