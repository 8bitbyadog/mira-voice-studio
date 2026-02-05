"""
Core modules for Mira Voice Studio.

Contains the main processing pipeline:
- Text processing and sentence splitting
- TTS engine abstraction
- Audio stitching
- Whisper alignment
- SRT/VTT generation
- Manifest creation
- Output management
"""

from voice_studio.core.text_processor import TextProcessor, Sentence
from voice_studio.core.tts_engine import TTSEngine, TTSResult
from voice_studio.core.tts_coqui import CoquiTTS
from voice_studio.core.audio_stitcher import AudioStitcher, StitchedAudio
from voice_studio.core.aligner import WhisperAligner, AlignmentResult, WordTiming
from voice_studio.core.srt_generator import SRTGenerator
from voice_studio.core.manifest import ManifestGenerator, ChunkInfo
from voice_studio.core.output_manager import OutputManager
from voice_studio.core.selection import Selection, SelectionExporter

__all__ = [
    # Text processing
    "TextProcessor",
    "Sentence",
    # TTS
    "TTSEngine",
    "TTSResult",
    "CoquiTTS",
    # Audio
    "AudioStitcher",
    "StitchedAudio",
    # Alignment
    "WhisperAligner",
    "AlignmentResult",
    "WordTiming",
    # Output
    "SRTGenerator",
    "ManifestGenerator",
    "ChunkInfo",
    "OutputManager",
    # Selection
    "Selection",
    "SelectionExporter",
]
