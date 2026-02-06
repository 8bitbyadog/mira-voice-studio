"""
Training modules for Mira Voice Studio.

Provides:
- In-app audio recording with session management
- Training script generation (phoneme, emotional, conversational)
- Audio preprocessing (normalization, splitting, transcription)
- Dataset management
"""

from voice_studio.training.recorder import Recorder, RecordingSession, Take
from voice_studio.training.script_generator import ScriptGenerator, ScriptLine
from voice_studio.training.preprocessor import AudioPreprocessor, AudioClip, ProcessingResult
from voice_studio.training.dataset import DatasetManager, Dataset, DatasetClip

__all__ = [
    # Recording
    "Recorder",
    "RecordingSession",
    "Take",
    # Script generation
    "ScriptGenerator",
    "ScriptLine",
    # Preprocessing
    "AudioPreprocessor",
    "AudioClip",
    "ProcessingResult",
    # Dataset management
    "DatasetManager",
    "Dataset",
    "DatasetClip",
]
