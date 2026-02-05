"""
Custom exceptions for Mira Voice Studio.
"""


class VoiceStudioError(Exception):
    """Base exception for Mira Voice Studio."""
    pass


class TTSError(VoiceStudioError):
    """TTS generation failed."""
    pass


class AlignmentError(VoiceStudioError):
    """Whisper alignment failed."""
    pass


class ModelNotFoundError(VoiceStudioError):
    """Requested voice model not found."""
    pass


class RecordingError(VoiceStudioError):
    """Audio recording failed."""
    pass


class TrainingError(VoiceStudioError):
    """Model training failed."""
    pass


class SelectionError(VoiceStudioError):
    """Invalid I/O selection."""
    pass


class ConfigurationError(VoiceStudioError):
    """Configuration error."""
    pass
