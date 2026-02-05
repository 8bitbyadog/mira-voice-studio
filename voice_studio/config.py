"""
Configuration and paths for Mira Voice Studio.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class AudioConfig:
    """Audio output settings."""
    sample_rate: int = 44100
    bit_depth: int = 16
    channels: int = 1  # Mono
    pause_between_sentences_ms: int = 300
    default_speed: float = 1.0


@dataclass
class CaptionConfig:
    """Caption generation settings."""
    generate_sentence_srt: bool = True
    generate_word_srt: bool = True
    generate_vtt: bool = True


@dataclass
class WhisperConfig:
    """Whisper alignment settings."""
    model_size: str = "base"  # tiny, base, small, medium, large
    language: str = "en"


@dataclass
class Config:
    """Main configuration for Mira Voice Studio."""

    # Base paths
    app_dir: Path = field(default_factory=lambda: Path.home() / "mira_voice_studio")
    output_dir: Path = field(default_factory=lambda: Path.home() / "Videos" / "VO")

    # Sub-configurations
    audio: AudioConfig = field(default_factory=AudioConfig)
    captions: CaptionConfig = field(default_factory=CaptionConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)

    # Derived paths
    @property
    def training_dir(self) -> Path:
        return self.app_dir / "training"

    @property
    def models_dir(self) -> Path:
        return self.app_dir / "models"

    @property
    def pretrained_models_dir(self) -> Path:
        return self.models_dir / "pretrained"

    @property
    def custom_models_dir(self) -> Path:
        return self.models_dir / "custom"

    @property
    def training_inbox_dir(self) -> Path:
        return self.training_dir / "inbox"

    @property
    def training_recordings_dir(self) -> Path:
        return self.training_dir / "recordings"

    @property
    def training_datasets_dir(self) -> Path:
        return self.training_dir / "datasets"

    @property
    def settings_file(self) -> Path:
        return self.app_dir / "settings.json"

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            self.app_dir,
            self.output_dir,
            self.training_dir,
            self.training_inbox_dir,
            self.training_recordings_dir,
            self.training_dir / "processing",
            self.training_datasets_dir,
            self.models_dir,
            self.pretrained_models_dir,
            self.custom_models_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        """Save configuration to settings file."""
        self.ensure_directories()
        data = {
            "output_dir": str(self.output_dir),
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "bit_depth": self.audio.bit_depth,
                "channels": self.audio.channels,
                "pause_between_sentences_ms": self.audio.pause_between_sentences_ms,
                "default_speed": self.audio.default_speed,
            },
            "captions": {
                "generate_sentence_srt": self.captions.generate_sentence_srt,
                "generate_word_srt": self.captions.generate_word_srt,
                "generate_vtt": self.captions.generate_vtt,
            },
            "whisper": {
                "model_size": self.whisper.model_size,
                "language": self.whisper.language,
            },
        }
        with open(self.settings_file, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, settings_file: Optional[Path] = None) -> "Config":
        """Load configuration from settings file."""
        config = cls()

        if settings_file is None:
            settings_file = config.settings_file

        if settings_file.exists():
            with open(settings_file) as f:
                data = json.load(f)

            if "output_dir" in data:
                config.output_dir = Path(data["output_dir"])

            if "audio" in data:
                audio = data["audio"]
                config.audio = AudioConfig(
                    sample_rate=audio.get("sample_rate", 44100),
                    bit_depth=audio.get("bit_depth", 16),
                    channels=audio.get("channels", 1),
                    pause_between_sentences_ms=audio.get("pause_between_sentences_ms", 300),
                    default_speed=audio.get("default_speed", 1.0),
                )

            if "captions" in data:
                captions = data["captions"]
                config.captions = CaptionConfig(
                    generate_sentence_srt=captions.get("generate_sentence_srt", True),
                    generate_word_srt=captions.get("generate_word_srt", True),
                    generate_vtt=captions.get("generate_vtt", True),
                )

            if "whisper" in data:
                whisper = data["whisper"]
                config.whisper = WhisperConfig(
                    model_size=whisper.get("model_size", "base"),
                    language=whisper.get("language", "en"),
                )

        return config


# Default configuration instance
default_config = Config()
