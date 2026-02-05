"""
Persistent settings storage for Mira Voice Studio.

Uses JSON file storage for cross-session persistence.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict
import threading


@dataclass
class AppSettings:
    """Application settings with defaults."""

    # Paths
    output_dir: str = ""
    training_dir: str = ""
    models_dir: str = ""

    # Audio settings
    sample_rate: int = 44100
    pause_between_sentences_ms: int = 300
    default_speed: float = 1.0

    # Caption settings
    generate_sentence_srt: bool = True
    generate_word_srt: bool = True
    generate_vtt: bool = True

    # Whisper settings
    whisper_model_size: str = "base"  # tiny, base, small, medium, large
    whisper_language: str = "en"

    # Recording settings
    recording_device: str = ""  # Empty = system default
    recording_format: str = "wav"

    # UI settings
    show_tooltips: bool = True
    enable_keyboard_shortcuts: bool = True
    show_first_run_tutorial: bool = True

    # Last used values (for convenience)
    last_voice: str = ""
    last_output_folder: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppSettings":
        """Create settings from dictionary."""
        # Filter out unknown keys
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


class SettingsManager:
    """
    Thread-safe settings manager with automatic persistence.

    Usage:
        settings = SettingsManager()
        settings.load()

        # Read settings
        rate = settings.get("sample_rate")

        # Update settings
        settings.set("sample_rate", 48000)

        # Or update multiple at once
        settings.update(sample_rate=48000, default_speed=1.1)

        # Settings are auto-saved on update, but you can force save
        settings.save()
    """

    _instance: Optional["SettingsManager"] = None
    _lock = threading.Lock()

    def __new__(cls, settings_path: Optional[Path] = None):
        """Singleton pattern for settings manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, settings_path: Optional[Path] = None):
        """Initialize the settings manager."""
        if self._initialized:
            return

        self._lock = threading.Lock()

        if settings_path is None:
            self._settings_path = Path.home() / "mira_voice_studio" / "settings.json"
        else:
            self._settings_path = Path(settings_path)

        self._settings = AppSettings()
        self._initialized = True

    @property
    def settings_path(self) -> Path:
        """Get the settings file path."""
        return self._settings_path

    @property
    def settings(self) -> AppSettings:
        """Get the current settings object."""
        return self._settings

    def load(self) -> bool:
        """
        Load settings from file.

        Returns:
            True if settings were loaded, False if using defaults.
        """
        with self._lock:
            if not self._settings_path.exists():
                # Initialize with defaults
                self._set_default_paths()
                return False

            try:
                with open(self._settings_path, "r") as f:
                    data = json.load(f)
                self._settings = AppSettings.from_dict(data)

                # Ensure paths are set
                if not self._settings.output_dir:
                    self._set_default_paths()

                return True
            except (json.JSONDecodeError, IOError):
                self._set_default_paths()
                return False

    def _set_default_paths(self):
        """Set default paths based on home directory."""
        home = Path.home()
        app_dir = home / "mira_voice_studio"

        if not self._settings.output_dir:
            self._settings.output_dir = str(home / "Videos" / "VO")
        if not self._settings.training_dir:
            self._settings.training_dir = str(app_dir / "training")
        if not self._settings.models_dir:
            self._settings.models_dir = str(app_dir / "models")

    def save(self) -> bool:
        """
        Save settings to file.

        Returns:
            True if saved successfully, False otherwise.
        """
        with self._lock:
            try:
                self._settings_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._settings_path, "w") as f:
                    json.dump(self._settings.to_dict(), f, indent=2)
                return True
            except IOError:
                return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value.

        Args:
            key: Setting name.
            default: Default value if key doesn't exist.

        Returns:
            The setting value.
        """
        with self._lock:
            return getattr(self._settings, key, default)

    def set(self, key: str, value: Any, auto_save: bool = True) -> None:
        """
        Set a setting value.

        Args:
            key: Setting name.
            value: New value.
            auto_save: Whether to save immediately.
        """
        with self._lock:
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
                if auto_save:
                    self.save()

    def update(self, auto_save: bool = True, **kwargs) -> None:
        """
        Update multiple settings at once.

        Args:
            auto_save: Whether to save immediately.
            **kwargs: Setting key-value pairs.
        """
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._settings, key):
                    setattr(self._settings, key, value)
            if auto_save:
                self.save()

    def reset_to_defaults(self, auto_save: bool = True) -> None:
        """
        Reset all settings to defaults.

        Args:
            auto_save: Whether to save immediately.
        """
        with self._lock:
            self._settings = AppSettings()
            self._set_default_paths()
            if auto_save:
                self.save()


# Convenience function for quick access
def get_settings() -> SettingsManager:
    """Get the singleton settings manager instance."""
    manager = SettingsManager()
    manager.load()
    return manager
