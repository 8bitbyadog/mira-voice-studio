"""
Audio recording for Mira Voice Studio.

Provides in-app audio recording with:
- Real-time level metering
- Waveform visualization data
- Session management
- Take organization
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from typing import Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import threading
import queue
import time


@dataclass
class Take:
    """A single recording take."""

    index: int
    audio_data: np.ndarray
    sample_rate: int
    duration: float
    timestamp: datetime
    script_text: str
    script_index: int
    file_path: Optional[Path] = None
    approved: bool = True

    def save(self, directory: Path) -> Path:
        """Save the take to a file."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        filename = f"take_{self.index:03d}_{self.timestamp.strftime('%H%M%S')}.wav"
        self.file_path = directory / filename

        sf.write(str(self.file_path), self.audio_data, self.sample_rate)
        return self.file_path


@dataclass
class RecordingSession:
    """A recording session containing multiple takes."""

    name: str
    created_at: datetime = field(default_factory=datetime.now)
    takes: List[Take] = field(default_factory=list)
    sample_rate: int = 44100

    @property
    def total_duration(self) -> float:
        """Total duration of approved takes."""
        return sum(t.duration for t in self.takes if t.approved)

    @property
    def approved_takes(self) -> List[Take]:
        """Get only approved takes."""
        return [t for t in self.takes if t.approved]

    @property
    def take_count(self) -> int:
        """Number of takes in session."""
        return len(self.takes)

    def add_take(self, take: Take) -> None:
        """Add a take to the session."""
        self.takes.append(take)

    def delete_take(self, index: int) -> bool:
        """Mark a take as not approved (soft delete)."""
        for take in self.takes:
            if take.index == index:
                take.approved = False
                return True
        return False

    def get_session_dir(self, base_dir: Path) -> Path:
        """Get the directory for this session."""
        date_str = self.created_at.strftime("%Y-%m-%d")
        return base_dir / f"session_{date_str}_{self.name}"


class Recorder:
    """
    Audio recorder with real-time monitoring.

    Features:
    - List available input devices
    - Real-time level metering
    - Recording with callback for waveform updates
    - Session and take management
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 1,
        device: Optional[str] = None
    ):
        """
        Initialize the recorder.

        Args:
            sample_rate: Recording sample rate.
            channels: Number of channels (1 = mono).
            device: Input device name (None = default).
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self._device = device

        # Recording state
        self._recording = False
        self._monitoring = False
        self._audio_queue: queue.Queue = queue.Queue()
        self._recorded_frames: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None

        # Level monitoring
        self._current_level_db: float = -60.0
        self._peak_level_db: float = -60.0

        # Session management
        self._current_session: Optional[RecordingSession] = None
        self._take_counter: int = 0

    @staticmethod
    def get_input_devices() -> List[dict]:
        """
        Get list of available audio input devices.

        Returns:
            List of device info dictionaries.
        """
        devices = []
        for i, device in enumerate(sd.query_devices()):
            if device['max_input_channels'] > 0:
                devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate'],
                })
        return devices

    @staticmethod
    def get_default_input_device() -> Optional[str]:
        """Get the name of the default input device."""
        try:
            device = sd.query_devices(kind='input')
            return device['name']
        except Exception:
            return None

    def set_device(self, device_name: Optional[str]) -> None:
        """Set the input device by name."""
        self._device = device_name

    def _get_device_index(self) -> Optional[int]:
        """Get device index from name."""
        if self._device is None:
            return None

        for i, device in enumerate(sd.query_devices()):
            if device['name'] == self._device:
                return i
        return None

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags
    ) -> None:
        """Callback for audio stream."""
        # Calculate level
        if len(indata) > 0:
            rms = np.sqrt(np.mean(indata ** 2))
            if rms > 0:
                self._current_level_db = 20 * np.log10(rms)
                self._peak_level_db = max(self._peak_level_db, self._current_level_db)
            else:
                self._current_level_db = -60.0

        # Store frames if recording
        if self._recording:
            self._audio_queue.put(indata.copy())

    def start_monitoring(self) -> bool:
        """
        Start monitoring audio input (for level meter).

        Returns:
            True if started successfully.
        """
        if self._monitoring:
            return True

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self._get_device_index(),
                callback=self._audio_callback,
                blocksize=1024,
            )
            self._stream.start()
            self._monitoring = True
            return True
        except Exception as e:
            print(f"Failed to start monitoring: {e}")
            return False

    def stop_monitoring(self) -> None:
        """Stop monitoring audio input."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._monitoring = False

    def get_level(self) -> Tuple[float, float]:
        """
        Get current audio level.

        Returns:
            Tuple of (current_db, peak_db).
        """
        return self._current_level_db, self._peak_level_db

    def reset_peak(self) -> None:
        """Reset peak level meter."""
        self._peak_level_db = -60.0

    def get_level_indicator(self) -> Tuple[str, str]:
        """
        Get level quality indicator.

        Returns:
            Tuple of (status, message).
        """
        level = self._current_level_db

        if level < -40:
            return "too_quiet", "Too quiet - move closer to mic"
        elif level > -6:
            return "too_loud", "Too loud - move away from mic"
        elif level > -3:
            return "clipping", "Clipping! Move away from mic"
        else:
            return "good", "Good level"

    def start_session(self, name: str = "recording") -> RecordingSession:
        """
        Start a new recording session.

        Args:
            name: Session name.

        Returns:
            The new RecordingSession.
        """
        self._current_session = RecordingSession(
            name=name,
            sample_rate=self.sample_rate,
        )
        self._take_counter = 0
        return self._current_session

    def start_recording(self) -> bool:
        """
        Start recording audio.

        Returns:
            True if started successfully.
        """
        if self._recording:
            return False

        # Start monitoring if not already
        if not self._monitoring:
            if not self.start_monitoring():
                return False

        # Clear previous recording
        self._recorded_frames = []
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        self._recording = True

        # Start collecting frames in background
        self._collect_thread = threading.Thread(target=self._collect_frames)
        self._collect_thread.start()

        return True

    def _collect_frames(self) -> None:
        """Background thread to collect audio frames."""
        while self._recording:
            try:
                frame = self._audio_queue.get(timeout=0.1)
                self._recorded_frames.append(frame)
            except queue.Empty:
                continue

    def stop_recording(
        self,
        script_text: str = "",
        script_index: int = 0
    ) -> Optional[Take]:
        """
        Stop recording and create a take.

        Args:
            script_text: The text being read.
            script_index: Index of the script line.

        Returns:
            The recorded Take, or None if failed.
        """
        if not self._recording:
            return None

        self._recording = False

        # Wait for collection thread
        if hasattr(self, '_collect_thread'):
            self._collect_thread.join(timeout=1.0)

        # Combine frames
        if not self._recorded_frames:
            return None

        audio_data = np.concatenate(self._recorded_frames)
        audio_data = audio_data.flatten()

        # Create take
        self._take_counter += 1
        take = Take(
            index=self._take_counter,
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            duration=len(audio_data) / self.sample_rate,
            timestamp=datetime.now(),
            script_text=script_text,
            script_index=script_index,
        )

        # Add to session
        if self._current_session is not None:
            self._current_session.add_take(take)

        return take

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    def get_recording_duration(self) -> float:
        """Get duration of current recording in progress."""
        if not self._recording:
            return 0.0
        return len(self._recorded_frames) * 1024 / self.sample_rate

    def get_session(self) -> Optional[RecordingSession]:
        """Get the current session."""
        return self._current_session

    def save_session(self, base_dir: Path) -> Path:
        """
        Save the current session to disk.

        Args:
            base_dir: Base directory for recordings.

        Returns:
            Path to session directory.
        """
        if self._current_session is None:
            raise ValueError("No active session")

        session_dir = self._current_session.get_session_dir(base_dir)
        raw_dir = session_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Save all approved takes
        for take in self._current_session.approved_takes:
            take.save(raw_dir)

        # Save session metadata
        import json
        metadata = {
            "name": self._current_session.name,
            "created_at": self._current_session.created_at.isoformat(),
            "sample_rate": self._current_session.sample_rate,
            "total_duration": self._current_session.total_duration,
            "take_count": self._current_session.take_count,
            "approved_count": len(self._current_session.approved_takes),
            "takes": [
                {
                    "index": t.index,
                    "duration": t.duration,
                    "script_text": t.script_text,
                    "script_index": t.script_index,
                    "approved": t.approved,
                    "file": t.file_path.name if t.file_path else None,
                }
                for t in self._current_session.takes
            ]
        }

        metadata_path = session_dir / "session.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return session_dir

    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_monitoring()
        self._current_session = None
