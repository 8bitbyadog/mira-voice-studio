"""
Audio preprocessing for Mira Voice Studio training.

Handles:
- Audio normalization
- Silence detection and splitting
- Transcription with Whisper
- Dataset preparation
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import json
from datetime import datetime

from voice_studio.utils.audio_utils import (
    normalize_audio,
    get_audio_duration,
    load_audio,
    save_audio,
)


@dataclass
class AudioClip:
    """A single audio clip for training."""

    index: int
    audio_data: np.ndarray
    sample_rate: int
    duration: float
    transcript: str
    source_file: str
    start_time: float = 0.0  # In source file
    end_time: float = 0.0
    approved: bool = True

    @property
    def filename(self) -> str:
        return f"{self.index:04d}.wav"


@dataclass
class ProcessingResult:
    """Result of preprocessing audio files."""

    clips: List[AudioClip]
    total_duration: float
    source_files: List[str]
    errors: List[str] = field(default_factory=list)

    @property
    def clip_count(self) -> int:
        return len(self.clips)

    @property
    def approved_count(self) -> int:
        return sum(1 for c in self.clips if c.approved)


class AudioPreprocessor:
    """
    Preprocess audio for voice model training.

    Features:
    - Normalize audio levels
    - Split on silence into clips
    - Transcribe with Whisper
    - Filter clips by duration
    - Prepare training dataset
    """

    def __init__(
        self,
        target_sample_rate: int = 44100,
        min_clip_duration: float = 1.0,
        max_clip_duration: float = 15.0,
        silence_threshold_db: float = -40.0,
        min_silence_duration: float = 0.3,
    ):
        """
        Initialize the preprocessor.

        Args:
            target_sample_rate: Output sample rate.
            min_clip_duration: Minimum clip length in seconds.
            max_clip_duration: Maximum clip length in seconds.
            silence_threshold_db: Threshold for silence detection.
            min_silence_duration: Minimum silence gap for splitting.
        """
        self.target_sample_rate = target_sample_rate
        self.min_clip_duration = min_clip_duration
        self.max_clip_duration = max_clip_duration
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_duration = min_silence_duration

        self._whisper_model = None

    def _load_whisper(self) -> None:
        """Load Whisper model for transcription."""
        if self._whisper_model is None:
            import whisper
            self._whisper_model = whisper.load_model("base")

    def _unload_whisper(self) -> None:
        """Unload Whisper to free memory."""
        if self._whisper_model is not None:
            del self._whisper_model
            self._whisper_model = None

            import gc
            gc.collect()

    def detect_silence(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[Tuple[float, float]]:
        """
        Detect silence regions in audio.

        Args:
            audio: Audio data.
            sample_rate: Sample rate.

        Returns:
            List of (start, end) tuples for silence regions.
        """
        # Calculate RMS in windows
        window_size = int(sample_rate * 0.02)  # 20ms windows
        hop_size = window_size // 2

        threshold = 10 ** (self.silence_threshold_db / 20)
        silence_regions = []
        in_silence = False
        silence_start = 0

        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))

            time = i / sample_rate

            if rms < threshold:
                if not in_silence:
                    silence_start = time
                    in_silence = True
            else:
                if in_silence:
                    silence_end = time
                    duration = silence_end - silence_start
                    if duration >= self.min_silence_duration:
                        silence_regions.append((silence_start, silence_end))
                    in_silence = False

        # Handle trailing silence
        if in_silence:
            silence_end = len(audio) / sample_rate
            duration = silence_end - silence_start
            if duration >= self.min_silence_duration:
                silence_regions.append((silence_start, silence_end))

        return silence_regions

    def split_on_silence(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        Split audio on silence regions.

        Args:
            audio: Audio data.
            sample_rate: Sample rate.

        Returns:
            List of (audio_segment, start_time, end_time).
        """
        silence_regions = self.detect_silence(audio, sample_rate)

        if not silence_regions:
            # No silence found, return whole audio if within limits
            duration = len(audio) / sample_rate
            if self.min_clip_duration <= duration <= self.max_clip_duration:
                return [(audio, 0.0, duration)]
            return []

        segments = []
        prev_end = 0.0

        for silence_start, silence_end in silence_regions:
            # Extract segment before this silence
            start_sample = int(prev_end * sample_rate)
            end_sample = int(silence_start * sample_rate)

            if end_sample > start_sample:
                segment = audio[start_sample:end_sample]
                duration = len(segment) / sample_rate

                if self.min_clip_duration <= duration <= self.max_clip_duration:
                    segments.append((segment, prev_end, silence_start))
                elif duration > self.max_clip_duration:
                    # Split long segment into chunks
                    chunk_samples = int(self.max_clip_duration * sample_rate)
                    for i in range(0, len(segment), chunk_samples):
                        chunk = segment[i:i + chunk_samples]
                        chunk_duration = len(chunk) / sample_rate
                        if chunk_duration >= self.min_clip_duration:
                            chunk_start = prev_end + (i / sample_rate)
                            chunk_end = chunk_start + chunk_duration
                            segments.append((chunk, chunk_start, chunk_end))

            prev_end = silence_end

        # Handle final segment
        if prev_end < len(audio) / sample_rate:
            start_sample = int(prev_end * sample_rate)
            segment = audio[start_sample:]
            duration = len(segment) / sample_rate

            if self.min_clip_duration <= duration <= self.max_clip_duration:
                segments.append((segment, prev_end, prev_end + duration))

        return segments

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """
        Transcribe audio using Whisper.

        Args:
            audio: Audio data.
            sample_rate: Sample rate.

        Returns:
            Transcription text.
        """
        self._load_whisper()

        # Resample to 16kHz for Whisper
        if sample_rate != 16000:
            import torchaudio
            import torch

            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_16k = resampler(audio_tensor).squeeze().numpy()
        else:
            audio_16k = audio

        # Transcribe
        result = self._whisper_model.transcribe(
            audio_16k,
            language="en",
            verbose=False,
        )

        return result.get("text", "").strip()

    def process_file(
        self,
        audio_path: Path,
        transcribe: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[AudioClip]:
        """
        Process a single audio file into training clips.

        Args:
            audio_path: Path to audio file.
            transcribe: Whether to transcribe clips.
            progress_callback: Optional callback(message, progress).

        Returns:
            List of AudioClip objects.
        """
        audio_path = Path(audio_path)

        if progress_callback:
            progress_callback(f"Loading {audio_path.name}...", 0.0)

        # Load audio
        audio, sample_rate = load_audio(
            audio_path,
            target_sample_rate=self.target_sample_rate,
            mono=True
        )

        if progress_callback:
            progress_callback("Normalizing audio...", 0.1)

        # Normalize
        audio = normalize_audio(audio, target_db=-3.0)

        if progress_callback:
            progress_callback("Splitting on silence...", 0.2)

        # Split on silence
        segments = self.split_on_silence(audio, self.target_sample_rate)

        if progress_callback:
            progress_callback(f"Found {len(segments)} segments", 0.3)

        # Create clips
        clips = []
        for i, (segment_audio, start_time, end_time) in enumerate(segments):
            if progress_callback:
                progress = 0.3 + (0.7 * i / len(segments))
                progress_callback(f"Processing clip {i + 1}/{len(segments)}...", progress)

            transcript = ""
            if transcribe:
                transcript = self.transcribe(segment_audio, self.target_sample_rate)

            clip = AudioClip(
                index=i + 1,
                audio_data=segment_audio,
                sample_rate=self.target_sample_rate,
                duration=len(segment_audio) / self.target_sample_rate,
                transcript=transcript,
                source_file=audio_path.name,
                start_time=start_time,
                end_time=end_time,
            )
            clips.append(clip)

        if progress_callback:
            progress_callback("Complete", 1.0)

        return clips

    def process_files(
        self,
        audio_paths: List[Path],
        transcribe: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> ProcessingResult:
        """
        Process multiple audio files.

        Args:
            audio_paths: List of paths to audio files.
            transcribe: Whether to transcribe clips.
            progress_callback: Optional callback(message, progress).

        Returns:
            ProcessingResult with all clips.
        """
        all_clips = []
        errors = []
        source_files = []

        for i, audio_path in enumerate(audio_paths):
            file_progress = i / len(audio_paths)

            if progress_callback:
                progress_callback(
                    f"Processing file {i + 1}/{len(audio_paths)}: {audio_path.name}",
                    file_progress
                )

            try:
                clips = self.process_file(audio_path, transcribe=transcribe)

                # Re-index clips
                for clip in clips:
                    clip.index = len(all_clips) + 1
                    all_clips.append(clip)

                source_files.append(str(audio_path))

            except Exception as e:
                errors.append(f"{audio_path.name}: {str(e)}")

        # Unload Whisper to free memory
        self._unload_whisper()

        total_duration = sum(c.duration for c in all_clips)

        return ProcessingResult(
            clips=all_clips,
            total_duration=total_duration,
            source_files=source_files,
            errors=errors,
        )

    def save_dataset(
        self,
        result: ProcessingResult,
        output_dir: Path,
        dataset_name: str = "dataset"
    ) -> Path:
        """
        Save processed clips as a training dataset.

        Args:
            result: ProcessingResult from processing.
            output_dir: Output directory.
            dataset_name: Name for the dataset.

        Returns:
            Path to dataset directory.
        """
        dataset_dir = Path(output_dir) / dataset_name
        wavs_dir = dataset_dir / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)

        # Save audio files
        transcripts = []
        for clip in result.clips:
            if not clip.approved:
                continue

            wav_path = wavs_dir / clip.filename
            save_audio(wav_path, clip.audio_data, clip.sample_rate)

            transcripts.append(f"{clip.filename}|{clip.transcript}")

        # Save transcripts
        transcripts_path = dataset_dir / "transcripts.txt"
        transcripts_path.write_text("\n".join(transcripts), encoding="utf-8")

        # Save metadata
        metadata = {
            "name": dataset_name,
            "created_at": datetime.now().isoformat(),
            "total_duration_seconds": result.total_duration,
            "clip_count": result.clip_count,
            "approved_count": result.approved_count,
            "sample_rate": self.target_sample_rate,
            "source_files": result.source_files,
            "clips": [
                {
                    "index": c.index,
                    "filename": c.filename,
                    "duration": c.duration,
                    "transcript": c.transcript,
                    "source_file": c.source_file,
                    "approved": c.approved,
                }
                for c in result.clips
            ]
        }

        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return dataset_dir
