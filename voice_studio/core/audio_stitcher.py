"""
Audio stitcher for Mira Voice Studio.

Combines individual sentence audio clips into a master audio file
with configurable pauses between sentences and automation support.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path

from voice_studio.core.tts_engine import TTSResult
from voice_studio.utils.audio_utils import generate_silence, save_audio

if TYPE_CHECKING:
    from voice_studio.core.automation import AutomationProject


@dataclass
class ChunkTiming:
    """Timing information for a single chunk in the master audio."""

    index: int  # 1-based sentence index
    start_time: float  # Start time in master (seconds)
    end_time: float  # End time in master (seconds)
    duration: float  # Duration (seconds)
    text: str  # Original text
    slug: str  # Filename slug
    success: bool = True  # Whether generation succeeded
    error: str = ""  # Error message if failed

    # Automation parameters applied (for manifest recording)
    automation: Optional[Dict[str, Any]] = None


@dataclass
class StitchedAudio:
    """Result of audio stitching operation."""

    audio_data: np.ndarray
    sample_rate: int
    duration_seconds: float
    chunk_timings: List[ChunkTiming]
    successful_count: int
    failed_count: int

    @property
    def total_chunks(self) -> int:
        return len(self.chunk_timings)


class AudioStitcher:
    """
    Stitch multiple audio clips into a master audio file.

    Features:
    - Configurable pause duration between sentences
    - Tracking of chunk timings for caption generation
    - Handles failed clips by inserting silence placeholders
    - Supports crossfade between clips (optional)
    """

    def __init__(
        self,
        pause_ms: int = 300,
        sample_rate: int = 44100,
        crossfade_ms: int = 0
    ):
        """
        Initialize the audio stitcher.

        Args:
            pause_ms: Pause duration between sentences in milliseconds.
            sample_rate: Target sample rate for output.
            crossfade_ms: Crossfade duration between clips (0 = no crossfade).
        """
        self.pause_ms = pause_ms
        self.sample_rate = sample_rate
        self.crossfade_ms = crossfade_ms

    def stitch(self, results: List[TTSResult]) -> StitchedAudio:
        """
        Stitch TTS results into a single audio file.

        Args:
            results: List of TTSResult objects from TTS generation.

        Returns:
            StitchedAudio with combined audio and timing information.
        """
        if not results:
            return StitchedAudio(
                audio_data=np.array([], dtype=np.float32),
                sample_rate=self.sample_rate,
                duration_seconds=0.0,
                chunk_timings=[],
                successful_count=0,
                failed_count=0,
            )

        # Generate pause silence
        pause_samples = generate_silence(self.pause_ms, self.sample_rate)

        # Collect audio segments and track timings
        segments: List[np.ndarray] = []
        chunk_timings: List[ChunkTiming] = []
        current_time = 0.0
        successful = 0
        failed = 0

        for result in results:
            sentence = result.sentence

            if result.success and result.audio_data is not None:
                # Use the actual audio
                audio = result.audio_data

                # Resample if needed
                if result.sample_rate != self.sample_rate:
                    audio = self._resample(audio, result.sample_rate, self.sample_rate)

                duration = len(audio) / self.sample_rate
                successful += 1

            else:
                # Insert silence placeholder for failed sentences
                # Use estimated duration based on word count
                estimated_duration = max(1.0, sentence.word_count * 0.4)
                audio = generate_silence(int(estimated_duration * 1000), self.sample_rate)
                duration = estimated_duration
                failed += 1

            # Record timing
            timing = ChunkTiming(
                index=sentence.index,
                start_time=current_time,
                end_time=current_time + duration,
                duration=duration,
                text=sentence.text,
                slug=sentence.slug,
                success=result.success,
                error=result.error_message if not result.success else "",
            )
            chunk_timings.append(timing)

            # Add audio segment
            segments.append(audio)

            # Update current time
            current_time += duration

            # Add pause (except after last segment)
            if result != results[-1]:
                segments.append(pause_samples)
                current_time += self.pause_ms / 1000

        # Concatenate all segments
        if self.crossfade_ms > 0 and len(segments) > 1:
            audio_data = self._concatenate_with_crossfade(segments)
        else:
            audio_data = np.concatenate(segments)

        total_duration = len(audio_data) / self.sample_rate

        return StitchedAudio(
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            duration_seconds=total_duration,
            chunk_timings=chunk_timings,
            successful_count=successful,
            failed_count=failed,
        )

    def stitch_with_automation(
        self,
        results: List[TTSResult],
        automation: "AutomationProject"
    ) -> StitchedAudio:
        """
        Stitch TTS results with per-sentence automation applied.

        Like Ableton's arrangement view - each sentence can have
        different speed, volume, pause, and crossfade settings.

        Args:
            results: List of TTSResult objects from TTS generation.
            automation: AutomationProject with lane and per-sentence settings.

        Returns:
            StitchedAudio with combined audio and timing information.
        """
        if not results:
            return StitchedAudio(
                audio_data=np.array([], dtype=np.float32),
                sample_rate=self.sample_rate,
                duration_seconds=0.0,
                chunk_timings=[],
                successful_count=0,
                failed_count=0,
            )

        total_sentences = len(results)
        segments: List[np.ndarray] = []
        chunk_timings: List[ChunkTiming] = []
        current_time = 0.0
        successful = 0
        failed = 0

        for i, result in enumerate(results):
            sentence = result.sentence

            # Get automation parameters for this sentence
            auto_params = automation.get_sentence_params(
                sentence.index,
                total_sentences
            )

            # Get pause and crossfade for this transition
            pause_before_ms = int(auto_params.get("pause_before_ms", 0))
            pause_after_ms = int(auto_params.get("pause_after_ms", self.pause_ms))
            crossfade_ms = int(auto_params.get("crossfade_ms", 0))
            volume = float(auto_params.get("volume", 1.0))

            # Add pause before (if not first sentence)
            if i > 0 and pause_before_ms > 0:
                pause_samples = generate_silence(pause_before_ms, self.sample_rate)
                segments.append(pause_samples)
                current_time += pause_before_ms / 1000

            if result.success and result.audio_data is not None:
                audio = result.audio_data

                # Resample if needed
                if result.sample_rate != self.sample_rate:
                    audio = self._resample(audio, result.sample_rate, self.sample_rate)

                # Apply volume automation
                if volume != 1.0:
                    audio = audio * volume

                duration = len(audio) / self.sample_rate
                successful += 1
            else:
                # Insert silence placeholder for failed sentences
                estimated_duration = max(1.0, sentence.word_count * 0.4)
                audio = generate_silence(int(estimated_duration * 1000), self.sample_rate)
                duration = estimated_duration
                failed += 1

            # Apply crossfade with previous segment
            if crossfade_ms > 0 and len(segments) > 0:
                audio = self._apply_crossfade_to_segment(
                    segments[-1], audio, crossfade_ms
                )
                # Update the previous segment (it was modified by crossfade)
                segments[-1] = segments[-1][:-int(crossfade_ms * self.sample_rate / 1000)]

            # Record timing with automation parameters
            timing = ChunkTiming(
                index=sentence.index,
                start_time=current_time,
                end_time=current_time + duration,
                duration=duration,
                text=sentence.text,
                slug=sentence.slug,
                success=result.success,
                error=result.error_message if not result.success else "",
                automation=auto_params,
            )
            chunk_timings.append(timing)

            # Add audio segment
            segments.append(audio)
            current_time += duration

            # Add pause after (if not last sentence)
            if i < len(results) - 1 and pause_after_ms > 0:
                pause_samples = generate_silence(pause_after_ms, self.sample_rate)
                segments.append(pause_samples)
                current_time += pause_after_ms / 1000

        # Concatenate all segments
        if segments:
            audio_data = np.concatenate(segments)
        else:
            audio_data = np.array([], dtype=np.float32)

        total_duration = len(audio_data) / self.sample_rate

        return StitchedAudio(
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            duration_seconds=total_duration,
            chunk_timings=chunk_timings,
            successful_count=successful,
            failed_count=failed,
        )

    def _apply_crossfade_to_segment(
        self,
        prev_segment: np.ndarray,
        curr_segment: np.ndarray,
        crossfade_ms: int
    ) -> np.ndarray:
        """Apply crossfade between the end of prev_segment and start of curr_segment."""
        crossfade_samples = int(crossfade_ms * self.sample_rate / 1000)

        if len(prev_segment) < crossfade_samples or len(curr_segment) < crossfade_samples:
            return curr_segment

        # Create fade curves
        fade_out = np.linspace(1, 0, crossfade_samples)
        fade_in = np.linspace(0, 1, crossfade_samples)

        # Get the overlapping portions
        prev_tail = prev_segment[-crossfade_samples:] * fade_out
        curr_head = curr_segment[:crossfade_samples] * fade_in

        # Create the crossfaded portion
        crossfaded = prev_tail + curr_head

        # Construct the result: crossfaded portion + rest of current segment
        return np.concatenate([crossfaded, curr_segment[crossfade_samples:]])

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
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

    def _concatenate_with_crossfade(
        self,
        segments: List[np.ndarray]
    ) -> np.ndarray:
        """
        Concatenate segments with crossfade transitions.

        Args:
            segments: List of audio segments.

        Returns:
            Combined audio with crossfades.
        """
        if not segments:
            return np.array([], dtype=np.float32)

        crossfade_samples = int(self.crossfade_ms * self.sample_rate / 1000)

        # Start with first segment
        result = segments[0].copy()

        for i in range(1, len(segments)):
            segment = segments[i]

            # Check if segments are long enough for crossfade
            if len(result) >= crossfade_samples and len(segment) >= crossfade_samples:
                # Create fade curves
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)

                # Apply crossfade
                result[-crossfade_samples:] *= fade_out
                segment[:crossfade_samples] *= fade_in

                # Overlap-add
                result[-crossfade_samples:] += segment[:crossfade_samples]
                result = np.concatenate([result, segment[crossfade_samples:]])
            else:
                # Just concatenate if too short
                result = np.concatenate([result, segment])

        return result

    def save_master(
        self,
        stitched: StitchedAudio,
        output_path: Path
    ) -> bool:
        """
        Save the stitched audio to a file.

        Args:
            stitched: StitchedAudio result.
            output_path: Path to save the audio file.

        Returns:
            True if successful.
        """
        return save_audio(
            output_path,
            stitched.audio_data,
            stitched.sample_rate,
            subtype="PCM_16"
        )

    def save_chunks(
        self,
        results: List[TTSResult],
        output_dir: Path
    ) -> List[Path]:
        """
        Save individual sentence audio files.

        Args:
            results: List of TTSResult objects.
            output_dir: Directory to save chunks.

        Returns:
            List of paths to saved files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for result in results:
            if result.success and result.audio_data is not None:
                filename = result.sentence.get_filename("wav")
                file_path = output_dir / filename

                # Resample if needed
                audio = result.audio_data
                if result.sample_rate != self.sample_rate:
                    audio = self._resample(audio, result.sample_rate, self.sample_rate)

                save_audio(file_path, audio, self.sample_rate)
                saved_paths.append(file_path)

        return saved_paths
