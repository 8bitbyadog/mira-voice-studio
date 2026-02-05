"""
I/O Selection and export for Mira Voice Studio.

Handles exporting portions of the master audio based on In/Out points.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from voice_studio.core.audio_stitcher import ChunkTiming
from voice_studio.core.srt_generator import SRTGenerator, CaptionEntry
from voice_studio.core.manifest import ManifestGenerator
from voice_studio.utils.audio_utils import save_audio


@dataclass
class Selection:
    """Represents an In/Out selection on the timeline."""

    in_time: float   # Start time in seconds
    out_time: float  # End time in seconds

    def __post_init__(self):
        """Validate selection."""
        if self.in_time < 0:
            self.in_time = 0
        if self.out_time <= self.in_time:
            raise ValueError("Out time must be greater than In time")

    @property
    def duration(self) -> float:
        """Get selection duration in seconds."""
        return self.out_time - self.in_time

    def to_folder_name(self) -> str:
        """
        Generate folder name for this selection.

        Returns:
            Folder name like 'selection_00m32s_01m45s'
        """
        in_str = self._format_time(self.in_time)
        out_str = self._format_time(self.out_time)
        return f"selection_{in_str}_{out_str}"

    def _format_time(self, seconds: float) -> str:
        """Format time as XXmYYs."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}m{secs:02d}s"

    def to_timecode(self) -> Tuple[str, str]:
        """
        Get formatted timecodes for display.

        Returns:
            Tuple of (in_timecode, out_timecode) like ("00:00:32", "00:01:45")
        """
        def fmt(s: float) -> str:
            hours = int(s // 3600)
            minutes = int((s % 3600) // 60)
            secs = int(s % 60)
            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"
            return f"{minutes:02d}:{secs:02d}"

        return fmt(self.in_time), fmt(self.out_time)

    def contains_time(self, time: float) -> bool:
        """Check if a time falls within the selection."""
        return self.in_time <= time <= self.out_time

    def overlaps_chunk(self, chunk: ChunkTiming) -> bool:
        """Check if a chunk overlaps with the selection."""
        return not (chunk.end_time < self.in_time or chunk.start_time > self.out_time)

    @classmethod
    def from_string(cls, time_str: str) -> float:
        """
        Parse a time string into seconds.

        Accepts formats:
        - "32" or "32.5" (seconds)
        - "1:32" or "01:32" (MM:SS)
        - "1:01:32" (HH:MM:SS)

        Args:
            time_str: Time string to parse.

        Returns:
            Time in seconds.
        """
        time_str = time_str.strip()

        if ":" in time_str:
            parts = time_str.split(":")
            if len(parts) == 2:
                # MM:SS
                minutes, seconds = parts
                return int(minutes) * 60 + float(seconds)
            elif len(parts) == 3:
                # HH:MM:SS
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        else:
            # Just seconds
            return float(time_str)


class SelectionExporter:
    """
    Export audio and captions for a selection.

    Takes a portion of the master audio and generates:
    - Audio file for just the selection
    - SRT with timestamps starting at 00:00:00
    - Plain text of sentences in the selection
    """

    def __init__(self):
        """Initialize the selection exporter."""
        self.srt_generator = SRTGenerator()
        self.manifest_generator = ManifestGenerator()

    def export(
        self,
        selection: Selection,
        master_audio: np.ndarray,
        sample_rate: int,
        chunk_timings: List[ChunkTiming],
        output_dir: Path,
        manifest_path: Optional[Path] = None
    ) -> Path:
        """
        Export a selection.

        Args:
            selection: The In/Out selection.
            master_audio: Full master audio data.
            sample_rate: Audio sample rate.
            chunk_timings: List of chunk timings from the master.
            output_dir: Base output directory (will create selections/ subfolder).
            manifest_path: Optional path to manifest to update.

        Returns:
            Path to the selection folder.
        """
        # Create selection folder
        selections_dir = output_dir / "selections"
        selection_folder = selections_dir / selection.to_folder_name()
        selection_folder.mkdir(parents=True, exist_ok=True)

        # Extract audio for selection
        start_sample = int(selection.in_time * sample_rate)
        end_sample = int(selection.out_time * sample_rate)

        # Clamp to audio bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(master_audio), end_sample)

        selection_audio = master_audio[start_sample:end_sample]

        # Save audio
        audio_path = selection_folder / "selection.wav"
        save_audio(audio_path, selection_audio, sample_rate)

        # Find chunks that overlap with selection
        overlapping_chunks = [
            c for c in chunk_timings
            if selection.overlaps_chunk(c)
        ]

        # Generate SRT with reset timestamps
        entries = []
        for i, chunk in enumerate(overlapping_chunks, start=1):
            # Calculate new timestamps relative to selection start
            new_start = max(0, chunk.start_time - selection.in_time)
            new_end = min(selection.duration, chunk.end_time - selection.in_time)

            entries.append(CaptionEntry(
                index=i,
                start_time=new_start,
                end_time=new_end,
                text=chunk.text,
            ))

        srt_path = selection_folder / "selection.srt"
        self.srt_generator.write_srt(entries, srt_path)

        # Generate plain text
        text_content = "\n".join(chunk.text for chunk in overlapping_chunks)
        txt_path = selection_folder / "selection.txt"
        txt_path.write_text(text_content, encoding="utf-8")

        # Update manifest if provided
        if manifest_path and manifest_path.exists():
            relative_folder = f"selections/{selection.to_folder_name()}/"
            self.manifest_generator.add_selection_export(
                manifest_path,
                in_time=selection.in_time,
                out_time=selection.out_time,
                folder=relative_folder,
            )

        return selection_folder

    def get_chunks_in_selection(
        self,
        selection: Selection,
        chunk_timings: List[ChunkTiming]
    ) -> List[ChunkTiming]:
        """
        Get chunks that fall within a selection.

        Args:
            selection: The selection range.
            chunk_timings: All chunk timings.

        Returns:
            List of chunks that overlap with the selection.
        """
        return [c for c in chunk_timings if selection.overlaps_chunk(c)]

    def get_selection_text(
        self,
        selection: Selection,
        chunk_timings: List[ChunkTiming]
    ) -> str:
        """
        Get the text content for a selection.

        Args:
            selection: The selection range.
            chunk_timings: All chunk timings.

        Returns:
            Combined text of sentences in the selection.
        """
        chunks = self.get_chunks_in_selection(selection, chunk_timings)
        return " ".join(c.text for c in chunks)


def parse_time_argument(time_arg: str) -> float:
    """
    Parse a CLI time argument into seconds.

    Convenience function for CLI usage.

    Args:
        time_arg: Time string (e.g., "32", "1:32", "0:01:32").

    Returns:
        Time in seconds.
    """
    return Selection.from_string(time_arg)
