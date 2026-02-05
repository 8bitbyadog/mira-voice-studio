"""
SRT and VTT caption generator for Mira Voice Studio.

Generates subtitle files from timing information.
"""

from pathlib import Path
from typing import List, Optional, TextIO
from dataclasses import dataclass

from voice_studio.core.aligner import AlignmentResult, WordTiming, SegmentTiming
from voice_studio.core.audio_stitcher import ChunkTiming


def format_srt_timestamp(seconds: float) -> str:
    """
    Format seconds as SRT timestamp (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds.

    Returns:
        SRT-formatted timestamp.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    """
    Format seconds as VTT timestamp (HH:MM:SS.mmm).

    Args:
        seconds: Time in seconds.

    Returns:
        VTT-formatted timestamp.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


@dataclass
class CaptionEntry:
    """A single caption entry."""

    index: int
    start_time: float
    end_time: float
    text: str

    def to_srt(self) -> str:
        """Convert to SRT format."""
        return (
            f"{self.index}\n"
            f"{format_srt_timestamp(self.start_time)} --> "
            f"{format_srt_timestamp(self.end_time)}\n"
            f"{self.text}\n"
        )

    def to_vtt(self) -> str:
        """Convert to VTT format."""
        return (
            f"{format_vtt_timestamp(self.start_time)} --> "
            f"{format_vtt_timestamp(self.end_time)}\n"
            f"{self.text}\n"
        )


class SRTGenerator:
    """
    Generate SRT and VTT caption files.

    Supports:
    - Sentence-level captions (from chunk timings)
    - Word-level captions (from Whisper alignment)
    - Time offset adjustment for selections
    """

    def __init__(self):
        """Initialize the SRT generator."""
        pass

    def generate_sentence_captions(
        self,
        chunk_timings: List[ChunkTiming],
        time_offset: float = 0.0
    ) -> List[CaptionEntry]:
        """
        Generate caption entries from chunk timings (sentence-level).

        Args:
            chunk_timings: List of ChunkTiming objects.
            time_offset: Time to subtract from all timestamps (for selections).

        Returns:
            List of CaptionEntry objects.
        """
        entries = []

        for i, chunk in enumerate(chunk_timings, start=1):
            if not chunk.success:
                continue  # Skip failed chunks

            entry = CaptionEntry(
                index=i,
                start_time=max(0, chunk.start_time - time_offset),
                end_time=max(0, chunk.end_time - time_offset),
                text=chunk.text,
            )
            entries.append(entry)

        return entries

    def generate_word_captions(
        self,
        alignment: AlignmentResult,
        time_offset: float = 0.0,
        words_per_caption: int = 1
    ) -> List[CaptionEntry]:
        """
        Generate caption entries from word-level alignment.

        Args:
            alignment: AlignmentResult from Whisper.
            time_offset: Time to subtract from all timestamps.
            words_per_caption: Number of words per caption entry.

        Returns:
            List of CaptionEntry objects.
        """
        entries = []
        words = alignment.words

        if words_per_caption <= 1:
            # One word per caption
            for i, word in enumerate(words, start=1):
                entry = CaptionEntry(
                    index=i,
                    start_time=max(0, word.start_time - time_offset),
                    end_time=max(0, word.end_time - time_offset),
                    text=word.word.strip(),
                )
                entries.append(entry)
        else:
            # Group words
            for i in range(0, len(words), words_per_caption):
                group = words[i:i + words_per_caption]
                if not group:
                    continue

                text = " ".join(w.word.strip() for w in group)
                entry = CaptionEntry(
                    index=(i // words_per_caption) + 1,
                    start_time=max(0, group[0].start_time - time_offset),
                    end_time=max(0, group[-1].end_time - time_offset),
                    text=text,
                )
                entries.append(entry)

        return entries

    def write_srt(
        self,
        entries: List[CaptionEntry],
        output_path: Path
    ) -> None:
        """
        Write caption entries to an SRT file.

        Args:
            entries: List of CaptionEntry objects.
            output_path: Path to output file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for i, entry in enumerate(entries):
                # Re-index for consistency
                entry.index = i + 1
                f.write(entry.to_srt())
                f.write("\n")

    def write_vtt(
        self,
        entries: List[CaptionEntry],
        output_path: Path
    ) -> None:
        """
        Write caption entries to a VTT file.

        Args:
            entries: List of CaptionEntry objects.
            output_path: Path to output file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            # VTT header
            f.write("WEBVTT\n\n")

            for entry in entries:
                f.write(entry.to_vtt())
                f.write("\n")

    def generate_from_chunk_timings(
        self,
        chunk_timings: List[ChunkTiming],
        output_dir: Path,
        base_name: str,
        generate_word_level: bool = False,
        alignment: Optional[AlignmentResult] = None
    ) -> dict:
        """
        Generate all caption files from chunk timings.

        Args:
            chunk_timings: List of chunk timings.
            output_dir: Output directory.
            base_name: Base name for output files.
            generate_word_level: Whether to generate word-level captions.
            alignment: Alignment result for word-level captions.

        Returns:
            Dictionary of generated file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated = {}

        # Sentence-level SRT
        sentence_entries = self.generate_sentence_captions(chunk_timings)
        srt_path = output_dir / f"{base_name}.srt"
        self.write_srt(sentence_entries, srt_path)
        generated["srt_sentences"] = srt_path

        # VTT
        vtt_path = output_dir / f"{base_name}.vtt"
        self.write_vtt(sentence_entries, vtt_path)
        generated["vtt"] = vtt_path

        # Word-level SRT (if alignment provided)
        if generate_word_level and alignment:
            word_entries = self.generate_word_captions(alignment)
            word_srt_path = output_dir / f"{base_name}_words.srt"
            self.write_srt(word_entries, word_srt_path)
            generated["srt_words"] = word_srt_path

        return generated

    def generate_chunk_srt(
        self,
        chunk: ChunkTiming,
        output_path: Path,
        alignment: Optional[AlignmentResult] = None
    ) -> None:
        """
        Generate SRT for a single chunk (timestamps start at 00:00:00).

        Args:
            chunk: The chunk timing information.
            output_path: Path to output SRT file.
            alignment: Optional word-level alignment.
        """
        entries = [
            CaptionEntry(
                index=1,
                start_time=0.0,
                end_time=chunk.duration,
                text=chunk.text,
            )
        ]
        self.write_srt(entries, output_path)

    def filter_by_time_range(
        self,
        entries: List[CaptionEntry],
        start_time: float,
        end_time: float,
        reset_timestamps: bool = True
    ) -> List[CaptionEntry]:
        """
        Filter caption entries to a time range.

        Args:
            entries: List of caption entries.
            start_time: Start of range (seconds).
            end_time: End of range (seconds).
            reset_timestamps: If True, subtract start_time from all timestamps.

        Returns:
            Filtered list of entries.
        """
        filtered = []

        for entry in entries:
            # Check if entry overlaps with range
            if entry.end_time < start_time:
                continue
            if entry.start_time > end_time:
                continue

            # Adjust timestamps
            new_start = max(entry.start_time, start_time)
            new_end = min(entry.end_time, end_time)

            if reset_timestamps:
                new_start -= start_time
                new_end -= start_time

            new_entry = CaptionEntry(
                index=len(filtered) + 1,
                start_time=new_start,
                end_time=new_end,
                text=entry.text,
            )
            filtered.append(new_entry)

        return filtered
