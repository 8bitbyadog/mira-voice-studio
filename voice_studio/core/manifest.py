"""
Manifest generator for Mira Voice Studio.

Creates manifest.json files with metadata about generated outputs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict

from voice_studio.core.audio_stitcher import ChunkTiming


@dataclass
class ChunkInfo:
    """Information about a single chunk in the manifest."""

    index: int
    text: str
    slug: str
    duration_seconds: float
    start_time_in_master: float
    end_time_in_master: float
    status: str  # "success" or "failed"
    error: str = ""

    @classmethod
    def from_chunk_timing(cls, timing: ChunkTiming) -> "ChunkInfo":
        """Create from ChunkTiming object."""
        return cls(
            index=timing.index,
            text=timing.text,
            slug=timing.slug,
            duration_seconds=round(timing.duration, 3),
            start_time_in_master=round(timing.start_time, 3),
            end_time_in_master=round(timing.end_time, 3),
            status="success" if timing.success else "failed",
            error=timing.error if not timing.success else "",
        )


@dataclass
class SelectionInfo:
    """Information about an exported selection."""

    in_time: str  # Formatted time string
    out_time: str  # Formatted time string
    folder: str  # Relative path to selection folder


@dataclass
class ManifestSettings:
    """Settings recorded in the manifest."""

    pause_between_sentences_ms: int = 300
    sample_rate: int = 44100
    bit_depth: int = 16
    word_level_captions: bool = True


@dataclass
class MasterFiles:
    """Paths to master output files."""

    audio: str
    srt_sentences: str
    srt_words: str = ""
    vtt: str = ""


@dataclass
class Manifest:
    """Complete manifest for a generation session."""

    created_at: str
    source_file: str
    voice: str
    speed: float
    total_duration_seconds: float
    total_sentences: int
    successful_sentences: int
    failed_sentences: int
    settings: ManifestSettings
    master_files: MasterFiles
    chunks_exported: bool = False
    selections_exported: List[SelectionInfo] = field(default_factory=list)
    chunks: List[ChunkInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "created_at": self.created_at,
            "source_file": self.source_file,
            "voice": self.voice,
            "speed": self.speed,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "total_sentences": self.total_sentences,
            "successful_sentences": self.successful_sentences,
            "failed_sentences": self.failed_sentences,
            "settings": asdict(self.settings),
            "master_files": asdict(self.master_files),
            "chunks_exported": self.chunks_exported,
            "selections_exported": [asdict(s) for s in self.selections_exported],
            "chunks": [asdict(c) for c in self.chunks],
        }
        return data

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class ManifestGenerator:
    """
    Generate manifest.json files for output sessions.

    The manifest contains:
    - Generation metadata (timestamp, voice, speed)
    - Statistics (duration, sentence counts)
    - File paths for all outputs
    - Per-chunk information with timings
    - Selection export history
    """

    def __init__(self):
        """Initialize the manifest generator."""
        pass

    def create_manifest(
        self,
        source_name: str,
        voice: str,
        speed: float,
        chunk_timings: List[ChunkTiming],
        master_files: Dict[str, Path],
        output_dir: Path,
        settings: Optional[ManifestSettings] = None,
    ) -> Manifest:
        """
        Create a new manifest.

        Args:
            source_name: Name of the source file/input.
            voice: Voice model used.
            speed: Speaking speed used.
            chunk_timings: List of chunk timing information.
            master_files: Dictionary of master file paths.
            output_dir: Base output directory.
            settings: Optional settings override.

        Returns:
            Manifest object.
        """
        if settings is None:
            settings = ManifestSettings()

        # Calculate statistics
        total_sentences = len(chunk_timings)
        successful = sum(1 for c in chunk_timings if c.success)
        failed = total_sentences - successful

        # Calculate total duration
        if chunk_timings:
            total_duration = chunk_timings[-1].end_time
        else:
            total_duration = 0.0

        # Create relative paths for master files
        output_dir = Path(output_dir)
        master_files_rel = MasterFiles(
            audio=self._relative_path(master_files.get("audio"), output_dir),
            srt_sentences=self._relative_path(master_files.get("srt_sentences"), output_dir),
            srt_words=self._relative_path(master_files.get("srt_words"), output_dir),
            vtt=self._relative_path(master_files.get("vtt"), output_dir),
        )

        # Create chunk info
        chunks = [ChunkInfo.from_chunk_timing(t) for t in chunk_timings]

        return Manifest(
            created_at=datetime.now().isoformat(),
            source_file=source_name,
            voice=voice,
            speed=speed,
            total_duration_seconds=total_duration,
            total_sentences=total_sentences,
            successful_sentences=successful,
            failed_sentences=failed,
            settings=settings,
            master_files=master_files_rel,
            chunks=chunks,
        )

    def _relative_path(
        self,
        path: Optional[Path],
        base_dir: Path
    ) -> str:
        """Get relative path string."""
        if path is None:
            return ""
        try:
            return str(Path(path).relative_to(base_dir))
        except ValueError:
            return str(path)

    def save_manifest(
        self,
        manifest: Manifest,
        output_path: Path
    ) -> None:
        """
        Save manifest to a JSON file.

        Args:
            manifest: Manifest object to save.
            output_path: Path to save the manifest.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(manifest.to_json())

    def load_manifest(self, manifest_path: Path) -> Manifest:
        """
        Load a manifest from a JSON file.

        Args:
            manifest_path: Path to the manifest file.

        Returns:
            Manifest object.
        """
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Parse settings
        settings_data = data.get("settings", {})
        settings = ManifestSettings(
            pause_between_sentences_ms=settings_data.get("pause_between_sentences_ms", 300),
            sample_rate=settings_data.get("sample_rate", 44100),
            bit_depth=settings_data.get("bit_depth", 16),
            word_level_captions=settings_data.get("word_level_captions", True),
        )

        # Parse master files
        mf_data = data.get("master_files", {})
        master_files = MasterFiles(
            audio=mf_data.get("audio", ""),
            srt_sentences=mf_data.get("srt_sentences", ""),
            srt_words=mf_data.get("srt_words", ""),
            vtt=mf_data.get("vtt", ""),
        )

        # Parse selections
        selections = [
            SelectionInfo(
                in_time=s.get("in_time", ""),
                out_time=s.get("out_time", ""),
                folder=s.get("folder", ""),
            )
            for s in data.get("selections_exported", [])
        ]

        # Parse chunks
        chunks = [
            ChunkInfo(
                index=c.get("index", 0),
                text=c.get("text", ""),
                slug=c.get("slug", ""),
                duration_seconds=c.get("duration_seconds", 0.0),
                start_time_in_master=c.get("start_time_in_master", 0.0),
                end_time_in_master=c.get("end_time_in_master", 0.0),
                status=c.get("status", "unknown"),
                error=c.get("error", ""),
            )
            for c in data.get("chunks", [])
        ]

        return Manifest(
            created_at=data.get("created_at", ""),
            source_file=data.get("source_file", ""),
            voice=data.get("voice", ""),
            speed=data.get("speed", 1.0),
            total_duration_seconds=data.get("total_duration_seconds", 0.0),
            total_sentences=data.get("total_sentences", 0),
            successful_sentences=data.get("successful_sentences", 0),
            failed_sentences=data.get("failed_sentences", 0),
            settings=settings,
            master_files=master_files,
            chunks_exported=data.get("chunks_exported", False),
            selections_exported=selections,
            chunks=chunks,
        )

    def update_manifest_chunks_exported(
        self,
        manifest_path: Path,
        exported: bool = True
    ) -> None:
        """
        Update the chunks_exported flag in an existing manifest.

        Args:
            manifest_path: Path to the manifest file.
            exported: Whether chunks have been exported.
        """
        manifest = self.load_manifest(manifest_path)
        manifest.chunks_exported = exported
        self.save_manifest(manifest, manifest_path)

    def add_selection_export(
        self,
        manifest_path: Path,
        in_time: float,
        out_time: float,
        folder: str
    ) -> None:
        """
        Add a selection export record to the manifest.

        Args:
            manifest_path: Path to the manifest file.
            in_time: Selection start time in seconds.
            out_time: Selection end time in seconds.
            folder: Relative path to the selection folder.
        """
        manifest = self.load_manifest(manifest_path)

        # Format times
        in_str = self._format_time(in_time)
        out_str = self._format_time(out_time)

        selection = SelectionInfo(
            in_time=in_str,
            out_time=out_str,
            folder=folder,
        )
        manifest.selections_exported.append(selection)

        self.save_manifest(manifest, manifest_path)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS.mmm."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
