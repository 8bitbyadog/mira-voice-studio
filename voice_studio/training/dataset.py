"""
Dataset management for Mira Voice Studio.

Handles:
- Loading and saving datasets
- Clip approval/rejection
- Dataset statistics
- Export for training
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np

from voice_studio.utils.audio_utils import load_audio, save_audio, get_audio_duration


@dataclass
class DatasetClip:
    """A clip in a training dataset."""

    index: int
    filename: str
    duration: float
    transcript: str
    source_file: str
    approved: bool = True
    audio_path: Optional[Path] = None

    def load_audio(self) -> Optional[np.ndarray]:
        """Load the audio data for this clip."""
        if self.audio_path and self.audio_path.exists():
            audio, _ = load_audio(self.audio_path)
            return audio
        return None


@dataclass
class Dataset:
    """A training dataset."""

    name: str
    path: Path
    sample_rate: int
    clips: List[DatasetClip] = field(default_factory=list)
    created_at: str = ""
    source_files: List[str] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        """Total duration of all clips."""
        return sum(c.duration for c in self.clips)

    @property
    def approved_duration(self) -> float:
        """Duration of approved clips only."""
        return sum(c.duration for c in self.clips if c.approved)

    @property
    def clip_count(self) -> int:
        """Total number of clips."""
        return len(self.clips)

    @property
    def approved_count(self) -> int:
        """Number of approved clips."""
        return sum(1 for c in self.clips if c.approved)

    @property
    def rejected_count(self) -> int:
        """Number of rejected clips."""
        return sum(1 for c in self.clips if not c.approved)

    def get_clip(self, index: int) -> Optional[DatasetClip]:
        """Get a clip by index."""
        for clip in self.clips:
            if clip.index == index:
                return clip
        return None

    def approve_clip(self, index: int) -> bool:
        """Mark a clip as approved."""
        clip = self.get_clip(index)
        if clip:
            clip.approved = True
            return True
        return False

    def reject_clip(self, index: int) -> bool:
        """Mark a clip as rejected."""
        clip = self.get_clip(index)
        if clip:
            clip.approved = False
            return True
        return False

    def update_transcript(self, index: int, transcript: str) -> bool:
        """Update a clip's transcript."""
        clip = self.get_clip(index)
        if clip:
            clip.transcript = transcript
            return True
        return False

    def get_approved_clips(self) -> List[DatasetClip]:
        """Get only approved clips."""
        return [c for c in self.clips if c.approved]

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "name": self.name,
            "total_clips": self.clip_count,
            "approved_clips": self.approved_count,
            "rejected_clips": self.rejected_count,
            "total_duration": self.total_duration,
            "approved_duration": self.approved_duration,
            "sample_rate": self.sample_rate,
            "source_files": len(self.source_files),
        }


class DatasetManager:
    """
    Manage training datasets.

    Features:
    - List available datasets
    - Load/save datasets
    - Create new datasets from processed audio
    - Export datasets for training
    """

    def __init__(self, datasets_dir: Optional[Path] = None):
        """
        Initialize the dataset manager.

        Args:
            datasets_dir: Base directory for datasets.
        """
        if datasets_dir is None:
            datasets_dir = Path.home() / "mira_voice_studio" / "training" / "datasets"

        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> List[str]:
        """
        List available datasets.

        Returns:
            List of dataset names.
        """
        datasets = []
        for path in self.datasets_dir.iterdir():
            if path.is_dir():
                metadata_path = path / "metadata.json"
                if metadata_path.exists():
                    datasets.append(path.name)
        return sorted(datasets)

    def load_dataset(self, name: str) -> Optional[Dataset]:
        """
        Load a dataset by name.

        Args:
            name: Dataset name.

        Returns:
            Dataset object, or None if not found.
        """
        dataset_path = self.datasets_dir / name
        metadata_path = dataset_path / "metadata.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        wavs_dir = dataset_path / "wavs"

        clips = []
        for clip_data in metadata.get("clips", []):
            clip = DatasetClip(
                index=clip_data.get("index", 0),
                filename=clip_data.get("filename", ""),
                duration=clip_data.get("duration", 0.0),
                transcript=clip_data.get("transcript", ""),
                source_file=clip_data.get("source_file", ""),
                approved=clip_data.get("approved", True),
                audio_path=wavs_dir / clip_data.get("filename", ""),
            )
            clips.append(clip)

        return Dataset(
            name=name,
            path=dataset_path,
            sample_rate=metadata.get("sample_rate", 44100),
            clips=clips,
            created_at=metadata.get("created_at", ""),
            source_files=metadata.get("source_files", []),
        )

    def save_dataset(self, dataset: Dataset) -> None:
        """
        Save a dataset's metadata.

        Args:
            dataset: Dataset to save.
        """
        metadata = {
            "name": dataset.name,
            "created_at": dataset.created_at,
            "sample_rate": dataset.sample_rate,
            "total_duration_seconds": dataset.total_duration,
            "clip_count": dataset.clip_count,
            "approved_count": dataset.approved_count,
            "source_files": dataset.source_files,
            "clips": [
                {
                    "index": c.index,
                    "filename": c.filename,
                    "duration": c.duration,
                    "transcript": c.transcript,
                    "source_file": c.source_file,
                    "approved": c.approved,
                }
                for c in dataset.clips
            ]
        }

        metadata_path = dataset.path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Also update transcripts file
        transcripts = []
        for clip in dataset.clips:
            if clip.approved:
                transcripts.append(f"{clip.filename}|{clip.transcript}")

        transcripts_path = dataset.path / "transcripts.txt"
        transcripts_path.write_text("\n".join(transcripts), encoding="utf-8")

    def delete_dataset(self, name: str) -> bool:
        """
        Delete a dataset.

        Args:
            name: Dataset name.

        Returns:
            True if deleted successfully.
        """
        import shutil

        dataset_path = self.datasets_dir / name
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
            return True
        return False

    def rename_dataset(self, old_name: str, new_name: str) -> bool:
        """
        Rename a dataset.

        Args:
            old_name: Current name.
            new_name: New name.

        Returns:
            True if renamed successfully.
        """
        old_path = self.datasets_dir / old_name
        new_path = self.datasets_dir / new_name

        if not old_path.exists():
            return False

        if new_path.exists():
            return False

        old_path.rename(new_path)

        # Update metadata
        dataset = self.load_dataset(new_name)
        if dataset:
            dataset.name = new_name
            self.save_dataset(dataset)

        return True

    def export_for_training(
        self,
        dataset_name: str,
        output_dir: Path,
        format: str = "gptsovits"
    ) -> Path:
        """
        Export a dataset for training.

        Args:
            dataset_name: Name of the dataset.
            output_dir: Output directory.
            format: Export format ("gptsovits", "coqui").

        Returns:
            Path to exported dataset.
        """
        dataset = self.load_dataset(dataset_name)
        if dataset is None:
            raise ValueError(f"Dataset not found: {dataset_name}")

        output_dir = Path(output_dir)
        export_dir = output_dir / dataset_name
        export_dir.mkdir(parents=True, exist_ok=True)

        if format == "gptsovits":
            return self._export_gptsovits(dataset, export_dir)
        elif format == "coqui":
            return self._export_coqui(dataset, export_dir)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _export_gptsovits(self, dataset: Dataset, output_dir: Path) -> Path:
        """Export for GPT-SoVITS training."""
        import shutil

        wavs_dir = output_dir / "wavs"
        wavs_dir.mkdir(exist_ok=True)

        # Copy approved clips
        lines = []
        for clip in dataset.get_approved_clips():
            if clip.audio_path and clip.audio_path.exists():
                dest = wavs_dir / clip.filename
                shutil.copy2(clip.audio_path, dest)
                lines.append(f"{clip.filename}|{clip.transcript}")

        # Write file list
        filelist_path = output_dir / "filelist.txt"
        filelist_path.write_text("\n".join(lines), encoding="utf-8")

        return output_dir

    def _export_coqui(self, dataset: Dataset, output_dir: Path) -> Path:
        """Export for Coqui TTS training."""
        import shutil
        import csv

        wavs_dir = output_dir / "wavs"
        wavs_dir.mkdir(exist_ok=True)

        # Copy approved clips and build metadata
        metadata_rows = []
        for clip in dataset.get_approved_clips():
            if clip.audio_path and clip.audio_path.exists():
                # Coqui expects specific filename format
                new_filename = f"audio_{clip.index:05d}.wav"
                dest = wavs_dir / new_filename
                shutil.copy2(clip.audio_path, dest)

                # Format: audio_file|text|speaker_name
                metadata_rows.append([new_filename, clip.transcript, "speaker"])

        # Write metadata CSV
        metadata_path = output_dir / "metadata.csv"
        with open(metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="|")
            for row in metadata_rows:
                writer.writerow(row)

        return output_dir

    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all datasets."""
        stats = []
        for name in self.list_datasets():
            dataset = self.load_dataset(name)
            if dataset:
                stats.append(dataset.get_stats())
        return stats
