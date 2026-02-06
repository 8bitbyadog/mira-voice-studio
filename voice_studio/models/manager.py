"""
Voice model management for Mira Voice Studio.

Handles:
- Listing available models (custom and pretrained)
- Loading model metadata
- Importing/exporting models
- Deleting models
"""

import json
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class VoiceModel:
    """Information about a voice model."""

    name: str
    path: Path
    model_type: str  # "custom" or "pretrained"
    created_at: Optional[str] = None
    language: str = "en"
    description: str = ""
    training_duration_hours: float = 0.0
    training_quality: str = ""
    has_reference_audio: bool = False
    has_gpt_model: bool = False
    has_sovits_model: bool = False

    def get_info(self) -> Dict[str, Any]:
        """Get model info as dictionary."""
        return {
            "name": self.name,
            "type": self.model_type,
            "created_at": self.created_at,
            "language": self.language,
            "description": self.description,
            "training_duration_hours": self.training_duration_hours,
            "training_quality": self.training_quality,
            "has_reference_audio": self.has_reference_audio,
            "has_gpt_model": self.has_gpt_model,
            "has_sovits_model": self.has_sovits_model,
        }


class ModelManager:
    """
    Manage voice models for TTS.

    Features:
    - List custom and pretrained models
    - Load model metadata
    - Test models with sample text
    - Import/export models as .zip
    - Delete models
    """

    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize the model manager.

        Args:
            models_dir: Base directory for models.
        """
        if models_dir is None:
            models_dir = Path.home() / "mira_voice_studio" / "models"

        self.models_dir = Path(models_dir)
        self.pretrained_dir = self.models_dir / "pretrained"
        self.custom_dir = self.models_dir / "custom"

        # Ensure directories exist
        self.pretrained_dir.mkdir(parents=True, exist_ok=True)
        self.custom_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self, model_type: Optional[str] = None) -> List[VoiceModel]:
        """
        List available voice models.

        Args:
            model_type: Filter by type ("custom", "pretrained", or None for all).

        Returns:
            List of VoiceModel objects.
        """
        models = []

        if model_type in (None, "custom"):
            for path in self.custom_dir.iterdir():
                if path.is_dir():
                    model = self._load_model_info(path, "custom")
                    if model:
                        models.append(model)

        if model_type in (None, "pretrained"):
            for path in self.pretrained_dir.iterdir():
                if path.is_dir():
                    model = self._load_model_info(path, "pretrained")
                    if model:
                        models.append(model)

        return sorted(models, key=lambda m: m.name)

    def _load_model_info(self, model_path: Path, model_type: str) -> Optional[VoiceModel]:
        """Load model information from a directory."""
        # Check for config or metadata file
        config_path = model_path / "config.json"
        metadata_path = model_path / "metadata.json"

        config = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                config = json.load(f)
        elif config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        # Check for model files
        has_reference = any(model_path.glob("reference.*")) or any(model_path.glob("ref_audio.*"))
        has_gpt = any(model_path.glob("*gpt*.ckpt")) or any(model_path.glob("*gpt*.pth"))
        has_sovits = any(model_path.glob("*sovits*.pth")) or any(model_path.glob("*.pth"))

        # Filter out gpt files from sovits check
        if has_sovits and not has_gpt:
            pth_files = list(model_path.glob("*.pth"))
            has_sovits = any("gpt" not in f.name.lower() for f in pth_files)

        return VoiceModel(
            name=model_path.name,
            path=model_path,
            model_type=model_type,
            created_at=config.get("created_at"),
            language=config.get("language", "en"),
            description=config.get("description", config.get("notes", "")),
            training_duration_hours=config.get("training_duration_hours", 0.0),
            training_quality=config.get("training_quality", ""),
            has_reference_audio=has_reference,
            has_gpt_model=has_gpt,
            has_sovits_model=has_sovits,
        )

    def get_model(self, name: str) -> Optional[VoiceModel]:
        """
        Get a specific model by name.

        Args:
            name: Model name.

        Returns:
            VoiceModel or None if not found.
        """
        # Check custom first
        custom_path = self.custom_dir / name
        if custom_path.exists():
            return self._load_model_info(custom_path, "custom")

        # Check pretrained
        pretrained_path = self.pretrained_dir / name
        if pretrained_path.exists():
            return self._load_model_info(pretrained_path, "pretrained")

        return None

    def delete_model(self, name: str) -> bool:
        """
        Delete a custom model.

        Args:
            name: Model name.

        Returns:
            True if deleted successfully.
        """
        model_path = self.custom_dir / name

        if not model_path.exists():
            return False

        try:
            shutil.rmtree(model_path)
            return True
        except Exception:
            return False

    def rename_model(self, old_name: str, new_name: str) -> bool:
        """
        Rename a custom model.

        Args:
            old_name: Current name.
            new_name: New name.

        Returns:
            True if renamed successfully.
        """
        old_path = self.custom_dir / old_name
        new_path = self.custom_dir / new_name

        if not old_path.exists():
            return False

        if new_path.exists():
            return False

        try:
            old_path.rename(new_path)

            # Update metadata
            metadata_path = new_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                metadata["name"] = new_name
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            return True
        except Exception:
            return False

    def export_model(self, name: str, output_path: Path) -> Path:
        """
        Export a model as a .zip file.

        Args:
            name: Model name.
            output_path: Output directory or file path.

        Returns:
            Path to the created .zip file.
        """
        model = self.get_model(name)
        if model is None:
            raise ValueError(f"Model not found: {name}")

        output_path = Path(output_path)

        if output_path.is_dir():
            zip_path = output_path / f"{name}.zip"
        else:
            zip_path = output_path

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in model.path.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(model.path)
                    zf.write(file_path, arcname)

        return zip_path

    def import_model(self, zip_path: Path, name: Optional[str] = None) -> VoiceModel:
        """
        Import a model from a .zip file.

        Args:
            zip_path: Path to the .zip file.
            name: Optional name override.

        Returns:
            The imported VoiceModel.
        """
        zip_path = Path(zip_path)

        if not zip_path.exists():
            raise ValueError(f"File not found: {zip_path}")

        # Determine model name
        if name is None:
            name = zip_path.stem

        # Check if already exists
        model_path = self.custom_dir / name
        if model_path.exists():
            raise ValueError(f"Model already exists: {name}")

        # Extract
        model_path.mkdir(parents=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(model_path)

        # Load and return
        model = self._load_model_info(model_path, "custom")
        if model is None:
            shutil.rmtree(model_path)
            raise ValueError("Invalid model archive")

        return model

    def create_model_placeholder(
        self,
        name: str,
        reference_audio: Optional[Path] = None,
        reference_text: Optional[str] = None,
        description: str = ""
    ) -> Path:
        """
        Create a placeholder model directory for training.

        Args:
            name: Model name.
            reference_audio: Optional reference audio file.
            reference_text: Optional reference text.
            description: Model description.

        Returns:
            Path to the created model directory.
        """
        model_path = self.custom_dir / name
        model_path.mkdir(parents=True, exist_ok=True)

        # Create metadata
        metadata = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "language": "en",
            "description": description,
            "reference_text": reference_text or "",
            "status": "placeholder",
        }

        metadata_path = model_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Copy reference audio if provided
        if reference_audio and reference_audio.exists():
            dest = model_path / f"reference{reference_audio.suffix}"
            shutil.copy2(reference_audio, dest)

        # Save reference text
        if reference_text:
            ref_text_path = model_path / "reference.txt"
            ref_text_path.write_text(reference_text, encoding="utf-8")

        return model_path

    def get_model_stats(self) -> Dict[str, Any]:
        """Get overall model statistics."""
        custom = self.list_models("custom")
        pretrained = self.list_models("pretrained")

        return {
            "custom_count": len(custom),
            "pretrained_count": len(pretrained),
            "total_count": len(custom) + len(pretrained),
            "custom_models": [m.name for m in custom],
            "pretrained_models": [m.name for m in pretrained],
        }
