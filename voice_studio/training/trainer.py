"""
Voice model trainer for Mira Voice Studio.

Handles:
- Data preparation for GPT-SoVITS training
- Feature extraction (semantic tokens)
- Training loop with progress callbacks
- Model checkpointing and saving
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Generator, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

from voice_studio.training.dataset import DatasetManager, Dataset
from voice_studio.utils.device import get_device, get_device_info


class TrainingQuality(Enum):
    """Training quality presets."""
    QUICK = "quick"      # ~30 minutes, basic fine-tuning
    STANDARD = "standard"  # ~2 hours, balanced training
    HIGH = "high"        # ~6 hours, thorough training


@dataclass
class TrainingConfig:
    """Configuration for training a voice model."""

    # Basic settings
    dataset_name: str
    voice_name: str
    quality: TrainingQuality = TrainingQuality.STANDARD

    # Training parameters (auto-set based on quality)
    epochs: int = 0
    batch_size: int = 4
    learning_rate: float = 1e-4
    validation_split: float = 0.1

    # Feature extraction
    semantic_model: str = "base"  # tiny, base, small for HuBERT/wav2vec

    # Output settings
    output_dir: Optional[Path] = None
    save_checkpoints: bool = True
    checkpoint_interval: int = 10

    def __post_init__(self):
        """Set parameters based on quality preset."""
        if self.epochs == 0:
            quality_epochs = {
                TrainingQuality.QUICK: 10,
                TrainingQuality.STANDARD: 30,
                TrainingQuality.HIGH: 60,
            }
            self.epochs = quality_epochs.get(self.quality, 30)


@dataclass
class TrainingProgress:
    """Progress information during training."""

    stage: str
    current_step: int
    total_steps: int
    epoch: int = 0
    total_epochs: int = 0
    loss: float = 0.0
    message: str = ""

    @property
    def progress(self) -> float:
        """Overall progress as a fraction 0-1."""
        if self.total_steps == 0:
            return 0.0
        return self.current_step / self.total_steps


@dataclass
class TrainingResult:
    """Result of a training run."""

    success: bool
    voice_name: str
    model_path: Optional[Path] = None
    training_duration_hours: float = 0.0
    final_loss: float = 0.0
    epochs_completed: int = 0
    error_message: str = ""


class VoiceTrainer:
    """
    Train custom voice models for GPT-SoVITS.

    Features:
    - Prepares dataset for training
    - Extracts semantic features from audio
    - Runs training loop with progress updates
    - Saves trained models
    """

    def __init__(self, models_dir: Optional[Path] = None, datasets_dir: Optional[Path] = None):
        """
        Initialize the trainer.

        Args:
            models_dir: Directory to save trained models.
            datasets_dir: Directory containing training datasets.
        """
        if models_dir is None:
            models_dir = Path.home() / "mira_voice_studio" / "models" / "custom"
        if datasets_dir is None:
            datasets_dir = Path.home() / "mira_voice_studio" / "training" / "datasets"

        self.models_dir = Path(models_dir)
        self.datasets_dir = Path(datasets_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_manager = DatasetManager(datasets_dir)
        self.device = get_device()
        self._stop_requested = False

        # Lazy-loaded components
        self._feature_extractor = None
        self._tokenizer = None

    def train(
        self,
        config: TrainingConfig,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None
    ) -> Generator[TrainingProgress, None, TrainingResult]:
        """
        Train a voice model.

        Args:
            config: Training configuration.
            progress_callback: Optional callback for progress updates.

        Yields:
            TrainingProgress objects during training.

        Returns:
            TrainingResult when complete.
        """
        self._stop_requested = False
        start_time = datetime.now()

        # Stage 1: Prepare data
        yield from self._stage_prepare_data(config, progress_callback)
        if self._stop_requested:
            return self._create_cancelled_result(config.voice_name)

        # Stage 2: Extract features
        yield from self._stage_extract_features(config, progress_callback)
        if self._stop_requested:
            return self._create_cancelled_result(config.voice_name)

        # Stage 3: Train model
        result = yield from self._stage_train_model(config, progress_callback)
        if self._stop_requested:
            return self._create_cancelled_result(config.voice_name)

        # Stage 4: Save model
        yield from self._stage_save_model(config, result, progress_callback)

        # Calculate training duration
        duration = datetime.now() - start_time
        result.training_duration_hours = duration.total_seconds() / 3600

        return result

    def stop(self):
        """Request training to stop."""
        self._stop_requested = True

    def _stage_prepare_data(
        self,
        config: TrainingConfig,
        callback: Optional[Callable]
    ) -> Generator[TrainingProgress, None, None]:
        """Stage 1: Load and prepare training data."""
        progress = TrainingProgress(
            stage="prepare",
            current_step=0,
            total_steps=3,
            message="Loading dataset..."
        )
        yield progress
        if callback:
            callback(progress)

        # Load dataset
        dataset = self.dataset_manager.load_dataset(config.dataset_name)
        if dataset is None:
            raise ValueError(f"Dataset not found: {config.dataset_name}")

        self._dataset = dataset  # Store for later use
        approved_clips = dataset.get_approved_clips()
        if len(approved_clips) < 5:
            raise ValueError(f"Not enough approved clips. Need at least 5, have {len(approved_clips)}")

        progress.current_step = 1
        progress.message = f"Loaded {len(approved_clips)} clips ({dataset.approved_duration:.1f}s)"
        yield progress
        if callback:
            callback(progress)

        # Use persistent export directory within the dataset folder (not temp)
        self._export_dir = dataset.path / "training_export"
        self._export_dir.mkdir(parents=True, exist_ok=True)

        # Check if we need to re-export (clips changed)
        filelist_path = self._export_dir / "filelist.txt"
        needs_export = True
        if filelist_path.exists():
            existing_lines = filelist_path.read_text().strip().split("\n")
            if len(existing_lines) == len(approved_clips):
                needs_export = False
                progress.message = "Using cached export data"

        if needs_export:
            export_dir = self.dataset_manager.export_for_training(
                config.dataset_name,
                self._export_dir.parent,
                format="gptsovits"
            )
            # Move to our expected location if different
            if export_dir != self._export_dir:
                import shutil
                if self._export_dir.exists():
                    shutil.rmtree(self._export_dir)
                shutil.move(str(export_dir), str(self._export_dir))

        progress.current_step = 2
        progress.message = "Dataset prepared for training"
        yield progress
        if callback:
            callback(progress)

        # Validate export
        filelist_path = self._export_dir / "filelist.txt"
        if not filelist_path.exists():
            raise ValueError("Export failed: filelist.txt not created")

        lines = filelist_path.read_text().strip().split("\n")
        self._training_items = []
        for line in lines:
            if "|" in line:
                filename, transcript = line.split("|", 1)
                audio_path = self._export_dir / "wavs" / filename
                if audio_path.exists():
                    self._training_items.append({
                        "audio_path": audio_path,
                        "transcript": transcript.strip()
                    })

        progress.current_step = 3
        progress.message = f"Prepared {len(self._training_items)} training samples"
        yield progress
        if callback:
            callback(progress)

    def _stage_extract_features(
        self,
        config: TrainingConfig,
        callback: Optional[Callable]
    ) -> Generator[TrainingProgress, None, None]:
        """Stage 2: Extract semantic features from audio (with caching)."""
        total_items = len(self._training_items)

        # Check for cached features
        cache_path = self._dataset.path / "features_cache.npz"
        cache_meta_path = self._dataset.path / "features_cache_meta.json"

        cached_features = self._load_cached_features(cache_path, cache_meta_path, config.semantic_model)

        if cached_features is not None and len(cached_features) == total_items:
            # Use cached features - skip the expensive extraction!
            progress = TrainingProgress(
                stage="features",
                current_step=total_items,
                total_steps=total_items,
                message=f"Loaded {total_items} cached features (skipping extraction)"
            )
            yield progress
            if callback:
                callback(progress)

            self._features = []
            for i, item in enumerate(self._training_items):
                self._features.append({
                    "audio_features": cached_features[i],
                    "transcript": item["transcript"],
                    "audio_path": item["audio_path"]
                })
            return

        # No cache or cache invalid - extract features
        progress = TrainingProgress(
            stage="features",
            current_step=0,
            total_steps=total_items,
            message="Extracting audio features (this will be cached for next time)..."
        )
        yield progress
        if callback:
            callback(progress)

        # Load feature extractor (HuBERT-like model via Whisper)
        self._load_feature_extractor(config.semantic_model)

        self._features = []
        all_features = []
        for i, item in enumerate(self._training_items):
            if self._stop_requested:
                return

            # Extract features
            features = self._extract_features(item["audio_path"])
            all_features.append(features)
            self._features.append({
                "audio_features": features,
                "transcript": item["transcript"],
                "audio_path": item["audio_path"]
            })

            progress.current_step = i + 1
            progress.message = f"Extracted features: {i + 1}/{total_items}"
            yield progress
            if callback:
                callback(progress)

        # Save features to cache for next time
        self._save_features_cache(all_features, cache_path, cache_meta_path, config.semantic_model)
        progress.message = f"Extracted and cached {total_items} features"
        yield progress
        if callback:
            callback(progress)

    def _load_cached_features(
        self,
        cache_path: Path,
        meta_path: Path,
        model_size: str
    ) -> Optional[List[np.ndarray]]:
        """Load cached features if they exist and are valid."""
        if not cache_path.exists() or not meta_path.exists():
            return None

        try:
            # Check metadata matches current config
            with open(meta_path, "r") as f:
                meta = json.load(f)

            if meta.get("model_size") != model_size:
                print(f"Feature cache model mismatch: {meta.get('model_size')} vs {model_size}")
                return None

            if meta.get("num_items") != len(self._training_items):
                print(f"Feature cache count mismatch: {meta.get('num_items')} vs {len(self._training_items)}")
                return None

            # Load the cached features
            data = np.load(cache_path)
            features = [data[f"features_{i}"] for i in range(meta["num_items"])]
            print(f"Loaded {len(features)} cached features from {cache_path}")
            return features

        except Exception as e:
            print(f"Error loading feature cache: {e}")
            return None

    def _save_features_cache(
        self,
        features: List[np.ndarray],
        cache_path: Path,
        meta_path: Path,
        model_size: str
    ) -> None:
        """Save extracted features to cache."""
        try:
            # Save features as compressed numpy archive
            feature_dict = {f"features_{i}": f for i, f in enumerate(features)}
            np.savez_compressed(cache_path, **feature_dict)

            # Save metadata
            meta = {
                "model_size": model_size,
                "num_items": len(features),
                "created_at": datetime.now().isoformat(),
                "feature_shape": list(features[0].shape) if features else [],
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            print(f"Saved feature cache to {cache_path} ({len(features)} items)")

        except Exception as e:
            print(f"Error saving feature cache: {e}")

    def _stage_train_model(
        self,
        config: TrainingConfig,
        callback: Optional[Callable]
    ) -> Generator[TrainingProgress, None, TrainingResult]:
        """Stage 3: Run training loop."""
        import torch
        import torch.nn as nn
        import torch.optim as optim

        total_epochs = config.epochs
        samples_per_epoch = len(self._features)
        total_steps = total_epochs * samples_per_epoch

        progress = TrainingProgress(
            stage="training",
            current_step=0,
            total_steps=total_steps,
            epoch=0,
            total_epochs=total_epochs,
            message="Initializing training..."
        )
        yield progress
        if callback:
            callback(progress)

        # Initialize model components
        model = self._create_voice_model(config)
        model = model.to(self.device)

        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        best_loss = float('inf')
        total_loss = 0.0
        step = 0

        for epoch in range(total_epochs):
            if self._stop_requested:
                break

            epoch_loss = 0.0
            np.random.shuffle(self._features)

            for i, sample in enumerate(self._features):
                if self._stop_requested:
                    break

                # Forward pass
                audio_features = torch.tensor(
                    sample["audio_features"],
                    dtype=torch.float32,
                    device=self.device
                )

                # Simple reconstruction loss for voice adaptation
                optimizer.zero_grad()
                output = model(audio_features.unsqueeze(0))
                loss = criterion(output, audio_features.unsqueeze(0))

                # Backward pass
                loss.backward()
                optimizer.step()

                step_loss = loss.item()
                epoch_loss += step_loss
                total_loss = epoch_loss / (i + 1)
                step += 1

                progress.current_step = step
                progress.epoch = epoch + 1
                progress.loss = total_loss
                progress.message = f"Epoch {epoch + 1}/{total_epochs} | Loss: {total_loss:.4f}"
                yield progress
                if callback:
                    callback(progress)

            # Save checkpoint
            if config.save_checkpoints and (epoch + 1) % config.checkpoint_interval == 0:
                self._save_checkpoint(model, optimizer, epoch, total_loss, config)

            # Track best model
            avg_epoch_loss = epoch_loss / samples_per_epoch
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                self._best_model_state = model.state_dict().copy()

        # Store final model
        self._final_model = model
        self._final_loss = best_loss

        return TrainingResult(
            success=not self._stop_requested,
            voice_name=config.voice_name,
            final_loss=best_loss,
            epochs_completed=epoch + 1 if not self._stop_requested else epoch,
        )

    def _stage_save_model(
        self,
        config: TrainingConfig,
        result: TrainingResult,
        callback: Optional[Callable]
    ) -> Generator[TrainingProgress, None, None]:
        """Stage 4: Save the trained model."""
        import torch

        progress = TrainingProgress(
            stage="saving",
            current_step=0,
            total_steps=4,
            message="Saving trained model..."
        )
        yield progress
        if callback:
            callback(progress)

        # Create model directory
        model_dir = self.models_dir / config.voice_name
        model_dir.mkdir(parents=True, exist_ok=True)

        progress.current_step = 1
        progress.message = "Saving model weights..."
        yield progress
        if callback:
            callback(progress)

        # Save model weights
        model_path = model_dir / f"{config.voice_name}_sovits.pth"
        if hasattr(self, '_best_model_state'):
            torch.save({
                'model_state_dict': self._best_model_state,
                'config': {
                    'voice_name': config.voice_name,
                    'quality': config.quality.value,
                    'epochs': config.epochs,
                }
            }, model_path)
        elif hasattr(self, '_final_model'):
            torch.save({
                'model_state_dict': self._final_model.state_dict(),
                'config': {
                    'voice_name': config.voice_name,
                    'quality': config.quality.value,
                    'epochs': config.epochs,
                }
            }, model_path)

        progress.current_step = 2
        progress.message = "Selecting reference audio..."
        yield progress
        if callback:
            callback(progress)

        # Copy best reference audio (longest approved clip)
        if self._training_items:
            # Find the best reference clip (around 5-10 seconds is ideal)
            best_ref = None
            best_score = 0
            for item in self._training_items:
                import soundfile as sf
                info = sf.info(item["audio_path"])
                duration = info.duration
                # Score: prefer 5-10 second clips
                if 5 <= duration <= 10:
                    score = 100 - abs(duration - 7.5)
                elif duration >= 3:
                    score = 50 - abs(duration - 7.5)
                else:
                    score = duration * 10
                if score > best_score:
                    best_score = score
                    best_ref = item

            if best_ref:
                ref_dest = model_dir / f"reference{best_ref['audio_path'].suffix}"
                shutil.copy2(best_ref["audio_path"], ref_dest)
                # Save reference text
                ref_text_path = model_dir / "reference.txt"
                ref_text_path.write_text(best_ref["transcript"], encoding="utf-8")

        progress.current_step = 3
        progress.message = "Writing metadata..."
        yield progress
        if callback:
            callback(progress)

        # Save metadata
        metadata = {
            "name": config.voice_name,
            "created_at": datetime.now().isoformat(),
            "language": "en",
            "description": f"Trained from {config.dataset_name} dataset",
            "training_duration_hours": result.training_duration_hours,
            "training_quality": config.quality.value,
            "epochs_completed": result.epochs_completed,
            "final_loss": result.final_loss,
            "source_dataset": config.dataset_name,
            "device": str(self.device),
        }

        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        progress.current_step = 4
        progress.message = f"Model saved to {model_dir.name}"
        yield progress
        if callback:
            callback(progress)

        # Update result
        result.model_path = model_dir

        # Note: Training data and features are now cached in the dataset folder
        # for reuse in future training runs

    def _load_feature_extractor(self, model_size: str):
        """Load the feature extraction model (using Whisper encoder)."""
        import whisper

        if self._feature_extractor is None:
            # Use Whisper's encoder as a feature extractor
            # This gives us robust audio representations
            self._feature_extractor = whisper.load_model(model_size, device=str(self.device))

    def _extract_features(self, audio_path: Path) -> np.ndarray:
        """Extract semantic features from audio."""
        import whisper

        # Load and preprocess audio for Whisper
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)

        # Get mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        # Extract encoder features
        with torch.no_grad():
            import torch
            features = self._feature_extractor.encoder(mel.unsqueeze(0))
            features = features.squeeze(0).cpu().numpy()

        # Average pool to fixed size
        target_length = 256
        if features.shape[0] > target_length:
            # Downsample
            indices = np.linspace(0, features.shape[0] - 1, target_length, dtype=int)
            features = features[indices]
        elif features.shape[0] < target_length:
            # Pad
            pad_length = target_length - features.shape[0]
            features = np.pad(features, ((0, pad_length), (0, 0)), mode='edge')

        return features

    def _create_voice_model(self, config: TrainingConfig):
        """Create the voice adaptation model."""
        import torch
        import torch.nn as nn

        # Simple voice adaptation network
        # This learns to transform generic features into speaker-specific ones
        class VoiceAdapterModel(nn.Module):
            def __init__(self, feature_dim: int = 512):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(feature_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, feature_dim),
                )
                # Speaker embedding layer
                self.speaker_embedding = nn.Parameter(torch.randn(1, 512) * 0.01)

            def forward(self, x):
                # x: (batch, seq_len, feature_dim)
                encoded = self.encoder(x)
                # Add speaker embedding
                speaker = self.speaker_embedding.expand(encoded.shape[0], encoded.shape[1], -1)
                combined = encoded + speaker
                decoded = self.decoder(combined)
                return decoded

        return VoiceAdapterModel(feature_dim=512)

    def _save_checkpoint(self, model, optimizer, epoch: int, loss: float, config: TrainingConfig):
        """Save a training checkpoint."""
        import torch

        checkpoint_dir = self.models_dir / config.voice_name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)

    def _create_cancelled_result(self, voice_name: str) -> TrainingResult:
        """Create a result for cancelled training."""
        return TrainingResult(
            success=False,
            voice_name=voice_name,
            error_message="Training was cancelled"
        )

    def get_training_requirements(self, dataset_name: str) -> Dict[str, Any]:
        """
        Check if a dataset meets training requirements.

        Returns:
            Dictionary with requirements status.
        """
        dataset = self.dataset_manager.load_dataset(dataset_name)
        if dataset is None:
            return {
                "valid": False,
                "error": "Dataset not found"
            }

        approved_clips = dataset.get_approved_clips()
        approved_duration = dataset.approved_duration

        # Requirements
        min_clips = 5
        min_duration = 30  # 30 seconds minimum
        recommended_duration = 300  # 5 minutes recommended

        issues = []
        warnings = []

        if len(approved_clips) < min_clips:
            issues.append(f"Need at least {min_clips} approved clips (have {len(approved_clips)})")

        if approved_duration < min_duration:
            issues.append(f"Need at least {min_duration}s of audio (have {approved_duration:.1f}s)")

        if approved_duration < recommended_duration:
            warnings.append(f"Recommend at least {recommended_duration}s for best results (have {approved_duration:.1f}s)")

        # Check for cached features
        cache_info = self.get_cache_info(dataset_name)

        return {
            "valid": len(issues) == 0,
            "clip_count": len(approved_clips),
            "duration_seconds": approved_duration,
            "issues": issues,
            "warnings": warnings,
            "quality_estimate": self._estimate_quality(approved_duration, len(approved_clips)),
            "has_cached_features": cache_info["has_cache"],
            "cache_info": cache_info,
        }

    def get_cache_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Check if cached training data exists for a dataset.

        Returns:
            Dictionary with cache status and info.
        """
        dataset = self.dataset_manager.load_dataset(dataset_name)
        if dataset is None:
            return {"has_cache": False, "error": "Dataset not found"}

        cache_path = dataset.path / "features_cache.npz"
        meta_path = dataset.path / "features_cache_meta.json"
        export_dir = dataset.path / "training_export"

        result = {
            "has_cache": False,
            "has_export": export_dir.exists(),
            "cache_path": str(cache_path) if cache_path.exists() else None,
            "export_path": str(export_dir) if export_dir.exists() else None,
            "cached_items": 0,
            "model_size": None,
            "created_at": None,
        }

        if cache_path.exists() and meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                result["has_cache"] = True
                result["cached_items"] = meta.get("num_items", 0)
                result["model_size"] = meta.get("model_size")
                result["created_at"] = meta.get("created_at")
            except Exception:
                pass

        return result

    def clear_cache(self, dataset_name: str) -> bool:
        """
        Clear cached features for a dataset (forces re-extraction on next train).

        Returns:
            True if cache was cleared.
        """
        dataset = self.dataset_manager.load_dataset(dataset_name)
        if dataset is None:
            return False

        cache_path = dataset.path / "features_cache.npz"
        meta_path = dataset.path / "features_cache_meta.json"

        cleared = False
        if cache_path.exists():
            cache_path.unlink()
            cleared = True
        if meta_path.exists():
            meta_path.unlink()
            cleared = True

        return cleared

    def _estimate_quality(self, duration: float, clip_count: int) -> str:
        """Estimate expected training quality based on data."""
        if duration >= 600 and clip_count >= 100:
            return "excellent"
        elif duration >= 300 and clip_count >= 50:
            return "good"
        elif duration >= 120 and clip_count >= 20:
            return "fair"
        elif duration >= 30 and clip_count >= 5:
            return "basic"
        else:
            return "insufficient"


def create_quick_voice(
    name: str,
    reference_audio: Path,
    reference_text: str,
    models_dir: Optional[Path] = None
) -> Path:
    """
    Create a quick voice model from reference audio (no training).

    This is useful for zero-shot voice cloning where you just need
    to set up reference audio for GPT-SoVITS inference.

    Args:
        name: Voice name.
        reference_audio: Path to reference audio file.
        reference_text: Transcript of reference audio.
        models_dir: Output directory for models.

    Returns:
        Path to the created model directory.
    """
    if models_dir is None:
        models_dir = Path.home() / "mira_voice_studio" / "models" / "custom"

    model_dir = models_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Copy reference audio
    ref_dest = model_dir / f"reference{reference_audio.suffix}"
    shutil.copy2(reference_audio, ref_dest)

    # Save reference text
    ref_text_path = model_dir / "reference.txt"
    ref_text_path.write_text(reference_text, encoding="utf-8")

    # Create metadata
    metadata = {
        "name": name,
        "created_at": datetime.now().isoformat(),
        "language": "en",
        "description": "Quick voice from reference audio (zero-shot)",
        "training_quality": "reference_only",
        "reference_text": reference_text,
    }

    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return model_dir


# Import torch at module level for type hints
try:
    import torch
except ImportError:
    torch = None
