"""
Train Tab for Mira Voice Studio.

Provides UI for:
- Recording voice samples with script teleprompter
- Importing audio files
- Managing datasets
- Starting training
"""

import gradio as gr
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tempfile
import threading
import time

from voice_studio.training.recorder import Recorder, RecordingSession
from voice_studio.training.script_generator import ScriptGenerator
from voice_studio.training.preprocessor import AudioPreprocessor
from voice_studio.training.dataset import DatasetManager
from voice_studio.training.trainer import (
    VoiceTrainer,
    TrainingConfig,
    TrainingQuality,
    TrainingProgress,
)
from voice_studio.utils.settings import get_settings
from voice_studio.utils.audio_utils import format_duration


# Global state for clip review
_current_clip_index: int = 0
_clip_list: List[Path] = []


def get_voice_clips(voice_name: str) -> List[Dict]:
    """Get list of clips for a voice model with metadata."""
    if not voice_name:
        return []

    voice_dir = Path.home() / "mira_voice_studio" / "models" / "custom" / voice_name / "references"
    if not voice_dir.exists():
        return []

    clips = []
    for clip_path in sorted(voice_dir.glob("*.wav")):
        try:
            import soundfile as sf
            info = sf.info(str(clip_path))
            duration = info.duration

            # Check if clip has actual audio (not silence)
            audio, sr = sf.read(str(clip_path))
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            rms = np.sqrt(np.mean(audio**2))
            is_silent = rms < 0.01  # Very quiet = likely silent

            clips.append({
                "path": str(clip_path),
                "name": clip_path.name,
                "duration": duration,
                "is_silent": is_silent,
                "rms": rms
            })
        except Exception as e:
            print(f"Error reading {clip_path}: {e}")

    return clips


def get_clip_list_display(voice_name: str) -> str:
    """Get formatted list of clips for display."""
    clips = get_voice_clips(voice_name)
    if not clips:
        return "No clips found"

    lines = []
    for i, clip in enumerate(clips):
        status = "⚠️ SILENT" if clip["is_silent"] else "✓"
        lines.append(f"{i+1}. {clip['name']} - {clip['duration']:.1f}s {status}")

    silent_count = sum(1 for c in clips if c["is_silent"])
    header = f"**{len(clips)} clips** ({silent_count} potentially silent)\n\n"

    return header + "\n".join(lines)


def load_clip_for_preview(voice_name: str, clip_index: int) -> Tuple[Optional[str], str]:
    """Load a specific clip for preview."""
    clips = get_voice_clips(voice_name)
    if not clips or clip_index < 1 or clip_index > len(clips):
        return None, "Invalid clip index"

    clip = clips[clip_index - 1]
    status = f"**{clip['name']}**\nDuration: {clip['duration']:.2f}s\nLevel: {clip['rms']:.4f}"
    if clip["is_silent"]:
        status += "\n⚠️ **This clip appears to be silent!**"

    return clip["path"], status


def delete_clip(voice_name: str, clip_index: int) -> Tuple[str, str, None, str]:
    """Delete a clip from the voice model."""
    clips = get_voice_clips(voice_name)
    if not clips or clip_index < 1 or clip_index > len(clips):
        return "❌ Invalid clip index", get_clip_list_display(voice_name), None, ""

    clip = clips[clip_index - 1]
    clip_path = Path(clip["path"])

    try:
        clip_path.unlink()
        remaining = len(clips) - 1
        status = f"✅ **DELETED:** {clip['name']}\n\n*{remaining} clips remaining*"
        return status, get_clip_list_display(voice_name), None, ""
    except Exception as e:
        return f"❌ Error deleting: {e}", get_clip_list_display(voice_name), None, ""


def detect_silence_boundaries(voice_name: str, clip_index: int, threshold: float = 0.02) -> Tuple[float, float, float]:
    """
    Detect silence at start and end of clip.
    Returns (suggested_start, suggested_end, total_duration).
    """
    clips = get_voice_clips(voice_name)
    if not clips or clip_index < 1 or clip_index > len(clips):
        return 0.0, 1.0, 1.0

    clip = clips[clip_index - 1]
    try:
        import soundfile as sf
        audio, sr = sf.read(clip["path"])
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        duration = len(audio) / sr

        # Find where audio starts (first sample above threshold)
        window_size = int(sr * 0.05)  # 50ms windows
        start_sample = 0
        for i in range(0, len(audio) - window_size, window_size // 2):
            window_rms = np.sqrt(np.mean(audio[i:i+window_size]**2))
            if window_rms > threshold:
                start_sample = max(0, i - window_size)  # Back up slightly
                break

        # Find where audio ends (last sample above threshold)
        end_sample = len(audio)
        for i in range(len(audio) - window_size, window_size, -window_size // 2):
            window_rms = np.sqrt(np.mean(audio[i:i+window_size]**2))
            if window_rms > threshold:
                end_sample = min(len(audio), i + window_size * 2)  # Add buffer
                break

        start_time = start_sample / sr
        end_time = end_sample / sr

        return start_time, end_time, duration

    except Exception as e:
        print(f"Error detecting silence: {e}")
        return 0.0, 1.0, 1.0


def create_trim_preview(voice_name: str, clip_index: int, start_time: float, end_time: float) -> Optional[str]:
    """Create a temporary trimmed audio file for preview."""
    clips = get_voice_clips(voice_name)
    if not clips or clip_index < 1 or clip_index > len(clips):
        return None

    clip = clips[clip_index - 1]
    try:
        import soundfile as sf
        audio, sr = sf.read(clip["path"])
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        trimmed = audio[start_sample:end_sample]

        # Save to temp file
        temp_path = Path(tempfile.gettempdir()) / f"trim_preview_{clip_index}.wav"
        sf.write(str(temp_path), trimmed.astype(np.float32), sr)

        return str(temp_path)
    except Exception as e:
        print(f"Error creating preview: {e}")
        return None


def save_trimmed_clip(voice_name: str, clip_index: int, start_time: float, end_time: float) -> Tuple[str, str]:
    """Save the trimmed audio using start/end times."""
    clips = get_voice_clips(voice_name)
    if not clips or clip_index < 1 or clip_index > len(clips):
        return "❌ Invalid clip index", get_clip_list_display(voice_name)

    clip = clips[clip_index - 1]
    clip_path = Path(clip["path"])
    original_duration = clip["duration"]

    try:
        import soundfile as sf
        audio, sr = sf.read(str(clip_path))

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        trimmed = audio[start_sample:end_sample]

        new_duration = len(trimmed) / sr

        if new_duration < 0.5:
            return "❌ Trimmed clip too short (min 0.5s)", get_clip_list_display(voice_name)

        # Save back to same file
        sf.write(str(clip_path), trimmed.astype(np.float32), sr)

        status = f"✅ **SAVED:** {clip['name']}\n\nNew duration: {new_duration:.2f}s (was {original_duration:.2f}s)"

        return status, get_clip_list_display(voice_name)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error saving: {e}", get_clip_list_display(voice_name)


def regenerate_combined_reference(voice_name: str) -> str:
    """Regenerate the combined reference after editing clips."""
    if not voice_name:
        return "Select a voice first"

    voice_dir = Path.home() / "mira_voice_studio" / "models" / "custom" / voice_name
    refs_dir = voice_dir / "references"

    if not refs_dir.exists():
        return "No references folder found"

    try:
        import soundfile as sf

        all_refs = sorted(refs_dir.glob("*.wav"))
        if not all_refs:
            return "No reference clips found"

        combined_audio = []
        sr = 44100
        pause = np.zeros(int(sr * 0.3), dtype=np.float32)

        for ref_path in all_refs:
            audio, ref_sr = sf.read(str(ref_path))
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            if ref_sr != sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=ref_sr, target_sr=sr)
            combined_audio.append(audio.astype(np.float32))
            combined_audio.append(pause)

        combined = np.concatenate(combined_audio)
        combined_path = voice_dir / "reference_combined.wav"
        sf.write(str(combined_path), combined, sr)

        # Update metadata
        import json
        metadata_path = voice_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {"name": voice_name}

        metadata["reference_count"] = len(all_refs)
        metadata["total_reference_duration"] = len(combined) / sr

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return f"Regenerated combined reference\n{len(all_refs)} clips, {len(combined)/sr:.1f}s total"

    except Exception as e:
        return f"Error: {e}"


# Global recorder instance
_recorder: Optional[Recorder] = None
_script_lines: List = []
_current_script_index: int = 0


def get_recorder() -> Recorder:
    """Get or create the global recorder."""
    global _recorder
    if _recorder is None:
        _recorder = Recorder()
    return _recorder


def get_input_devices() -> List[str]:
    """Get list of audio input device names."""
    devices = Recorder.get_input_devices()
    return [d['name'] for d in devices]


def generate_script(script_type: str, num_sentences: int) -> Tuple[str, str]:
    """Generate a training script."""
    global _script_lines, _current_script_index

    generator = ScriptGenerator()
    _script_lines = generator.generate(script_type, num_sentences)
    _current_script_index = 0

    if _script_lines:
        current = _script_lines[0]
        script_display = f'"{current.text}"'
        nav_display = f"1 / {len(_script_lines)}"
    else:
        script_display = "No script generated"
        nav_display = "0 / 0"

    return script_display, nav_display


def navigate_script(direction: str) -> Tuple[str, str]:
    """Navigate through script lines."""
    global _current_script_index

    if not _script_lines:
        return "No script loaded", "0 / 0"

    if direction == "prev":
        _current_script_index = max(0, _current_script_index - 1)
    elif direction == "next":
        _current_script_index = min(len(_script_lines) - 1, _current_script_index + 1)

    current = _script_lines[_current_script_index]
    script_display = f'"{current.text}"'

    if current.emotion:
        script_display = f"[{current.emotion.upper()}]\n{script_display}"

    nav_display = f"{_current_script_index + 1} / {len(_script_lines)}"

    return script_display, nav_display


def get_current_script_text() -> str:
    """Get the current script line text."""
    if _script_lines and 0 <= _current_script_index < len(_script_lines):
        return _script_lines[_current_script_index].text
    return ""


def start_recording_session(session_name: str, device: str) -> str:
    """Start a new recording session."""
    recorder = get_recorder()

    if device:
        recorder.set_device(device)

    recorder.start_session(session_name or "recording")
    recorder.start_monitoring()

    return f"Session started: {session_name or 'recording'}"


def toggle_recording() -> Tuple[str, str, Optional[str]]:
    """Toggle recording on/off."""
    recorder = get_recorder()

    if recorder.is_recording():
        # Stop recording
        script_text = get_current_script_text()
        take = recorder.stop_recording(
            script_text=script_text,
            script_index=_current_script_index
        )

        if take:
            # Save to temp file for playback
            temp_path = Path(tempfile.gettempdir()) / f"take_{take.index}.wav"
            import soundfile as sf
            sf.write(str(temp_path), take.audio_data, take.sample_rate)

            session = recorder.get_session()
            if session:
                session_info = f"Takes: {session.take_count} | Total: {format_duration(session.total_duration)}"
            else:
                session_info = "Take recorded"

            return "Start Recording (R)", session_info, str(temp_path)

        return "Start Recording (R)", "Recording stopped", None

    else:
        # Start recording - ensure session exists first
        session = recorder.get_session()
        if session is None:
            # Auto-start a session if none exists
            recorder.start_session("recording")

        if recorder.start_recording():
            return "Stop Recording (R)", "Recording...", None
        else:
            return "Start Recording (R)", "Failed to start recording", None


def get_level_display() -> Tuple[float, str, str]:
    """Get current audio level for display."""
    recorder = get_recorder()
    current_db, peak_db = recorder.get_level()
    status, message = recorder.get_level_indicator()

    # Normalize to 0-100 for progress bar
    level_pct = max(0, min(100, (current_db + 60) * 100 / 60))

    return level_pct, f"{current_db:.1f} dB", message


def finish_session(output_dir: str) -> str:
    """Finish recording session and save."""
    recorder = get_recorder()
    session = recorder.get_session()

    if session is None:
        return "No active session"

    if not session.takes:
        return "No takes recorded"

    settings = get_settings()
    base_dir = Path(output_dir or settings.get("training_dir") or
                    Path.home() / "mira_voice_studio" / "training" / "recordings")

    try:
        session_dir = recorder.save_session(base_dir)
        recorder.stop_monitoring()
        return f"Session saved to: {session_dir}\n{session.take_count} takes, {format_duration(session.total_duration)} total"
    except Exception as e:
        return f"Error saving session: {e}"


def process_uploaded_files(files: List[str], transcribe: bool) -> Tuple[str, str]:
    """Process uploaded audio files."""
    if not files:
        return "No files uploaded", ""

    preprocessor = AudioPreprocessor()

    file_paths = [Path(f) for f in files]

    status_lines = []

    def progress_callback(message: str, progress: float):
        status_lines.append(f"{message} ({progress * 100:.0f}%)")

    result = preprocessor.process_files(
        file_paths,
        transcribe=transcribe,
        progress_callback=progress_callback
    )

    # Save as dataset
    settings = get_settings()
    datasets_dir = Path(settings.get("training_dir") or
                        Path.home() / "mira_voice_studio" / "training" / "datasets")

    from datetime import datetime
    dataset_name = f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    dataset_dir = preprocessor.save_dataset(result, datasets_dir, dataset_name)

    summary = f"""
**Processing Complete**

- Files processed: {len(files)}
- Clips created: {result.clip_count}
- Total duration: {format_duration(result.total_duration)}
- Dataset saved: {dataset_name}

{f"Errors: {len(result.errors)}" if result.errors else ""}
"""

    clip_list = "\n".join([
        f"{c.index}. [{format_duration(c.duration)}] {c.transcript[:50]}..."
        for c in result.clips[:20]
    ])

    if len(result.clips) > 20:
        clip_list += f"\n... and {len(result.clips) - 20} more"

    return summary, clip_list


def get_datasets() -> List[str]:
    """Get list of available datasets."""
    manager = DatasetManager()
    return manager.list_datasets()


def load_dataset_info(dataset_name: str) -> Tuple[str, str]:
    """Load and display dataset information."""
    if not dataset_name:
        return "Select a dataset", ""

    manager = DatasetManager()
    dataset = manager.load_dataset(dataset_name)

    if dataset is None:
        return "Dataset not found", ""

    info = f"""
**{dataset.name}**

- Total clips: {dataset.clip_count}
- Approved: {dataset.approved_count}
- Rejected: {dataset.rejected_count}
- Duration: {format_duration(dataset.approved_duration)}
- Sample rate: {dataset.sample_rate} Hz
"""

    clips_display = "\n".join([
        f"{'✓' if c.approved else '✗'} {c.index:04d} [{format_duration(c.duration)}] {c.transcript[:40]}..."
        for c in dataset.clips[:30]
    ])

    if len(dataset.clips) > 30:
        clips_display += f"\n... and {len(dataset.clips) - 30} more"

    return info, clips_display


def toggle_clip_approval(dataset_name: str, clip_index: int) -> str:
    """Toggle a clip's approval status."""
    manager = DatasetManager()
    dataset = manager.load_dataset(dataset_name)

    if dataset is None:
        return "Dataset not found"

    clip = dataset.get_clip(clip_index)
    if clip is None:
        return "Clip not found"

    if clip.approved:
        dataset.reject_clip(clip_index)
        status = "rejected"
    else:
        dataset.approve_clip(clip_index)
        status = "approved"

    manager.save_dataset(dataset)

    return f"Clip {clip_index} {status}"


def check_training_requirements(dataset_name: str) -> str:
    """Check if dataset meets training requirements."""
    if not dataset_name:
        return "Select a dataset first"

    trainer = VoiceTrainer()
    reqs = trainer.get_training_requirements(dataset_name)

    if not reqs["valid"]:
        issues_text = "\n".join([f"- {i}" for i in reqs["issues"]])
        return f"""
**Cannot train**

Issues:
{issues_text}
"""

    warnings_text = ""
    if reqs["warnings"]:
        warnings_text = "\n".join([f"- {w}" for w in reqs["warnings"]])
        warnings_text = f"\n\n**Warnings:**\n{warnings_text}"

    # Show cache status
    cache_text = ""
    if reqs.get("has_cached_features"):
        cache_info = reqs.get("cache_info", {})
        cache_text = f"""

**Cached Features Available**
- {cache_info.get('cached_items', 0)} cached audio features
- Model: {cache_info.get('model_size', 'unknown')}
- Created: {cache_info.get('created_at', 'unknown')[:10] if cache_info.get('created_at') else 'unknown'}

*Feature extraction will be skipped - training will start faster!*
"""
    else:
        cache_text = """

**No Cached Features**
- First training run will extract features (slower)
- Features will be cached for future training runs
"""

    return f"""
**Ready to train**

- Clips: {reqs["clip_count"]}
- Duration: {format_duration(reqs["duration_seconds"])}
- Expected quality: {reqs["quality_estimate"]}
{warnings_text}
{cache_text}
"""


def clear_feature_cache(dataset_name: str) -> str:
    """Clear cached features for a dataset."""
    if not dataset_name:
        return "Select a dataset first"

    trainer = VoiceTrainer()
    if trainer.clear_cache(dataset_name):
        return "Feature cache cleared. Next training will re-extract features."
    else:
        return "No cache to clear or dataset not found."


# Global trainer instance for stopping
_current_trainer: Optional[VoiceTrainer] = None


def run_training(
    dataset_name: str,
    voice_name: str,
    quality: str,
    progress=gr.Progress()
) -> str:
    """Run the training process with progress updates."""
    global _current_trainer

    if not dataset_name:
        return "**Error:** Please select a dataset"
    if not voice_name:
        return "**Error:** Please enter a voice model name"

    # Validate voice name
    voice_name = voice_name.strip().replace(" ", "_")
    if not voice_name:
        return "**Error:** Invalid voice name"

    # Check requirements
    trainer = VoiceTrainer()
    _current_trainer = trainer

    reqs = trainer.get_training_requirements(dataset_name)
    if not reqs["valid"]:
        issues = "\n".join([f"- {i}" for i in reqs["issues"]])
        return f"**Cannot train:**\n{issues}"

    # Map quality string to enum
    quality_map = {
        "quick": TrainingQuality.QUICK,
        "standard": TrainingQuality.STANDARD,
        "high": TrainingQuality.HIGH,
    }
    training_quality = quality_map.get(quality, TrainingQuality.STANDARD)

    # Create config
    config = TrainingConfig(
        dataset_name=dataset_name,
        voice_name=voice_name,
        quality=training_quality,
    )

    # Run training with progress updates
    try:
        result = None
        for prog in trainer.train(config):
            # Update Gradio progress
            progress(prog.progress, desc=f"{prog.stage}: {prog.message}")

            # Yield intermediate status
            if prog.stage == "training":
                status = f"""
**Training in progress...**

- Stage: {prog.stage}
- Epoch: {prog.epoch}/{prog.total_epochs}
- Loss: {prog.loss:.4f}
- Progress: {prog.progress * 100:.1f}%

{prog.message}
"""
            else:
                status = f"""
**{prog.stage.title()}...**

{prog.message}

Progress: {prog.progress * 100:.1f}%
"""

        # Get final result
        if hasattr(trainer, '_final_loss'):
            result_info = f"""
**Training Complete!**

- Voice Model: **{voice_name}**
- Epochs: {config.epochs}
- Final Loss: {trainer._final_loss:.4f}
- Quality: {quality}

Your voice model is now available in the **Models** tab.

*You can use it by selecting "{voice_name}" in the Generate tab.*
"""
        else:
            result_info = f"""
**Training Complete!**

- Voice Model: **{voice_name}**
- Quality: {quality}

Your voice model is now available in the **Models** tab.
"""

        return result_info

    except Exception as e:
        return f"**Training failed:**\n\n{str(e)}"
    finally:
        _current_trainer = None


def stop_training() -> str:
    """Stop the current training process."""
    global _current_trainer
    if _current_trainer:
        _current_trainer.stop()
        return "Stopping training... Please wait."
    return "No training in progress"


def get_existing_voices() -> List[str]:
    """Get list of existing custom voice models."""
    try:
        from voice_studio.models.manager import ModelManager
        manager = ModelManager()
        models = manager.list_models("custom")
        return [m.name for m in models if m.has_reference_audio]
    except Exception:
        return []


def add_recordings_to_voice(voice_name: str) -> str:
    """Add current session recordings to an existing voice model."""
    recorder = get_recorder()
    session = recorder.get_session()

    if session is None or not session.takes:
        return "No recordings to add. Record some samples first."

    if not voice_name:
        return "Select a voice model first."

    # Get voice model references folder
    from pathlib import Path
    voice_dir = Path.home() / "mira_voice_studio" / "models" / "custom" / voice_name
    refs_dir = voice_dir / "references"

    if not voice_dir.exists():
        return f"Voice model not found: {voice_name}"

    refs_dir.mkdir(exist_ok=True)

    # Find the next index
    existing = list(refs_dir.glob("*.wav"))
    next_idx = len(existing) + 1

    # Save each take to the references folder
    import soundfile as sf
    added_count = 0
    for take in session.takes:
        ref_path = refs_dir / f"ref_{next_idx:03d}.wav"
        sf.write(str(ref_path), take.audio_data, take.sample_rate)
        next_idx += 1
        added_count += 1

    # Regenerate combined reference
    try:
        all_refs = sorted(refs_dir.glob("*.wav"))
        if all_refs:
            import numpy as np
            combined_audio = []
            sr = 44100
            pause = np.zeros(int(sr * 0.3), dtype=np.float32)

            for ref_path in all_refs:
                audio, ref_sr = sf.read(str(ref_path))
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                if ref_sr != sr:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=ref_sr, target_sr=sr)
                combined_audio.append(audio.astype(np.float32))
                combined_audio.append(pause)

            combined = np.concatenate(combined_audio)
            combined_path = voice_dir / "reference_combined.wav"
            sf.write(str(combined_path), combined, sr)

            # Update metadata
            import json
            metadata_path = voice_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
            else:
                metadata = {"name": voice_name}

            metadata["reference_count"] = len(all_refs)
            metadata["total_reference_duration"] = len(combined) / sr

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

    except Exception as e:
        print(f"Warning: Could not regenerate combined reference: {e}")

    # Clear session
    recorder.stop_monitoring()

    return f"Added {added_count} recordings to '{voice_name}' voice model!\nTotal references: {next_idx - 1}\n\nYou can now test the improved voice in the Generate tab."


def create_train_tab() -> Dict[str, Any]:
    """
    Create the Train tab UI with sub-tabs.

    Returns:
        Dictionary of key components.
    """
    components = {}

    with gr.Tabs() as train_tabs:
        # Record Sub-tab
        with gr.Tab("Record", id="record"):
            gr.Markdown("### Record Your Voice")
            gr.Markdown("Read the script aloud to add more training samples.")

            with gr.Row():
                with gr.Column(scale=2):
                    # Script display
                    gr.Markdown("**Script to read:**")

                    with gr.Row():
                        script_type = gr.Dropdown(
                            choices=["mixed", "phoneme", "emotional", "conversational"],
                            value="mixed",
                            label="Script Type"
                        )
                        num_sentences = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=25,
                            step=5,
                            label="Sentences"
                        )
                        generate_btn = gr.Button("Generate Script")

                    script_display = gr.Textbox(
                        value='"Click Generate Script to start"',
                        label="",
                        lines=4,
                        interactive=False,
                        elem_classes=["script-teleprompter"]
                    )

                    with gr.Row():
                        prev_btn = gr.Button("◀ Previous")
                        nav_display = gr.Markdown("0 / 0")
                        next_btn = gr.Button("Next ▶")

                with gr.Column(scale=1):
                    # Recording controls
                    gr.Markdown("**Recording**")

                    device_dropdown = gr.Dropdown(
                        choices=get_input_devices(),
                        label="Audio Input",
                        value=Recorder.get_default_input_device()
                    )

                    session_name = gr.Textbox(
                        label="Session Name",
                        value="my_voice",
                        placeholder="e.g., my_voice"
                    )

                    start_session_btn = gr.Button("Start Session")
                    session_status = gr.Markdown("")

                    # Level meter (simplified)
                    level_display = gr.Markdown("Level: -- dB")

                    record_btn = gr.Button(
                        "Start Recording (R)",
                        variant="primary",
                        elem_classes=["record-button"]
                    )

                    # Last take preview
                    last_take_audio = gr.Audio(
                        label="Last Take",
                        type="filepath",
                        interactive=False
                    )

                    # Session info
                    takes_display = gr.Markdown("Takes: 0 | Total: 0:00")

                    gr.Markdown("---")
                    gr.Markdown("**Add to Existing Voice**")

                    existing_voice_dropdown = gr.Dropdown(
                        choices=get_existing_voices(),
                        label="Select Voice Model",
                        info="Add recordings to improve this voice"
                    )

                    add_to_voice_btn = gr.Button(
                        "Add Recordings to Voice",
                        variant="primary"
                    )

                    add_status = gr.Markdown("")

                    gr.Markdown("---")
                    finish_btn = gr.Button("Finish Session (New Dataset)")
                    finish_status = gr.Markdown("")

            # Wire up recording controls
            generate_btn.click(
                fn=generate_script,
                inputs=[script_type, num_sentences],
                outputs=[script_display, nav_display]
            )

            prev_btn.click(
                fn=lambda: navigate_script("prev"),
                outputs=[script_display, nav_display]
            )

            next_btn.click(
                fn=lambda: navigate_script("next"),
                outputs=[script_display, nav_display]
            )

            start_session_btn.click(
                fn=start_recording_session,
                inputs=[session_name, device_dropdown],
                outputs=[session_status]
            )

            record_btn.click(
                fn=toggle_recording,
                outputs=[record_btn, takes_display, last_take_audio]
            )

            add_to_voice_btn.click(
                fn=add_recordings_to_voice,
                inputs=[existing_voice_dropdown],
                outputs=[add_status]
            )

            finish_btn.click(
                fn=finish_session,
                inputs=[gr.Textbox(visible=False, value="")],
                outputs=[finish_status]
            )

        # Import Sub-tab
        with gr.Tab("Import", id="import"):
            gr.Markdown("### Import Audio Files")
            gr.Markdown("Upload audio files to create a training dataset.")

            with gr.Row():
                with gr.Column():
                    file_upload = gr.File(
                        label="Drop audio files here",
                        file_count="multiple",
                        file_types=["audio"],
                        type="filepath"
                    )

                    transcribe_checkbox = gr.Checkbox(
                        value=True,
                        label="Auto-transcribe with Whisper",
                        info="Slower but creates transcripts automatically"
                    )

                    process_btn = gr.Button("Process Files", variant="primary")

                with gr.Column():
                    process_status = gr.Markdown("")
                    clips_preview = gr.Textbox(
                        label="Processed Clips",
                        lines=10,
                        interactive=False
                    )

            process_btn.click(
                fn=process_uploaded_files,
                inputs=[file_upload, transcribe_checkbox],
                outputs=[process_status, clips_preview]
            )

        # Review Clips Sub-tab
        with gr.Tab("Review Clips", id="review"):
            gr.Markdown("### Review & Clean Voice Clips")
            gr.Markdown("Preview clips, trim silence, delete bad ones, then regenerate the combined reference.")

            with gr.Row():
                with gr.Column(scale=1):
                    review_voice_dropdown = gr.Dropdown(
                        choices=get_existing_voices(),
                        label="Select Voice Model",
                        info="Choose a voice to review its clips"
                    )

                    refresh_clips_btn = gr.Button("Refresh List", size="sm")

                    clip_list_display = gr.Markdown("Select a voice model to see clips")

                with gr.Column(scale=2):
                    gr.Markdown("**Clip Preview**")

                    with gr.Row():
                        clip_num_input = gr.Number(
                            label="Clip #",
                            value=1,
                            precision=0,
                            minimum=1
                        )
                        load_clip_btn = gr.Button("Load Clip")

                    # Read-only audio preview (just for listening)
                    clip_audio_preview = gr.Audio(
                        label="Original Clip",
                        type="filepath",
                        interactive=False
                    )

                    clip_info_display = gr.Markdown("")

                    with gr.Row():
                        prev_clip_btn = gr.Button("◀ Prev")
                        next_clip_btn = gr.Button("Next ▶")

                    gr.Markdown("---")
                    gr.Markdown("**Trim Controls**")

                    # Hidden state for clip duration
                    clip_duration_state = gr.State(value=5.0)

                    with gr.Row():
                        trim_start = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=0,
                            step=0.01,
                            label="Start Time (sec)",
                            info="Drag to set where audio should start"
                        )
                        trim_end = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=10,
                            step=0.01,
                            label="End Time (sec)",
                            info="Drag to set where audio should end"
                        )

                    trim_info = gr.Markdown("*Adjust sliders to trim, or click Auto-Trim*")

                    with gr.Row():
                        auto_trim_btn = gr.Button("Auto-Trim Silence", size="sm")
                        preview_trim_btn = gr.Button("Preview Trim", size="sm")

                    # Trimmed preview player
                    trim_preview_audio = gr.Audio(
                        label="Trimmed Preview",
                        type="filepath",
                        interactive=False,
                        visible=True
                    )

                    with gr.Row():
                        save_trim_btn = gr.Button("Save Trim", variant="primary")
                        delete_clip_btn = gr.Button("Delete Clip", variant="stop")

                    trim_status = gr.Markdown("")

                    gr.Markdown("---")

                    gr.Markdown("*After editing clips, regenerate the combined reference:*")
                    regenerate_btn = gr.Button(
                        "Regenerate Combined Reference",
                        variant="primary"
                    )

                    regenerate_status = gr.Markdown("")

            # Wire up review controls
            def refresh_clip_list(voice_name):
                return get_clip_list_display(voice_name)

            review_voice_dropdown.change(
                fn=refresh_clip_list,
                inputs=[review_voice_dropdown],
                outputs=[clip_list_display]
            )

            refresh_clips_btn.click(
                fn=refresh_clip_list,
                inputs=[review_voice_dropdown],
                outputs=[clip_list_display]
            )

            def load_clip_with_sliders(voice_name, clip_idx):
                """Load clip and set up trim sliders based on duration."""
                audio_path, info = load_clip_for_preview(voice_name, int(clip_idx))
                clips = get_voice_clips(voice_name)
                if clips and 0 < clip_idx <= len(clips):
                    duration = clips[int(clip_idx) - 1]["duration"]
                    return (
                        audio_path,
                        info,
                        gr.update(maximum=duration, value=0),
                        gr.update(maximum=duration, value=duration),
                        duration,
                        None,
                        ""
                    )
                return audio_path, info, gr.update(), gr.update(), 5.0, None, ""

            load_clip_btn.click(
                fn=load_clip_with_sliders,
                inputs=[review_voice_dropdown, clip_num_input],
                outputs=[clip_audio_preview, clip_info_display, trim_start, trim_end, clip_duration_state, trim_preview_audio, trim_status]
            )

            def nav_clip_with_sliders(voice_name, current_idx, direction):
                clips = get_voice_clips(voice_name)
                if direction == "prev":
                    new_idx = max(1, int(current_idx) - 1)
                else:
                    new_idx = min(len(clips), int(current_idx) + 1)
                audio_path, info = load_clip_for_preview(voice_name, new_idx)
                if clips and 0 < new_idx <= len(clips):
                    duration = clips[new_idx - 1]["duration"]
                    return (
                        new_idx,
                        audio_path,
                        info,
                        gr.update(maximum=duration, value=0),
                        gr.update(maximum=duration, value=duration),
                        duration,
                        None,
                        ""
                    )
                return new_idx, audio_path, info, gr.update(), gr.update(), 5.0, None, ""

            prev_clip_btn.click(
                fn=lambda v, i: nav_clip_with_sliders(v, i, "prev"),
                inputs=[review_voice_dropdown, clip_num_input],
                outputs=[clip_num_input, clip_audio_preview, clip_info_display, trim_start, trim_end, clip_duration_state, trim_preview_audio, trim_status]
            )

            next_clip_btn.click(
                fn=lambda v, i: nav_clip_with_sliders(v, i, "next"),
                inputs=[review_voice_dropdown, clip_num_input],
                outputs=[clip_num_input, clip_audio_preview, clip_info_display, trim_start, trim_end, clip_duration_state, trim_preview_audio, trim_status]
            )

            def do_auto_trim(voice_name, clip_idx):
                """Auto-detect silence and set trim points."""
                start, end, duration = detect_silence_boundaries(voice_name, int(clip_idx))
                trimmed_duration = end - start
                info = f"Auto-detected: **{start:.2f}s** to **{end:.2f}s** (trimmed duration: {trimmed_duration:.2f}s)"
                return gr.update(value=start), gr.update(value=end), info

            auto_trim_btn.click(
                fn=do_auto_trim,
                inputs=[review_voice_dropdown, clip_num_input],
                outputs=[trim_start, trim_end, trim_info]
            )

            def do_preview_trim(voice_name, clip_idx, start_time, end_time):
                """Create and return trimmed preview."""
                preview_path = create_trim_preview(voice_name, int(clip_idx), start_time, end_time)
                trimmed_duration = end_time - start_time
                info = f"Preview: **{start_time:.2f}s** to **{end_time:.2f}s** ({trimmed_duration:.2f}s)"
                return preview_path, info

            preview_trim_btn.click(
                fn=do_preview_trim,
                inputs=[review_voice_dropdown, clip_num_input, trim_start, trim_end],
                outputs=[trim_preview_audio, trim_info]
            )

            def do_save_trim(voice_name, clip_idx, start_time, end_time):
                """Save the trim and reload the clip."""
                status, clip_list = save_trimmed_clip(voice_name, int(clip_idx), start_time, end_time)
                # Reload the clip to show updated audio
                audio_path, info = load_clip_for_preview(voice_name, int(clip_idx))
                clips = get_voice_clips(voice_name)
                if clips and 0 < clip_idx <= len(clips):
                    duration = clips[int(clip_idx) - 1]["duration"]
                    return (
                        status,
                        clip_list,
                        audio_path,
                        info,
                        gr.update(maximum=duration, value=0),
                        gr.update(maximum=duration, value=duration),
                        None
                    )
                return status, clip_list, audio_path, info, gr.update(), gr.update(), None

            save_trim_btn.click(
                fn=do_save_trim,
                inputs=[review_voice_dropdown, clip_num_input, trim_start, trim_end],
                outputs=[trim_status, clip_list_display, clip_audio_preview, clip_info_display, trim_start, trim_end, trim_preview_audio]
            )

            delete_clip_btn.click(
                fn=delete_clip,
                inputs=[review_voice_dropdown, clip_num_input],
                outputs=[trim_status, clip_list_display, clip_audio_preview, clip_info_display]
            )

            regenerate_btn.click(
                fn=regenerate_combined_reference,
                inputs=[review_voice_dropdown],
                outputs=[regenerate_status]
            )

        # Datasets Sub-tab
        with gr.Tab("Datasets", id="datasets"):
            gr.Markdown("### Manage Datasets")
            gr.Markdown("Review and edit your training datasets.")

            with gr.Row():
                with gr.Column(scale=1):
                    dataset_dropdown = gr.Dropdown(
                        choices=get_datasets(),
                        label="Select Dataset",
                        interactive=True
                    )

                    refresh_btn = gr.Button("Refresh List")

                    dataset_info = gr.Markdown("")

                with gr.Column(scale=2):
                    clips_display = gr.Textbox(
                        label="Clips",
                        lines=15,
                        interactive=False
                    )

                    with gr.Row():
                        clip_index_input = gr.Number(
                            label="Clip #",
                            value=1,
                            precision=0
                        )
                        toggle_approval_btn = gr.Button("Toggle Approval")
                        approval_status = gr.Markdown("")

            # Training controls
            gr.Markdown("---")
            gr.Markdown("### Train Model")

            with gr.Row():
                with gr.Column():
                    voice_name_input = gr.Textbox(
                        label="Voice Model Name",
                        placeholder="e.g., my_natural_voice"
                    )

                    training_quality = gr.Radio(
                        choices=[
                            ("Quick (~30 min)", "quick"),
                            ("Standard (~2 hours)", "standard"),
                            ("High Quality (~6 hours)", "high"),
                        ],
                        value="standard",
                        label="Training Quality"
                    )

                    with gr.Row():
                        check_btn = gr.Button("Check Requirements")
                        clear_cache_btn = gr.Button("Clear Cache", size="sm")

                    requirements_status = gr.Markdown("")

                    with gr.Row():
                        train_btn = gr.Button(
                            "Start Training",
                            variant="primary"
                        )
                        stop_btn = gr.Button(
                            "Stop Training",
                            variant="stop"
                        )

                with gr.Column():
                    training_status = gr.Markdown(
                        """
                        **Status:** Ready

                        Select a dataset and configure training options above.

                        **Tips:**
                        - More data = better quality
                        - 5+ minutes of audio recommended
                        - Training uses GPU (MPS on Apple Silicon)
                        - Your Mac may be slow during training
                        """
                    )

            # Wire up dataset controls
            def refresh_datasets():
                return gr.update(choices=get_datasets())

            refresh_btn.click(
                fn=refresh_datasets,
                outputs=[dataset_dropdown]
            )

            dataset_dropdown.change(
                fn=load_dataset_info,
                inputs=[dataset_dropdown],
                outputs=[dataset_info, clips_display]
            )

            toggle_approval_btn.click(
                fn=toggle_clip_approval,
                inputs=[dataset_dropdown, clip_index_input],
                outputs=[approval_status]
            ).then(
                fn=load_dataset_info,
                inputs=[dataset_dropdown],
                outputs=[dataset_info, clips_display]
            )

            # Check requirements button
            check_btn.click(
                fn=check_training_requirements,
                inputs=[dataset_dropdown],
                outputs=[requirements_status]
            )

            # Clear cache button
            clear_cache_btn.click(
                fn=clear_feature_cache,
                inputs=[dataset_dropdown],
                outputs=[requirements_status]
            ).then(
                fn=check_training_requirements,
                inputs=[dataset_dropdown],
                outputs=[requirements_status]
            )

            # Start training button
            train_btn.click(
                fn=run_training,
                inputs=[dataset_dropdown, voice_name_input, training_quality],
                outputs=[training_status]
            )

            # Stop training button
            stop_btn.click(
                fn=stop_training,
                outputs=[training_status]
            )

    components["train_tabs"] = train_tabs

    return components
