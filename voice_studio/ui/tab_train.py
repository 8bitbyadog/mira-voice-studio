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
            session_info = f"Takes: {session.take_count} | Total: {format_duration(session.total_duration)}"

            return "Start Recording (R)", session_info, str(temp_path)

        return "Start Recording (R)", "Recording stopped", None

    else:
        # Start recording
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

    return f"""
**Ready to train**

- Clips: {reqs["clip_count"]}
- Duration: {format_duration(reqs["duration_seconds"])}
- Expected quality: {reqs["quality_estimate"]}
{warnings_text}
"""


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
            gr.Markdown("Read the script aloud to create training data.")

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

                    finish_btn = gr.Button("Finish Session")
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

                    check_btn = gr.Button("Check Requirements")
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
