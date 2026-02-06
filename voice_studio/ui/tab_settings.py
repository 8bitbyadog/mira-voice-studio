"""
Settings Tab for Mira Voice Studio.

Provides UI for configuring:
- Default paths
- Audio settings
- Caption preferences
- Recording settings
- Performance options
- Keyboard shortcuts reference
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any

from voice_studio.utils.settings import get_settings, SettingsManager
from voice_studio.training.recorder import Recorder


def load_current_settings() -> Dict[str, Any]:
    """Load current settings values."""
    settings = get_settings()
    return {
        "output_dir": settings.get("output_dir", ""),
        "training_dir": settings.get("training_dir", ""),
        "models_dir": settings.get("models_dir", ""),
        "sample_rate": settings.get("sample_rate", 44100),
        "pause_between_sentences_ms": settings.get("pause_between_sentences_ms", 300),
        "default_speed": settings.get("default_speed", 1.0),
        "generate_sentence_srt": settings.get("generate_sentence_srt", True),
        "generate_word_srt": settings.get("generate_word_srt", True),
        "generate_vtt": settings.get("generate_vtt", True),
        "whisper_model_size": settings.get("whisper_model_size", "base"),
        "recording_device": settings.get("recording_device", ""),
        "show_tooltips": settings.get("show_tooltips", True),
        "enable_keyboard_shortcuts": settings.get("enable_keyboard_shortcuts", True),
        "show_first_run_tutorial": settings.get("show_first_run_tutorial", True),
    }


def save_settings(
    output_dir: str,
    training_dir: str,
    models_dir: str,
    sample_rate: int,
    pause_ms: int,
    default_speed: float,
    sentence_srt: bool,
    word_srt: bool,
    vtt: bool,
    whisper_size: str,
    recording_device: str,
    show_tooltips: bool,
    keyboard_shortcuts: bool,
    first_run: bool
) -> str:
    """Save settings to file."""
    settings = get_settings()

    settings.update(
        output_dir=output_dir,
        training_dir=training_dir,
        models_dir=models_dir,
        sample_rate=sample_rate,
        pause_between_sentences_ms=pause_ms,
        default_speed=default_speed,
        generate_sentence_srt=sentence_srt,
        generate_word_srt=word_srt,
        generate_vtt=vtt,
        whisper_model_size=whisper_size,
        recording_device=recording_device,
        show_tooltips=show_tooltips,
        enable_keyboard_shortcuts=keyboard_shortcuts,
        show_first_run_tutorial=first_run,
    )

    return "Settings saved!"


def reset_settings() -> Dict[str, Any]:
    """Reset settings to defaults."""
    settings = get_settings()
    settings.reset_to_defaults()
    return load_current_settings()


def get_audio_devices() -> list:
    """Get list of audio input devices."""
    try:
        devices = Recorder.get_input_devices()
        return ["System Default"] + [d['name'] for d in devices]
    except Exception:
        return ["System Default"]


def create_settings_tab() -> Dict[str, Any]:
    """
    Create the Settings tab UI.

    Returns:
        Dictionary of key components.
    """
    components = {}
    current = load_current_settings()

    gr.Markdown("### Settings")

    with gr.Tabs():
        # Paths Tab
        with gr.Tab("Paths"):
            gr.Markdown("**Default Directories**")

            output_dir = gr.Textbox(
                value=current["output_dir"],
                label="Default Output Folder",
                placeholder="~/Videos/VO/",
                info="Where generated audio and captions are saved"
            )

            training_dir = gr.Textbox(
                value=current["training_dir"],
                label="Training Data Folder",
                placeholder="~/mira_voice_studio/training/",
                info="Where recordings and datasets are stored"
            )

            models_dir = gr.Textbox(
                value=current["models_dir"],
                label="Models Folder",
                placeholder="~/mira_voice_studio/models/",
                info="Where voice models are stored"
            )

        # Audio Tab
        with gr.Tab("Audio"):
            gr.Markdown("**Audio Output Settings**")

            sample_rate = gr.Dropdown(
                choices=[22050, 44100, 48000],
                value=current["sample_rate"],
                label="Sample Rate (Hz)",
                info="Higher = better quality, larger files"
            )

            pause_ms = gr.Slider(
                minimum=0,
                maximum=1000,
                value=current["pause_between_sentences_ms"],
                step=50,
                label="Pause Between Sentences (ms)",
                info="Silence gap between sentences"
            )

            default_speed = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=current["default_speed"],
                step=0.1,
                label="Default Speaking Speed",
                info="1.0 = normal speed"
            )

        # Captions Tab
        with gr.Tab("Captions"):
            gr.Markdown("**Caption Generation**")

            sentence_srt = gr.Checkbox(
                value=current["generate_sentence_srt"],
                label="Generate sentence-level SRT",
                info="One caption per sentence"
            )

            word_srt = gr.Checkbox(
                value=current["generate_word_srt"],
                label="Generate word-level SRT",
                info="One caption per word (requires Whisper)"
            )

            vtt = gr.Checkbox(
                value=current["generate_vtt"],
                label="Generate VTT (web format)",
                info="WebVTT format for web video players"
            )

        # Recording Tab
        with gr.Tab("Recording"):
            gr.Markdown("**Recording Settings**")

            recording_device = gr.Dropdown(
                choices=get_audio_devices(),
                value=current["recording_device"] or "System Default",
                label="Default Audio Input Device",
                info="Microphone for voice recording"
            )

            gr.Markdown("*Tip: Use an external microphone for best results*")

        # Performance Tab
        with gr.Tab("Performance"):
            gr.Markdown("**Whisper Model**")

            whisper_size = gr.Radio(
                choices=[
                    ("Tiny (fastest, less accurate)", "tiny"),
                    ("Base (balanced)", "base"),
                    ("Small (slower, more accurate)", "small"),
                    ("Medium (slow, very accurate)", "medium"),
                    ("Large (slowest, most accurate)", "large"),
                ],
                value=current["whisper_model_size"],
                label="Whisper Model Size",
                info="Used for word-level timestamps and transcription"
            )

            gr.Markdown("""
**Model Sizes:**
| Size | Memory | Speed |
|------|--------|-------|
| Tiny | ~1 GB | Very Fast |
| Base | ~1 GB | Fast |
| Small | ~2 GB | Medium |
| Medium | ~5 GB | Slow |
| Large | ~10 GB | Very Slow |
            """)

        # Help Tab
        with gr.Tab("Help & Shortcuts"):
            gr.Markdown("**Accessibility**")

            show_tooltips = gr.Checkbox(
                value=current["show_tooltips"],
                label="Show tooltips on hover"
            )

            keyboard_shortcuts = gr.Checkbox(
                value=current["enable_keyboard_shortcuts"],
                label="Enable keyboard shortcuts"
            )

            first_run = gr.Checkbox(
                value=current["show_first_run_tutorial"],
                label="Show tutorial on next launch"
            )

            gr.Markdown("---")
            gr.Markdown("**Keyboard Shortcuts**")

            gr.Markdown("""
| Shortcut | Action |
|----------|--------|
| **H** | Toggle help overlay |
| **Space** | Play/pause audio preview |
| **I** | Set In point at playhead |
| **O** | Set Out point at playhead |
| **R** | Start/stop recording |
| **Cmd+E** | Export current selection |
| **Cmd+Shift+E** | Export all chunks |
| **Cmd+S** | Save settings |
| **Cmd+,** | Open settings |
            """)

    # Save/Reset buttons
    gr.Markdown("---")

    with gr.Row():
        save_btn = gr.Button("Save Settings", variant="primary")
        reset_btn = gr.Button("Reset to Defaults")

    status = gr.Markdown("")

    # Wire up save
    save_btn.click(
        fn=save_settings,
        inputs=[
            output_dir, training_dir, models_dir,
            sample_rate, pause_ms, default_speed,
            sentence_srt, word_srt, vtt,
            whisper_size, recording_device,
            show_tooltips, keyboard_shortcuts, first_run
        ],
        outputs=[status]
    )

    # Wire up reset
    def do_reset():
        settings = reset_settings()
        return (
            settings["output_dir"],
            settings["training_dir"],
            settings["models_dir"],
            settings["sample_rate"],
            settings["pause_between_sentences_ms"],
            settings["default_speed"],
            settings["generate_sentence_srt"],
            settings["generate_word_srt"],
            settings["generate_vtt"],
            settings["whisper_model_size"],
            settings["recording_device"] or "System Default",
            settings["show_tooltips"],
            settings["enable_keyboard_shortcuts"],
            settings["show_first_run_tutorial"],
            "Settings reset to defaults"
        )

    reset_btn.click(
        fn=do_reset,
        outputs=[
            output_dir, training_dir, models_dir,
            sample_rate, pause_ms, default_speed,
            sentence_srt, word_srt, vtt,
            whisper_size, recording_device,
            show_tooltips, keyboard_shortcuts, first_run,
            status
        ]
    )

    components["save_btn"] = save_btn
    components["status"] = status

    return components
