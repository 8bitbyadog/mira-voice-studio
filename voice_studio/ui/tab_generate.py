"""
Generate Tab for Auto Voice.

Simplified TTS generation with custom voice support.
"""

import gradio as gr
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tempfile
import soundfile as sf

from voice_studio.core.text_processor import TextProcessor
from voice_studio.core.tts_edge import EdgeTTS, EDGE_VOICES
from voice_studio.models.manager import ModelManager
from voice_studio.utils.settings import get_settings
from voice_studio.utils.audio_utils import format_duration
from voice_studio.utils.file_utils import open_in_finder


def get_available_voices() -> List[str]:
    """Get list of available voices including custom trained ones."""
    voices = []

    # Custom trained voices (show first - these are the user's voices!)
    try:
        model_manager = ModelManager()
        custom_models = model_manager.list_models("custom")
        for model in custom_models:
            if model.has_reference_audio:
                voices.append(f"custom:{model.name}")
    except Exception as e:
        print(f"Error loading custom voices: {e}")

    # Edge TTS voices (always available as fallback)
    for voice_id, display_name in EDGE_VOICES.items():
        voices.append(f"edge:{voice_id}")

    if not voices:
        voices = ["edge:en-US-AriaNeural"]

    return voices


def get_text_stats(text: str) -> str:
    """Get statistics about the input text."""
    if not text.strip():
        return "Word count: 0 | Est. duration: 0:00"

    processor = TextProcessor()
    sentences = processor.process(text)
    stats = processor.get_stats(sentences)

    return f"Word count: {stats['word_count']} | Sentences: {stats['sentence_count']} | Est. duration: {stats['estimated_duration']}"


def browse_for_folder() -> str:
    """Open a folder picker dialog on macOS."""
    import subprocess
    import sys

    if sys.platform != "darwin":
        return ""

    script = '''
    tell application "System Events"
        activate
        set folderPath to choose folder with prompt "Select Output Folder"
        return POSIX path of folderPath
    end tell
    '''

    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return ""


def generate_voiceover(
    text: str,
    voice: str,
    speed: float,
    output_folder: str,
    progress: gr.Progress = gr.Progress()
) -> Tuple[Optional[str], str, str]:
    """
    Generate voiceover from text.

    Returns:
        Tuple of (audio_path, status_message, output_info)
    """
    if not text.strip():
        return None, "Please enter some text to generate.", ""

    try:
        progress(0, desc="Initializing...")

        # Parse voice selection
        if ":" in voice:
            engine, voice_name = voice.split(":", 1)
        else:
            engine, voice_name = "edge", voice

        # Initialize TTS engine based on voice type
        progress(0.1, desc=f"Loading {engine} voice...")

        if engine == "custom":
            from voice_studio.core.tts_custom import CustomVoiceTTS
            tts = CustomVoiceTTS()
            tts.load_voice(voice_name)
            progress(0.2, desc=f"Loaded custom voice: {voice_name}")
        else:
            tts = EdgeTTS()
            tts.load_voice(voice_name)

        # Generate audio - send full text to TTS (let it handle phrasing)
        progress(0.3, desc="Generating audio...")

        # For custom voices, send all text at once for natural phrasing
        # XTTS handles text splitting internally with enable_text_splitting=True
        full_audio, sample_rate = tts.synthesize(text, speed=speed)

        progress(0.85, desc="Processing audio...")

        # Determine output path
        if output_folder:
            output_dir = Path(output_folder)
        else:
            output_dir = Path.home() / "Desktop"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"voiceover_{timestamp}.wav"

        progress(0.95, desc="Saving file...")

        # Save audio
        sf.write(str(output_path), full_audio, sample_rate)

        # Cleanup
        tts.unload()

        progress(1.0, desc="Complete!")

        # Calculate duration
        duration = len(full_audio) / sample_rate

        # Count sentences for display
        processor = TextProcessor()
        sentences = processor.process(text)
        sentence_count = len(sentences)

        output_info = f"""
**Saved to:** `{output_path}`

**Duration:** {format_duration(duration)}
**Sentences:** {sentence_count}
**Sample rate:** {sample_rate} Hz
"""

        return str(output_path), "Generation complete!", output_info

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}", ""


def open_output_folder_fn(output_folder: str) -> str:
    """Open the output folder in Finder."""
    if output_folder:
        folder = Path(output_folder)
    else:
        folder = Path.home() / "Desktop"

    if folder.exists():
        open_in_finder(folder)
        return f"Opened: {folder}"
    else:
        return f"Folder not found: {folder}"


def create_generate_tab() -> Dict[str, Any]:
    """
    Create the Generate tab UI.

    Returns:
        Dictionary of key components for external access.
    """
    components = {}

    with gr.Row():
        # Left column: Input controls
        with gr.Column(scale=1):
            gr.Markdown("### Voice Settings")

            # Voice selection
            voice_dropdown = gr.Dropdown(
                choices=get_available_voices(),
                value=get_available_voices()[0] if get_available_voices() else "edge:en-US-AriaNeural",
                label="Voice",
                info="Custom voices use XTTS cloning, Edge voices are Microsoft TTS"
            )
            components["voice"] = voice_dropdown

            # Refresh voices button
            refresh_btn = gr.Button("Refresh Voices", size="sm")

            # Speed slider
            speed_slider = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Speed",
                info="Speaking speed (0.5 = slow, 2.0 = fast)"
            )
            components["speed"] = speed_slider

            gr.Markdown("### Output")

            # Output folder
            output_folder = gr.Textbox(
                value="",
                label="Output Folder",
                placeholder="~/Desktop (default)",
                info="Leave empty for Desktop"
            )
            components["output_folder"] = output_folder

            with gr.Row():
                browse_btn = gr.Button("Browse...", size="sm")
                open_folder_btn = gr.Button("Open Folder", size="sm")

        # Right column: Script input
        with gr.Column(scale=2):
            gr.Markdown("### Script")

            # Text input
            script_input = gr.Textbox(
                placeholder="Enter your script here...\n\nTip: Each sentence will become a separate audio segment.",
                lines=10,
                max_lines=20,
                label=""
            )
            components["script"] = script_input

            # Stats display
            stats_display = gr.Markdown("Word count: 0 | Est. duration: 0:00")

            # Generate button
            generate_btn = gr.Button(
                "Generate Voiceover",
                variant="primary",
                size="lg"
            )
            components["generate_btn"] = generate_btn

    # Status and output
    status_text = gr.Markdown("")
    components["status"] = status_text

    # Preview section
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Preview")

            audio_player = gr.Audio(
                label="Generated Audio",
                type="filepath"
            )
            components["audio"] = audio_player

            output_info = gr.Markdown("")
            components["output_info"] = output_info

    # Wire up events
    script_input.change(
        fn=get_text_stats,
        inputs=[script_input],
        outputs=[stats_display]
    )

    browse_btn.click(
        fn=browse_for_folder,
        outputs=[output_folder]
    )

    open_folder_btn.click(
        fn=open_output_folder_fn,
        inputs=[output_folder],
        outputs=[status_text]
    )

    def refresh_voices():
        voices = get_available_voices()
        return gr.update(choices=voices, value=voices[0] if voices else "edge:en-US-AriaNeural")

    refresh_btn.click(
        fn=refresh_voices,
        outputs=[voice_dropdown]
    )

    # Wire up the generate button
    generate_btn.click(
        fn=generate_voiceover,
        inputs=[script_input, voice_dropdown, speed_slider, output_folder],
        outputs=[audio_player, status_text, output_info]
    )

    return components
