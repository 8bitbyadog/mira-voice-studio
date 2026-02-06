"""
Generate Tab for Mira Voice Studio.

Handles text-to-speech generation with preview and export options.
"""

import gradio as gr
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import tempfile
import threading

from voice_studio.core.text_processor import TextProcessor
from voice_studio.core.tts_coqui import CoquiTTS
from voice_studio.core.tts_gptsovits import GPTSoVITS
from voice_studio.core.audio_stitcher import AudioStitcher
from voice_studio.core.aligner import WhisperAligner
from voice_studio.core.srt_generator import SRTGenerator
from voice_studio.core.output_manager import OutputManager, GenerationResult
from voice_studio.core.selection import Selection, SelectionExporter
from voice_studio.core.manifest import ManifestGenerator
from voice_studio.core.automation import AutomationProject
from voice_studio.ui.components.automation_panel import (
    create_automation_panel,
    update_sentence_list,
    get_automation_project,
)
from voice_studio.utils.settings import get_settings
from voice_studio.utils.audio_utils import format_duration
from voice_studio.utils.file_utils import open_in_finder


# Global state for the current generation
_current_result: Optional[GenerationResult] = None
_current_tts_results: List = []
_generation_lock = threading.Lock()


def get_available_voices() -> List[str]:
    """Get list of available voices from both engines."""
    voices = []

    # Coqui voices
    try:
        coqui = CoquiTTS()
        for voice in coqui.list_voices():
            voices.append(f"coqui:{voice}")
    except Exception:
        pass

    # GPT-SoVITS voices
    try:
        gptsovits = GPTSoVITS()
        for voice in gptsovits.list_voices():
            voices.append(f"gptsovits:{voice}")
    except Exception:
        pass

    # Default if nothing found
    if not voices:
        voices = ["coqui:default"]

    return voices


def estimate_duration(text: str) -> str:
    """Estimate audio duration from text."""
    processor = TextProcessor()
    sentences = processor.process(text)
    stats = processor.get_stats(sentences)
    return stats.get("estimated_duration", "0:00")


def get_text_stats(text: str) -> str:
    """Get statistics about the input text."""
    if not text.strip():
        return "Word count: 0 | Est. duration: 0:00"

    processor = TextProcessor()
    sentences = processor.process(text)
    stats = processor.get_stats(sentences)

    return f"Word count: {stats['word_count']} | Sentences: {stats['sentence_count']} | Est. duration: {stats['estimated_duration']}"


def generate_voiceover(
    text: str,
    voice: str,
    speed: float,
    output_folder: str,
    word_level: bool,
    automation_state: Dict = None,
    progress: gr.Progress = gr.Progress()
) -> Tuple[Optional[str], str, str, Dict]:
    """
    Generate voiceover from text with optional automation.

    Returns:
        Tuple of (audio_path, status_message, output_info, generation_data)
    """
    global _current_result, _current_tts_results

    if not text.strip():
        return None, "Please enter some text to generate.", "", {}

    with _generation_lock:
        try:
            progress(0, desc="Initializing...")

            # Parse automation state
            automation = get_automation_project(automation_state) if automation_state else None
            use_automation = automation is not None

            # Parse voice selection
            if ":" in voice:
                engine, voice_name = voice.split(":", 1)
            else:
                engine, voice_name = "coqui", voice

            # Initialize TTS engine
            progress(0.1, desc=f"Loading {engine} engine...")

            if engine == "gptsovits":
                tts = GPTSoVITS()
            else:
                tts = CoquiTTS()

            tts.load_voice(voice_name)

            # Initialize aligner if word-level requested
            aligner = None
            if word_level:
                progress(0.15, desc="Loading Whisper for alignment...")
                aligner = WhisperAligner(model_size="base")

            # Determine output path
            if output_folder:
                output_dir = Path(output_folder)
            else:
                settings = get_settings()
                output_dir = Path(settings.get("output_dir", str(Path.home() / "Videos" / "VO")))

            # Initialize output manager
            output_manager = OutputManager(
                base_output_dir=output_dir,
                sample_rate=44100,
                pause_ms=300,
            )

            # Process text
            progress(0.2, desc="Processing text...")
            processor = TextProcessor()
            sentences = processor.process(text)

            # Generate TTS with per-sentence automation
            total_sentences = len(sentences)
            _current_tts_results = []

            for i, sentence in enumerate(sentences):
                progress_pct = 0.2 + (0.6 * (i / total_sentences))
                progress(progress_pct, desc=f"Generating sentence {i + 1}/{total_sentences}...")

                # Get speed for this sentence (automation or global)
                if use_automation:
                    params = automation.get_sentence_params(sentence.index, total_sentences)
                    sentence_speed = params.get("speed", speed)
                else:
                    sentence_speed = speed

                result = tts.generate(sentence, speed=sentence_speed)
                _current_tts_results.append(result)

            # Stitch audio with automation
            progress(0.85, desc="Stitching audio...")
            stitcher = AudioStitcher(pause_ms=300, sample_rate=44100)

            if use_automation:
                progress(0.85, desc="Stitching with automation...")
                stitched = stitcher.stitch_with_automation(_current_tts_results, automation)
            else:
                stitched = stitcher.stitch(_current_tts_results)

            # Run alignment
            alignment = None
            if aligner:
                progress(0.9, desc="Aligning for word-level captions...")
                try:
                    alignment = aligner.align(stitched.audio_data, stitched.sample_rate)
                except Exception as e:
                    print(f"Alignment warning: {e}")

            # Save outputs
            progress(0.95, desc="Saving files...")

            source_name = "script"
            result = output_manager.generate(
                text=text,
                source_name=source_name,
                tts_engine=tts,
                aligner=None,  # We already have alignment
                speed=speed,
                output_dir=None,
                generate_word_captions=word_level,
                progress_callback=None,
            )

            _current_result = result

            # Cleanup
            tts.unload()
            if aligner:
                aligner.unload_model()

            progress(1.0, desc="Complete!")

            # Build output info
            output_info = f"""
**Output folder:** `{result.output_dir}`

**Files:**
- Audio: `{result.master_audio.name}`
- Captions: `{result.master_srt.name}`
{f"- Word captions: `{result.master_srt_words.name}`" if result.master_srt_words else ""}
- VTT: `{result.master_vtt.name}`

**Stats:**
- {result.manifest.successful_sentences}/{result.manifest.total_sentences} sentences
- Duration: {format_duration(result.manifest.total_duration_seconds)}
"""

            # Generation data for UI state
            gen_data = {
                "output_dir": str(result.output_dir),
                "audio_path": str(result.master_audio),
                "duration": result.manifest.total_duration_seconds,
                "sentences": result.manifest.total_sentences,
                "chunks": [
                    {
                        "index": c.index,
                        "text": c.text[:30] + "..." if len(c.text) > 30 else c.text,
                        "start": c.start_time_in_master,
                        "end": c.end_time_in_master,
                    }
                    for c in result.manifest.chunks
                ]
            }

            return str(result.master_audio), "Generation complete!", output_info, gen_data

        except Exception as e:
            return None, f"Error: {str(e)}", "", {}


def export_selection(
    in_time: float,
    out_time: float,
    gen_data: Dict
) -> str:
    """Export a selection of the audio."""
    global _current_result

    if _current_result is None:
        return "No generation to export from. Generate audio first."

    if in_time >= out_time:
        return "Invalid selection: In point must be before Out point."

    try:
        selection = Selection(in_time=in_time, out_time=out_time)
        exporter = SelectionExporter()

        selection_dir = exporter.export(
            selection=selection,
            master_audio=_current_result.stitched_audio.audio_data,
            sample_rate=_current_result.stitched_audio.sample_rate,
            chunk_timings=_current_result.stitched_audio.chunk_timings,
            output_dir=_current_result.output_dir,
            manifest_path=_current_result.manifest_path,
        )

        in_tc, out_tc = selection.to_timecode()
        return f"Selection exported: {in_tc} → {out_tc}\nFolder: {selection_dir}"

    except Exception as e:
        return f"Export error: {str(e)}"


def export_all_chunks(gen_data: Dict) -> str:
    """Export all individual sentence chunks."""
    global _current_result, _current_tts_results

    if _current_result is None:
        return "No generation to export from. Generate audio first."

    if not _current_tts_results:
        return "No TTS results available for chunk export."

    try:
        output_manager = OutputManager()
        chunks_dir = output_manager.export_chunks(_current_result, _current_tts_results)
        return f"Chunks exported to: {chunks_dir}"

    except Exception as e:
        return f"Export error: {str(e)}"


def open_output_folder(gen_data: Dict) -> str:
    """Open the output folder in Finder."""
    if not gen_data or "output_dir" not in gen_data:
        return "No output folder to open."

    output_dir = Path(gen_data["output_dir"])
    if output_dir.exists():
        open_in_finder(output_dir)
        return f"Opened: {output_dir}"
    else:
        return f"Folder not found: {output_dir}"


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
            gr.Markdown("### Voice Settings", elem_classes=["section-header"])

            # Voice selection
            voice_dropdown = gr.Dropdown(
                choices=get_available_voices(),
                value="coqui:default",
                label="Voice",
                info="Select a voice model",
                elem_id="voice-select"
            )
            components["voice"] = voice_dropdown

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

            # Word-level captions toggle
            word_level_checkbox = gr.Checkbox(
                value=False,
                label="Word-level captions",
                info="Generate captions for each word (slower)"
            )
            components["word_level"] = word_level_checkbox

            gr.Markdown("### Output", elem_classes=["section-header"])

            # Output folder
            output_folder = gr.Textbox(
                value="",
                label="Output Folder",
                placeholder="~/Videos/VO/ (default)",
                info="Leave empty for default location"
            )
            components["output_folder"] = output_folder

            browse_btn = gr.Button("Browse...", size="sm")

        # Right column: Script input
        with gr.Column(scale=2):
            gr.Markdown("### Script", elem_classes=["section-header"])

            # Text input
            script_input = gr.Textbox(
                placeholder="Enter your script here...\n\nTip: Each sentence will become a separate audio segment.",
                lines=10,
                max_lines=20,
                label="",
                elem_classes=["script-input"]
            )
            components["script"] = script_input

            # Stats display
            stats_display = gr.Markdown(
                "Word count: 0 | Est. duration: 0:00",
                elem_classes=["stats-row"]
            )

            # Update stats on text change
            script_input.change(
                fn=get_text_stats,
                inputs=[script_input],
                outputs=[stats_display]
            )

            # Generate button
            generate_btn = gr.Button(
                "Generate",
                variant="primary",
                size="lg",
                elem_id="generate-btn"
            )
            components["generate_btn"] = generate_btn

    # Automation Panel (Ableton-style per-sentence control)
    auto_components, auto_state = create_automation_panel()
    components["automation"] = auto_components
    components["auto_state"] = auto_state

    # Wire up script input to update automation timeline
    script_input.change(
        fn=update_sentence_list,
        inputs=[script_input, auto_state],
        outputs=[auto_state, auto_components["timeline"]]
    )

    # Progress and status
    with gr.Row():
        with gr.Column():
            status_text = gr.Markdown("", elem_classes=["status-text"])
            components["status"] = status_text

    # Generation state
    gen_data = gr.State({})
    components["gen_data"] = gen_data

    # Preview section (shown after generation)
    with gr.Row(visible=False) as preview_section:
        with gr.Column():
            gr.Markdown("### Preview & Export", elem_classes=["section-header"])

            # Audio player
            audio_player = gr.Audio(
                label="Generated Audio",
                type="filepath",
                elem_classes=["audio-player"]
            )
            components["audio"] = audio_player

            # I/O Selection
            with gr.Row():
                with gr.Column(scale=1):
                    in_time = gr.Number(
                        value=0,
                        label="In Point (seconds)",
                        info="Press I to set at playhead"
                    )
                    components["in_time"] = in_time

                with gr.Column(scale=1):
                    out_time = gr.Number(
                        value=0,
                        label="Out Point (seconds)",
                        info="Press O to set at playhead"
                    )
                    components["out_time"] = out_time

            # Selection info
            selection_info = gr.Markdown("Selection: 0:00 → 0:00 (0:00)")

            # Export buttons
            gr.Markdown("### Export Options", elem_classes=["section-header"])

            with gr.Row():
                export_master_btn = gr.Button(
                    "Export Master",
                    variant="secondary",
                    info="Full audio + captions (already saved)"
                )

                export_selection_btn = gr.Button(
                    "Export Selection (Cmd+E)",
                    variant="secondary",
                    elem_classes=["in-point-button"]
                )

                export_chunks_btn = gr.Button(
                    "Export All Chunks (Cmd+Shift+E)",
                    variant="secondary"
                )

            # Export status
            export_status = gr.Markdown("")

            # Output info
            output_info = gr.Markdown("", elem_classes=["output-files"])
            components["output_info"] = output_info

            # Open folder button
            open_folder_btn = gr.Button(
                "Open in Finder",
                size="sm"
            )

    components["preview_section"] = preview_section

    # Wire up the generate button (with automation support)
    generate_btn.click(
        fn=generate_voiceover,
        inputs=[script_input, voice_dropdown, speed_slider, output_folder, word_level_checkbox, auto_state],
        outputs=[audio_player, status_text, output_info, gen_data],
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[preview_section]
    ).then(
        fn=lambda data: data.get("duration", 0) if data else 0,
        inputs=[gen_data],
        outputs=[out_time]
    )

    # Wire up export buttons
    export_selection_btn.click(
        fn=export_selection,
        inputs=[in_time, out_time, gen_data],
        outputs=[export_status]
    )

    export_chunks_btn.click(
        fn=export_all_chunks,
        inputs=[gen_data],
        outputs=[export_status]
    )

    open_folder_btn.click(
        fn=open_output_folder,
        inputs=[gen_data],
        outputs=[export_status]
    )

    # Update selection info when in/out change
    def update_selection_info(in_t: float, out_t: float) -> str:
        if out_t <= in_t:
            return "Selection: Invalid (out must be after in)"
        duration = out_t - in_t
        return f"Selection: {format_duration(in_t)} → {format_duration(out_t)} ({format_duration(duration)})"

    in_time.change(
        fn=update_selection_info,
        inputs=[in_time, out_time],
        outputs=[selection_info]
    )
    out_time.change(
        fn=update_selection_info,
        inputs=[in_time, out_time],
        outputs=[selection_info]
    )

    return components
