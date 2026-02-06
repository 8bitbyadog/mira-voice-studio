"""
Automation Panel UI component for Mira Voice Studio.

Provides Ableton-inspired automation controls:
- Visual timeline showing sentences as clips
- Per-sentence parameter overrides
- Automation presets (fade in/out, speed ramps)
- Lane-based automation curves
"""

import gradio as gr
from typing import Dict, List, Optional, Any, Tuple
import json

from voice_studio.core.automation import (
    AutomationProject,
    AutomationLane,
    AutomationParameter,
    SentenceAutomation,
    InterpolationType,
    PARAMETER_CONFIG,
    create_fade_in,
    create_fade_out,
    create_speed_ramp,
    create_dramatic_pause,
)
from voice_studio.utils.audio_utils import format_duration


def create_automation_state() -> Dict[str, Any]:
    """Create initial automation state."""
    return {
        "enabled": False,
        "project": AutomationProject().to_dict(),
        "sentences": [],
        "selected_sentence": None,
    }


def parse_automation_state(state: Dict[str, Any]) -> AutomationProject:
    """Parse automation state into an AutomationProject."""
    if not state or "project" not in state:
        return AutomationProject()
    return AutomationProject.from_dict(state["project"])


def update_automation_state(
    state: Dict[str, Any],
    project: AutomationProject
) -> Dict[str, Any]:
    """Update state with new project data."""
    state = state.copy()
    state["project"] = project.to_dict()
    return state


def toggle_automation(state: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Toggle automation on/off."""
    state = state.copy()
    state["enabled"] = not state.get("enabled", False)

    if state["enabled"]:
        return state, "Automation: **ON** - Per-sentence parameters will be applied"
    else:
        return state, "Automation: **OFF** - Global parameters will be used"


def update_sentence_list(text: str, state: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Update the sentence list when script changes."""
    from voice_studio.core.text_processor import TextProcessor

    if not text.strip():
        state = state.copy()
        state["sentences"] = []
        return state, ""

    processor = TextProcessor()
    sentences = processor.process(text)

    state = state.copy()
    state["sentences"] = [
        {"index": s.index, "text": s.text, "word_count": s.word_count}
        for s in sentences
    ]

    # Build timeline display
    timeline_html = _build_timeline_display(state)

    return state, timeline_html


def _build_timeline_display(state: Dict[str, Any]) -> str:
    """Build HTML timeline display of sentences."""
    sentences = state.get("sentences", [])
    if not sentences:
        return ""

    project = parse_automation_state(state)
    total = len(sentences)

    lines = ["**Timeline** (click a sentence to edit)\n"]

    for s in sentences[:20]:  # Limit display
        idx = s["index"]
        text_preview = s["text"][:40] + "..." if len(s["text"]) > 40 else s["text"]

        # Get automation for this sentence
        params = project.get_sentence_params(idx, total)

        # Format parameter indicators
        indicators = []
        if params["speed"] != 1.0:
            indicators.append(f"S:{params['speed']:.1f}x")
        if params["volume"] != 1.0:
            indicators.append(f"V:{params['volume']:.1f}")
        if params["pause_after_ms"] != 300:
            indicators.append(f"P:{params['pause_after_ms']}ms")

        indicator_str = " ".join(indicators) if indicators else ""

        # Build line
        line = f"`{idx:02d}` {text_preview}"
        if indicator_str:
            line += f" [{indicator_str}]"

        lines.append(line)

    if len(sentences) > 20:
        lines.append(f"... and {len(sentences) - 20} more")

    return "\n".join(lines)


def select_sentence(
    sentence_idx: int,
    state: Dict[str, Any]
) -> Tuple[Dict[str, Any], float, float, int, int]:
    """Select a sentence for editing and return its current parameters."""
    state = state.copy()
    state["selected_sentence"] = sentence_idx

    project = parse_automation_state(state)
    total = len(state.get("sentences", []))

    params = project.get_sentence_params(sentence_idx, total)

    return (
        state,
        params["speed"],
        params["volume"],
        params["pause_after_ms"],
        params["crossfade_ms"],
    )


def update_sentence_params(
    sentence_idx: int,
    speed: float,
    volume: float,
    pause_ms: int,
    crossfade_ms: int,
    state: Dict[str, Any]
) -> Tuple[Dict[str, Any], str]:
    """Update parameters for a specific sentence."""
    if sentence_idx <= 0:
        return state, "Select a sentence first"

    project = parse_automation_state(state)

    # Set overrides
    project.set_sentence_override(
        sentence_idx,
        speed=speed if speed != 1.0 else None,
        volume=volume if volume != 1.0 else None,
        pause_after_ms=pause_ms if pause_ms != 300 else None,
        crossfade_ms=crossfade_ms if crossfade_ms != 0 else None,
    )

    state = update_automation_state(state, project)

    # Rebuild timeline
    timeline = _build_timeline_display(state)

    return state, timeline


def clear_sentence_params(
    sentence_idx: int,
    state: Dict[str, Any]
) -> Tuple[Dict[str, Any], str, float, float, int, int]:
    """Clear all overrides for a sentence."""
    if sentence_idx <= 0:
        return state, "", 1.0, 1.0, 300, 0

    project = parse_automation_state(state)
    project.clear_sentence_override(sentence_idx)

    state = update_automation_state(state, project)
    timeline = _build_timeline_display(state)

    # Return default values
    return state, timeline, 1.0, 1.0, 300, 0


def apply_preset(
    preset: str,
    state: Dict[str, Any]
) -> Tuple[Dict[str, Any], str]:
    """Apply an automation preset."""
    project = parse_automation_state(state)
    sentences = state.get("sentences", [])

    if not sentences:
        return state, "Add text first before applying presets"

    if preset == "fade_in":
        lane = project.lanes[AutomationParameter.VOLUME]
        create_fade_in(lane, duration=0.1)

    elif preset == "fade_out":
        lane = project.lanes[AutomationParameter.VOLUME]
        create_fade_out(lane, start=0.9)

    elif preset == "fade_in_out":
        lane = project.lanes[AutomationParameter.VOLUME]
        lane.clear()
        lane.add_point(0.0, 0.0, InterpolationType.SMOOTH)
        lane.add_point(0.1, 1.0, InterpolationType.LINEAR)
        lane.add_point(0.9, 1.0, InterpolationType.LINEAR)
        lane.add_point(1.0, 0.0, InterpolationType.SMOOTH)

    elif preset == "speed_up":
        lane = project.lanes[AutomationParameter.SPEED]
        create_speed_ramp(lane, start_speed=0.9, end_speed=1.2)

    elif preset == "slow_down":
        lane = project.lanes[AutomationParameter.SPEED]
        create_speed_ramp(lane, start_speed=1.1, end_speed=0.85)

    elif preset == "dramatic_pauses":
        # Add longer pauses at the end of each quarter
        total = len(sentences)
        quarter = max(1, total // 4)
        for i in range(1, 5):
            idx = min(quarter * i, total)
            project.set_sentence_override(idx, pause_after_ms=800)

    elif preset == "no_pauses":
        # Remove all pauses
        for s in sentences:
            project.set_sentence_override(s["index"], pause_after_ms=0)

    elif preset == "clear_all":
        # Clear all automation
        project = AutomationProject()

    state = update_automation_state(state, project)
    timeline = _build_timeline_display(state)

    return state, timeline


def set_global_defaults(
    speed: float,
    volume: float,
    pause_ms: int,
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """Set global default values."""
    project = parse_automation_state(state)

    project.default_speed = speed
    project.default_volume = volume
    project.default_pause_ms = pause_ms

    return update_automation_state(state, project)


def create_automation_panel() -> Tuple[Dict[str, Any], gr.State]:
    """
    Create the automation panel UI.

    Returns:
        Tuple of (component dict, automation state)
    """
    components = {}

    # Automation state
    auto_state = gr.State(create_automation_state())
    components["state"] = auto_state

    with gr.Accordion("Automation Lanes", open=False) as auto_accordion:
        components["accordion"] = auto_accordion

        # Enable toggle and status
        with gr.Row():
            enable_btn = gr.Button("Toggle Automation", size="sm")
            auto_status = gr.Markdown("Automation: **OFF** - Global parameters will be used")
            components["status"] = auto_status

        enable_btn.click(
            fn=toggle_automation,
            inputs=[auto_state],
            outputs=[auto_state, auto_status]
        )

        gr.Markdown("---")

        with gr.Row():
            # Left: Timeline view
            with gr.Column(scale=2):
                gr.Markdown("**Sentence Timeline**")
                gr.Markdown("_Each sentence appears as a block. Edit parameters below._")

                timeline_display = gr.Markdown("")
                components["timeline"] = timeline_display

                # Presets
                gr.Markdown("**Quick Presets**")
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=[
                            ("Fade In", "fade_in"),
                            ("Fade Out", "fade_out"),
                            ("Fade In + Out", "fade_in_out"),
                            ("Speed Up", "speed_up"),
                            ("Slow Down", "slow_down"),
                            ("Dramatic Pauses", "dramatic_pauses"),
                            ("No Pauses", "no_pauses"),
                            ("Clear All", "clear_all"),
                        ],
                        label="Preset",
                        value=None
                    )
                    apply_preset_btn = gr.Button("Apply", size="sm")

                apply_preset_btn.click(
                    fn=apply_preset,
                    inputs=[preset_dropdown, auto_state],
                    outputs=[auto_state, timeline_display]
                )

            # Right: Sentence editor
            with gr.Column(scale=1):
                gr.Markdown("**Edit Sentence**")

                sentence_select = gr.Number(
                    label="Sentence #",
                    value=1,
                    precision=0,
                    minimum=1,
                    info="Enter sentence number to edit"
                )
                components["sentence_select"] = sentence_select

                # Parameter controls
                sent_speed = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.05,
                    label="Speed",
                    info="Speaking rate for this sentence"
                )

                sent_volume = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.05,
                    label="Volume",
                    info="Volume level (1.0 = normal)"
                )

                sent_pause = gr.Slider(
                    minimum=0,
                    maximum=2000,
                    value=300,
                    step=50,
                    label="Pause After (ms)",
                    info="Silence after this sentence"
                )

                sent_crossfade = gr.Slider(
                    minimum=0,
                    maximum=500,
                    value=0,
                    step=10,
                    label="Crossfade (ms)",
                    info="Blend with next sentence"
                )

                with gr.Row():
                    apply_btn = gr.Button("Apply", variant="primary", size="sm")
                    clear_btn = gr.Button("Clear", size="sm")

                # Wire up sentence selection
                sentence_select.change(
                    fn=select_sentence,
                    inputs=[sentence_select, auto_state],
                    outputs=[auto_state, sent_speed, sent_volume, sent_pause, sent_crossfade]
                )

                # Apply changes
                apply_btn.click(
                    fn=update_sentence_params,
                    inputs=[sentence_select, sent_speed, sent_volume, sent_pause, sent_crossfade, auto_state],
                    outputs=[auto_state, timeline_display]
                )

                # Clear overrides
                clear_btn.click(
                    fn=clear_sentence_params,
                    inputs=[sentence_select, auto_state],
                    outputs=[auto_state, timeline_display, sent_speed, sent_volume, sent_pause, sent_crossfade]
                )

        # Global defaults (collapsed)
        with gr.Accordion("Global Defaults", open=False):
            gr.Markdown("_Default values for sentences without overrides_")

            with gr.Row():
                global_speed = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Default Speed"
                )

                global_volume = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Default Volume"
                )

                global_pause = gr.Slider(
                    minimum=0,
                    maximum=1000,
                    value=300,
                    step=50,
                    label="Default Pause (ms)"
                )

            set_defaults_btn = gr.Button("Set Defaults", size="sm")

            set_defaults_btn.click(
                fn=set_global_defaults,
                inputs=[global_speed, global_volume, global_pause, auto_state],
                outputs=[auto_state]
            )

    return components, auto_state


def get_automation_project(state: Dict[str, Any]) -> Optional[AutomationProject]:
    """Get the AutomationProject from state if automation is enabled."""
    if not state or not state.get("enabled", False):
        return None
    return parse_automation_state(state)
