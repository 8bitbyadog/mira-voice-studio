"""
Mira Voice Studio - Gradio Web Interface

Main application entry point for the web UI.
"""

import gradio as gr
from pathlib import Path
from typing import Optional

from voice_studio.ui.tab_generate import create_generate_tab
from voice_studio.ui.tab_train import create_train_tab
from voice_studio.ui.styles import get_custom_css
from voice_studio.utils.settings import get_settings


def create_app() -> gr.Blocks:
    """
    Create the main Gradio application.

    Returns:
        gr.Blocks: The Gradio application.
    """
    settings = get_settings()

    # Custom CSS for styling
    custom_css = get_custom_css()

    with gr.Blocks(
        title="Mira Voice Studio",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
    ) as app:
        # Header
        gr.Markdown(
            """
            # Mira Voice Studio
            Generate voiceovers with synced captions
            """,
            elem_classes=["app-header"]
        )

        # Help toggle state
        help_visible = gr.State(False)

        # Main tabs
        with gr.Tabs() as tabs:
            # Generate Tab
            with gr.Tab("Generate", id="generate", elem_classes=["main-tab"]):
                generate_components = create_generate_tab()

            # Train Tab
            with gr.Tab("Train", id="train", elem_classes=["main-tab"]):
                train_components = create_train_tab()

            # Models Tab (placeholder for Phase 5)
            with gr.Tab("Models", id="models", elem_classes=["main-tab"]):
                gr.Markdown(
                    """
                    ## Voice Models

                    *Coming in Phase 5*

                    You'll be able to:
                    - View all installed voices
                    - Test voices with sample text
                    - Import/export voice models
                    - Download new pretrained models
                    """
                )

            # Settings Tab (placeholder for Phase 6)
            with gr.Tab("Settings", id="settings", elem_classes=["main-tab"]):
                gr.Markdown(
                    """
                    ## Settings

                    *Coming in Phase 6*

                    You'll be able to configure:
                    - Default output paths
                    - Audio settings
                    - Caption preferences
                    - Keyboard shortcuts
                    """
                )

        # Footer with keyboard shortcut hint
        gr.Markdown(
            """
            ---
            Press **H** for help | **Space** play/pause | **I/O** set in/out points | **R** record
            """,
            elem_classes=["app-footer"]
        )

        # Keyboard shortcut handling via JavaScript
        app.load(
            fn=None,
            inputs=None,
            outputs=None,
            js=get_keyboard_js()
        )

    return app


def get_keyboard_js() -> str:
    """Get JavaScript for keyboard shortcut handling."""
    return """
    () => {
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            // Ignore if typing in an input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }

            switch(e.key.toLowerCase()) {
                case 'h':
                    // Toggle help (will be implemented with help overlay)
                    console.log('Help toggled');
                    break;
                case ' ':
                    // Play/pause
                    e.preventDefault();
                    const playBtn = document.querySelector('.play-button');
                    if (playBtn) playBtn.click();
                    break;
                case 'i':
                    // Set in point
                    const inBtn = document.querySelector('.in-point-button');
                    if (inBtn) inBtn.click();
                    break;
                case 'o':
                    // Set out point
                    const outBtn = document.querySelector('.out-point-button');
                    if (outBtn) outBtn.click();
                    break;
                case 'r':
                    // Toggle recording
                    const recordBtn = document.querySelector('.record-button');
                    if (recordBtn) recordBtn.click();
                    break;
            }
        });
    }
    """


def launch(
    share: bool = False,
    server_port: int = 7860,
    server_name: str = "127.0.0.1",
    debug: bool = False
) -> None:
    """
    Launch the Gradio application.

    Args:
        share: Create a public link.
        server_port: Port to run on.
        server_name: Server hostname.
        debug: Enable debug mode.
    """
    app = create_app()
    app.launch(
        share=share,
        server_port=server_port,
        server_name=server_name,
        debug=debug,
        show_error=True,
    )


if __name__ == "__main__":
    launch(debug=True)
