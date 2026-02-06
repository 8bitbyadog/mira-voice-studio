"""
Models Tab for Mira Voice Studio.

Provides UI for:
- Viewing installed voice models
- Testing voices with sample text
- Importing/exporting models
- Managing custom voices
"""

import gradio as gr
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tempfile

from voice_studio.models.manager import ModelManager, VoiceModel
from voice_studio.core.tts_coqui import CoquiTTS
from voice_studio.core.tts_gptsovits import GPTSoVITS
from voice_studio.utils.audio_utils import format_duration


def get_model_manager() -> ModelManager:
    """Get or create model manager."""
    return ModelManager()


def list_all_models() -> Tuple[List[str], List[str]]:
    """Get lists of custom and pretrained models."""
    manager = get_model_manager()

    custom = manager.list_models("custom")
    pretrained = manager.list_models("pretrained")

    custom_names = [m.name for m in custom]
    pretrained_names = [m.name for m in pretrained]

    return custom_names, pretrained_names


def get_model_info(model_name: str, model_type: str) -> str:
    """Get detailed info about a model."""
    if not model_name:
        return "Select a model to view details"

    manager = get_model_manager()
    model = manager.get_model(model_name)

    if model is None:
        return "Model not found"

    info = f"""
### {model.name}

**Type:** {model.model_type.title()}
**Language:** {model.language}
"""

    if model.created_at:
        info += f"**Created:** {model.created_at[:10]}\n"

    if model.training_quality:
        info += f"**Training Quality:** {model.training_quality}\n"

    if model.training_duration_hours:
        info += f"**Training Duration:** {model.training_duration_hours:.1f} hours\n"

    if model.description:
        info += f"\n**Description:** {model.description}\n"

    info += "\n**Components:**\n"
    info += f"- Reference Audio: {'✓' if model.has_reference_audio else '✗'}\n"
    info += f"- GPT Model: {'✓' if model.has_gpt_model else '✗'}\n"
    info += f"- SoVITS Model: {'✓' if model.has_sovits_model else '✗'}\n"

    return info


def test_voice(model_name: str, model_type: str, test_text: str) -> Optional[str]:
    """Generate test audio with a voice."""
    if not model_name or not test_text:
        return None

    try:
        # Determine which engine to use
        if model_type == "pretrained":
            # Check if it's a Coqui pretrained model
            tts = CoquiTTS()
            if model_name in tts.list_voices():
                tts.load_voice(model_name)
            else:
                tts.load_voice("default")
        else:
            # Custom models use GPT-SoVITS
            tts = GPTSoVITS()
            try:
                tts.load_voice(model_name)
            except Exception:
                # Fallback to Coqui
                tts = CoquiTTS()
                tts.load_voice("default")

        # Generate audio
        audio, sr = tts.synthesize(test_text)

        # Save to temp file
        temp_path = Path(tempfile.gettempdir()) / f"test_{model_name}.wav"
        import soundfile as sf
        sf.write(str(temp_path), audio, sr)

        tts.unload()

        return str(temp_path)

    except Exception as e:
        print(f"Test voice error: {e}")
        return None


def delete_model(model_name: str) -> Tuple[str, List[str]]:
    """Delete a custom model."""
    if not model_name:
        return "No model selected", []

    manager = get_model_manager()

    if manager.delete_model(model_name):
        custom_names, _ = list_all_models()
        return f"Deleted: {model_name}", custom_names
    else:
        custom_names, _ = list_all_models()
        return f"Failed to delete: {model_name}", custom_names


def rename_model(old_name: str, new_name: str) -> Tuple[str, List[str]]:
    """Rename a custom model."""
    if not old_name or not new_name:
        return "Please provide both names", []

    if old_name == new_name:
        return "Names are the same", []

    manager = get_model_manager()

    if manager.rename_model(old_name, new_name):
        custom_names, _ = list_all_models()
        return f"Renamed: {old_name} → {new_name}", custom_names
    else:
        custom_names, _ = list_all_models()
        return f"Failed to rename: {old_name}", custom_names


def export_model(model_name: str) -> Tuple[str, Optional[str]]:
    """Export a model to a .zip file."""
    if not model_name:
        return "No model selected", None

    manager = get_model_manager()

    try:
        # Export to temp directory
        temp_dir = Path(tempfile.gettempdir())
        zip_path = manager.export_model(model_name, temp_dir)
        return f"Exported: {zip_path.name}", str(zip_path)
    except Exception as e:
        return f"Export failed: {e}", None


def import_model(file_path: str, custom_name: str) -> Tuple[str, List[str]]:
    """Import a model from a .zip file."""
    if not file_path:
        return "No file selected", []

    manager = get_model_manager()

    try:
        name = custom_name if custom_name else None
        model = manager.import_model(Path(file_path), name)
        custom_names, _ = list_all_models()
        return f"Imported: {model.name}", custom_names
    except Exception as e:
        custom_names, _ = list_all_models()
        return f"Import failed: {e}", custom_names


def create_models_tab() -> Dict[str, Any]:
    """
    Create the Models tab UI.

    Returns:
        Dictionary of key components.
    """
    components = {}

    gr.Markdown("### Your Voice Models")

    with gr.Row():
        # Left column: Model lists
        with gr.Column(scale=1):
            gr.Markdown("**Custom Models**")

            custom_names, pretrained_names = list_all_models()

            custom_dropdown = gr.Dropdown(
                choices=custom_names,
                label="Custom Voices",
                info="Your trained voices"
            )

            gr.Markdown("**Pretrained Models**")

            pretrained_dropdown = gr.Dropdown(
                choices=pretrained_names,
                label="Pretrained Voices",
                info="Downloaded voices"
            )

            refresh_btn = gr.Button("Refresh List", size="sm")

        # Right column: Model details
        with gr.Column(scale=2):
            model_info = gr.Markdown("Select a model to view details")

            # Test voice section
            gr.Markdown("---")
            gr.Markdown("**Test Voice**")

            test_text = gr.Textbox(
                value="Hello, this is a test of my voice.",
                label="Test Text",
                lines=2
            )

            test_btn = gr.Button("Generate Test Audio")

            test_audio = gr.Audio(
                label="Test Output",
                type="filepath",
                interactive=False
            )

    # Model management section
    gr.Markdown("---")
    gr.Markdown("### Manage Models")

    with gr.Row():
        with gr.Column():
            gr.Markdown("**Rename Model**")

            rename_old = gr.Textbox(
                label="Current Name",
                placeholder="Select from dropdown above"
            )
            rename_new = gr.Textbox(
                label="New Name",
                placeholder="Enter new name"
            )
            rename_btn = gr.Button("Rename")
            rename_status = gr.Markdown("")

        with gr.Column():
            gr.Markdown("**Delete Model**")

            delete_name = gr.Textbox(
                label="Model to Delete",
                placeholder="Select from dropdown above"
            )
            delete_btn = gr.Button("Delete", variant="stop")
            delete_status = gr.Markdown("")

    gr.Markdown("---")
    gr.Markdown("### Import / Export")

    with gr.Row():
        with gr.Column():
            gr.Markdown("**Export Model**")

            export_name = gr.Textbox(
                label="Model to Export",
                placeholder="Select from dropdown above"
            )
            export_btn = gr.Button("Export as .zip")
            export_status = gr.Markdown("")
            export_file = gr.File(
                label="Download",
                visible=False
            )

        with gr.Column():
            gr.Markdown("**Import Model**")

            import_file = gr.File(
                label="Upload .zip file",
                file_types=[".zip"]
            )
            import_name = gr.Textbox(
                label="Custom Name (optional)",
                placeholder="Leave empty to use archive name"
            )
            import_btn = gr.Button("Import")
            import_status = gr.Markdown("")

    # Wire up interactions
    def refresh_models():
        custom, pretrained = list_all_models()
        return gr.update(choices=custom), gr.update(choices=pretrained)

    refresh_btn.click(
        fn=refresh_models,
        outputs=[custom_dropdown, pretrained_dropdown]
    )

    # Model selection updates
    def on_custom_select(name):
        return (
            get_model_info(name, "custom"),
            name,
            name,
            name
        )

    def on_pretrained_select(name):
        return get_model_info(name, "pretrained")

    custom_dropdown.change(
        fn=on_custom_select,
        inputs=[custom_dropdown],
        outputs=[model_info, rename_old, delete_name, export_name]
    )

    pretrained_dropdown.change(
        fn=on_pretrained_select,
        inputs=[pretrained_dropdown],
        outputs=[model_info]
    )

    # Test voice
    def do_test(custom, pretrained, text):
        if custom:
            return test_voice(custom, "custom", text)
        elif pretrained:
            return test_voice(pretrained, "pretrained", text)
        return None

    test_btn.click(
        fn=do_test,
        inputs=[custom_dropdown, pretrained_dropdown, test_text],
        outputs=[test_audio]
    )

    # Rename
    rename_btn.click(
        fn=rename_model,
        inputs=[rename_old, rename_new],
        outputs=[rename_status, custom_dropdown]
    )

    # Delete
    delete_btn.click(
        fn=delete_model,
        inputs=[delete_name],
        outputs=[delete_status, custom_dropdown]
    )

    # Export
    def do_export(name):
        status, path = export_model(name)
        if path:
            return status, gr.update(value=path, visible=True)
        return status, gr.update(visible=False)

    export_btn.click(
        fn=do_export,
        inputs=[export_name],
        outputs=[export_status, export_file]
    )

    # Import
    def do_import(file, name):
        if file is None:
            return "No file selected", gr.update()
        status, choices = import_model(file.name if hasattr(file, 'name') else file, name)
        return status, gr.update(choices=choices)

    import_btn.click(
        fn=do_import,
        inputs=[import_file, import_name],
        outputs=[import_status, custom_dropdown]
    )

    components["custom_dropdown"] = custom_dropdown
    components["pretrained_dropdown"] = pretrained_dropdown

    return components
