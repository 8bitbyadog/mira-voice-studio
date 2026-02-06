"""
UI modules for Mira Voice Studio.

Gradio-based web interface with:
- Generate tab: Text input, voice selection, generation, preview, export
- Train tab: Recording, import, dataset management (Phase 4)
- Models tab: Voice model management (Phase 5)
- Settings tab: Configuration (Phase 6)
"""

from voice_studio.ui.gradio_app import create_app, launch
from voice_studio.ui.tab_generate import create_generate_tab
from voice_studio.ui.tab_train import create_train_tab
from voice_studio.ui.tab_models import create_models_tab
from voice_studio.ui.tab_settings import create_settings_tab
from voice_studio.ui.styles import get_custom_css

__all__ = [
    "create_app",
    "launch",
    "create_generate_tab",
    "create_train_tab",
    "create_models_tab",
    "create_settings_tab",
    "get_custom_css",
]
