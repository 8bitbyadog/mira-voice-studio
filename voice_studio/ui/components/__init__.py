"""
UI components for Mira Voice Studio.

Phase 3+ implementation:
- Waveform display
- Recording interface
- Teleprompter
- Help overlay
- Automation lanes (Phase 8)
"""

from voice_studio.ui.components.automation_panel import (
    create_automation_panel,
    create_automation_state,
    update_sentence_list,
    get_automation_project,
)

__all__ = [
    "create_automation_panel",
    "create_automation_state",
    "update_sentence_list",
    "get_automation_project",
]
