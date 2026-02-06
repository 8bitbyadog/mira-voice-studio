"""
Custom CSS styles for Mira Voice Studio UI.
"""


def get_custom_css() -> str:
    """Get custom CSS for the Gradio interface."""
    return """
    /* App Header */
    .app-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid var(--border-color-primary);
        margin-bottom: 1rem;
    }

    .app-header h1 {
        margin: 0;
        font-size: 1.8rem;
        color: var(--body-text-color);
    }

    .app-header p {
        margin: 0.5rem 0 0 0;
        color: var(--body-text-color-subdued);
    }

    /* App Footer */
    .app-footer {
        text-align: center;
        font-size: 0.85rem;
        color: var(--body-text-color-subdued);
        padding: 0.5rem 0;
    }

    .app-footer hr {
        margin: 1rem 0;
    }

    /* Main tabs */
    .main-tab {
        padding: 1rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-color-primary);
    }

    /* Script input */
    .script-input textarea {
        min-height: 200px !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Stats display */
    .stats-row {
        display: flex;
        gap: 1rem;
        padding: 0.5rem 0;
        font-size: 0.9rem;
        color: var(--body-text-color-subdued);
    }

    /* Progress section */
    .progress-section {
        padding: 1rem;
        background: var(--background-fill-secondary);
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Waveform container */
    .waveform-container {
        background: var(--background-fill-primary);
        border: 1px solid var(--border-color-primary);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .waveform-display {
        height: 100px;
        background: linear-gradient(
            to bottom,
            var(--background-fill-secondary) 0%,
            var(--background-fill-primary) 50%,
            var(--background-fill-secondary) 100%
        );
        border-radius: 4px;
        position: relative;
        overflow: hidden;
    }

    /* Timeline controls */
    .timeline-controls {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        padding: 0.5rem 0;
    }

    .time-display {
        font-family: 'SF Mono', Monaco, monospace;
        font-size: 0.9rem;
        color: var(--body-text-color);
    }

    /* Sentence blocks */
    .sentence-blocks {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        padding: 0.5rem;
        background: var(--background-fill-secondary);
        border-radius: 4px;
        max-height: 80px;
        overflow-y: auto;
    }

    .sentence-block {
        padding: 4px 8px;
        background: var(--color-accent);
        color: white;
        border-radius: 4px;
        font-size: 0.75rem;
        cursor: pointer;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 120px;
    }

    .sentence-block:hover {
        opacity: 0.8;
    }

    .sentence-block.selected {
        background: var(--color-accent-soft);
        outline: 2px solid var(--color-accent);
    }

    /* Export buttons */
    .export-buttons {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .export-button {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .export-button .shortcut {
        font-size: 0.75rem;
        color: var(--body-text-color-subdued);
        margin-left: auto;
    }

    /* Output files list */
    .output-files {
        padding: 0.5rem;
        background: var(--background-fill-secondary);
        border-radius: 4px;
    }

    .output-file {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0;
    }

    .output-file .icon {
        width: 20px;
        text-align: center;
    }

    .output-file .name {
        flex: 1;
    }

    .output-file .actions {
        display: flex;
        gap: 0.25rem;
    }

    /* Status messages */
    .status-success {
        color: var(--color-green-500);
    }

    .status-error {
        color: var(--color-red-500);
    }

    .status-warning {
        color: var(--color-yellow-500);
    }

    /* Tooltips */
    [data-tooltip] {
        position: relative;
    }

    [data-tooltip]:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        padding: 0.5rem;
        background: var(--background-fill-primary);
        border: 1px solid var(--border-color-primary);
        border-radius: 4px;
        font-size: 0.8rem;
        white-space: nowrap;
        z-index: 1000;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    /* Help overlay */
    .help-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.8);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
    }

    .help-content {
        background: var(--background-fill-primary);
        border-radius: 12px;
        padding: 2rem;
        max-width: 600px;
        max-height: 80vh;
        overflow-y: auto;
    }

    .help-content h2 {
        margin-top: 0;
    }

    .shortcut-list {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 0.5rem 1rem;
    }

    .shortcut-key {
        font-family: 'SF Mono', Monaco, monospace;
        background: var(--background-fill-secondary);
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }

    /* Loading spinner */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 2px solid var(--border-color-primary);
        border-top-color: var(--color-accent);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Audio player */
    .audio-player {
        width: 100%;
        margin: 0.5rem 0;
    }

    /* Selection display */
    .selection-info {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        background: var(--background-fill-secondary);
        border-radius: 4px;
        font-size: 0.9rem;
    }

    .selection-info .arrow {
        color: var(--body-text-color-subdued);
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stats-row {
            flex-direction: column;
            gap: 0.25rem;
        }

        .timeline-controls {
            flex-wrap: wrap;
        }

        .sentence-blocks {
            max-height: 60px;
        }
    }
    """
