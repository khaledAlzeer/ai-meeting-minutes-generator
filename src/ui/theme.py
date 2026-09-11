"""Custom Gradio theme for the AI Meeting Minutes Generator.

Kept in its own module so the visual identity of the app can be tweaked
without touching layout or wiring logic in ``gradio_app.py``.
"""

from __future__ import annotations

import gradio as gr


def build_theme() -> gr.themes.Base:
    """Build the application's custom Gradio theme.

    Based on the built-in ``Soft`` theme with an indigo/teal accent
    palette and slightly larger radii for a modern, professional look.
    """
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.slate,
        font=(
            gr.themes.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
        font_mono=(
            gr.themes.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ),
    ).set(
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_700",
        button_primary_text_color="white",
        block_title_text_weight="600",
        block_border_width="1px",
        block_shadow="*shadow_drop_lg",
        block_radius="*radius_lg",
    )


CUSTOM_CSS: str = """
#app-header {
    text-align: center;
    padding: 1.25rem 1rem 0.5rem 1rem;
}
#app-header h1 {
    margin-bottom: 0.25rem;
    font-weight: 700;
}
#app-header p {
    color: var(--body-text-color-subdued);
    font-size: 1.02rem;
}
#status-box textarea {
    font-weight: 500;
}
#minutes-preview {
    max-height: 640px;
    overflow-y: auto;
    padding: 0.5rem 1rem;
}
#transcript-preview textarea {
    max-height: 420px;
}
.app-footer {
    text-align: center;
    padding-top: 1.5rem;
    color: var(--body-text-color-subdued);
    font-size: 0.85rem;
}
.app-footer a {
    color: var(--body-text-color-subdued);
    text-decoration: underline;
}
"""
