"""Gradio user interface for the AI Meeting Minutes Generator.

This module is purely presentational: it builds the Gradio ``Blocks``
layout and wires callbacks to the ``MeetingMinutesPipeline``. All
business logic (transcription, summarization, validation, exporting)
lives in ``src.services`` and ``src.utils`` — the UI layer only
translates between Gradio events and that pipeline.
"""

from __future__ import annotations

import gradio as gr

from src.config import settings
from src.models.schemas import MeetingMetadata
from src.services.minutes_generator import MeetingMinutesPipeline
from src.ui.theme import CUSTOM_CSS, build_theme
from src.utils.exceptions import MeetingMinutesError
from src.utils.exporters import (
    export_minutes_markdown,
    export_minutes_plain_text,
    export_transcript,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _discover_example_audio_files() -> list[str]:
    """Return paths to any sample media files bundled under ``examples/``.

    Keeping this dynamic rather than hard-coding filenames means users
    can drop their own supported media files into the ``examples/``
    folder and have them picked up automatically.
    """

    examples_dir = settings.paths.examples_dir

    if not examples_dir.exists():
        return []

    return sorted(
        str(path)
        for path in examples_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in settings.limits.allowed_audio_extensions
        )
    )


# The pipeline is instantiated once at import time.
# Its underlying services still lazy-load their models on first use,
# so this does not trigger model downloads at import time.
_pipeline = MeetingMinutesPipeline()


_STATUS_IDLE = "⚪ Ready. Upload or record a meeting to get started."

_STATUS_RUNNING = (
    "🟡 Processing... this may take a few minutes on first run "
    "while models download."
)

_STATUS_SUCCESS = "✅ Meeting minutes generated successfully."


def _error_status_message(error: Exception) -> str:
    """Build a user-facing status message for a raised exception."""

    if isinstance(error, MeetingMinutesError):
        return f"❌ {error}"

    logger.exception(
        "Unexpected error in the meeting minutes pipeline."
    )

    return (
        "❌ An unexpected error occurred. Please check the application "
        f"logs for details. ({type(error).__name__}: {error})"
    )


def generate_meeting_minutes(
    uploaded_file_path: str | None,
    recorded_audio_path: str | None,
    title: str,
    meeting_date: str,
    location: str,
    attendees: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),  # noqa: B008
) -> tuple[
    str,
    str,
    str,
    str,
    str | None,
    str | None,
    str | None,
]:
    """Callback wired to the 'Generate Minutes' button.

    The uploaded file is preferred when both an uploaded file and a
    microphone recording are provided.

    Returns:
        (
            status,
            transcript_text,
            minutes_markdown,
            minutes_raw_text,
            transcript_file,
            minutes_md_file,
            minutes_txt_file,
        )
    """

    # --------------------------------------------------------------
    # Choose the media source.
    # --------------------------------------------------------------
    media_path = uploaded_file_path or recorded_audio_path

    if not media_path:
        return (
            "❌ Please upload a media file or record a meeting first.",
            "",
            "",
            "",
            None,
            None,
            None,
        )

    # --------------------------------------------------------------
    # Build meeting metadata.
    # --------------------------------------------------------------
    metadata = MeetingMetadata(
        title=title or "",
        date=meeting_date or "",
        location=location or "",
        attendees=attendees or "",
    )

    # --------------------------------------------------------------
    # Progress callback.
    # --------------------------------------------------------------
    def on_progress(
        fraction: float,
        message: str,
    ) -> None:
        progress(
            fraction,
            desc=message,
        )

    # --------------------------------------------------------------
    # Run the complete pipeline.
    # --------------------------------------------------------------
    try:
        progress(
            0.0,
            desc="Starting...",
        )

        logger.info(
            "Starting meeting minutes generation for '%s'.",
            media_path,
        )

        result = _pipeline.generate(
            media_path,
            metadata=metadata,
            on_progress=on_progress,
        )

    except Exception as exc:  # noqa: BLE001
        status = _error_status_message(exc)

        return (
            status,
            "",
            "",
            "",
            None,
            None,
            None,
        )

    # --------------------------------------------------------------
    # Export generated files.
    # --------------------------------------------------------------
    display_title = result.metadata.display_title()

    try:
        transcript_path = export_transcript(
            result.transcript,
            display_title,
        )

        minutes_md_path = export_minutes_markdown(
            result.markdown,
            display_title,
        )

        minutes_txt_path = export_minutes_plain_text(
            result.markdown,
            display_title,
        )

    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to write export files; downloads will be unavailable."
        )

        transcript_path = None
        minutes_md_path = None
        minutes_txt_path = None

    # --------------------------------------------------------------
    # Return results to Gradio.
    # --------------------------------------------------------------
    return (
        _STATUS_SUCCESS,
        result.transcript,
        result.markdown,
        result.markdown,
        str(transcript_path) if transcript_path else None,
        str(minutes_md_path) if minutes_md_path else None,
        str(minutes_txt_path) if minutes_txt_path else None,
    )


def clear_all() -> tuple:
    """Callback wired to the 'Clear' button."""

    return (
        None,  # uploaded file
        None,  # recorded audio
        "",  # title
        "",  # date
        "",  # location
        "",  # attendees
        _STATUS_IDLE,  # status
        "",  # transcript textbox
        "",  # minutes markdown
        "",  # minutes raw textbox
        None,  # transcript file
        None,  # minutes md file
        None,  # minutes txt file
    )


def _build_header() -> None:
    """Build the application header."""

    logo_path = settings.paths.logo_path

    with gr.Row(elem_id="app-header"):
        with gr.Column():
            if logo_path.exists():
                gr.Image(
                    value=str(logo_path),
                    show_label=False,
                    interactive=False,
                    container=False,
                    height=72,
                    width=72,
                )

            gr.Markdown(
                f"# 🗒️ {settings.metadata.title}\n"
                f"<p>{settings.metadata.tagline}</p>"
            )


def _build_footer() -> None:
    """Build the application footer and About section."""

    with gr.Accordion(
        "ℹ️ About this application",
        open=False,
    ):
        gr.Markdown(
            f"""
{settings.metadata.description}

**How it works**

1. 🎙️ Upload an audio or video recording, or record one with your microphone.
2. 📝 [OpenAI Whisper](
https://huggingface.co/openai/whisper-medium.en
) transcribes the media locally.
3. 🤖 [Meta Llama 3.2 3B Instruct](
https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
) drafts structured meeting minutes from the transcript.
4. 📥 Preview, copy, or download the transcript and minutes in Markdown or plain text.

**Privacy** — Audio and transcripts are processed in-memory / in a local temp
directory for this session and are not sent to any third-party API; all
inference runs on open-source models loaded directly from Hugging Face.

"""
        )

    gr.Markdown(
        f"""
<div class="app-footer">

Built with ❤️ using Gradio, Whisper, and Llama 3.2 &nbsp;|&nbsp;

<a href="{settings.metadata.repository_url}" target="_blank">Source on GitHub</a>

</div>

"""
    )


def build_interface() -> gr.Blocks:
    """Construct and return the full Gradio ``Blocks`` application."""

    theme = build_theme()

    with gr.Blocks(
        title=settings.metadata.title,
        theme=theme,
        css=CUSTOM_CSS,
        fill_height=True,
    ) as demo:
        _build_header()

        with gr.Row(equal_height=False):
            # ========================================================
            # Left column: inputs
            # ========================================================
            with gr.Column(
                scale=1,
                min_width=360,
            ):
                gr.Markdown("### 1. Provide your meeting media")

                # ----------------------------------------------------
                # Upload media file.
                #
                # gr.File is used instead of gr.Audio because
                # gr.Audio rejects MP4 before our backend can process it.
                # ----------------------------------------------------
                uploaded_file = gr.File(
                    label="Upload audio or video",
                    file_types=[
                        ".mp3",
                        ".wav",
                        ".m4a",
                        ".flac",
                        ".ogg",
                        ".webm",
                        ".mp4",
                        ".mpeg",
                        ".mpga",
                    ],
                    type="filepath",
                )

                # ----------------------------------------------------
                # Microphone recording.
                # ----------------------------------------------------
                recorded_audio = gr.Audio(
                    label="Or record with your microphone",
                    sources=["microphone"],
                    type="filepath",
                )

                gr.Markdown("### 2. Optional meeting details")

                with gr.Group():
                    title_input = gr.Textbox(
                        label="Meeting Title",
                        placeholder="e.g. Q3 Budget Review",
                    )

                    with gr.Row():
                        date_input = gr.Textbox(
                            label="Meeting Date",
                            placeholder="e.g. 2026-09-10",
                        )

                        location_input = gr.Textbox(
                            label="Meeting Location",
                            placeholder="e.g. City Hall / Zoom",
                        )

                    attendees_input = gr.Textbox(
                        label="Attendees",
                        placeholder="e.g. Alice, Bob, Carla",
                        lines=2,
                    )

                # ----------------------------------------------------
                # Buttons.
                # ----------------------------------------------------
                with gr.Row():
                    clear_button = gr.Button(
                        "🗑️ Clear",
                        variant="secondary",
                    )

                    generate_button = gr.Button(
                        "✨ Generate Minutes",
                        variant="primary",
                    )

                # ----------------------------------------------------
                # Status.
                # ----------------------------------------------------
                status_box = gr.Textbox(
                    value=_STATUS_IDLE,
                    label="Status",
                    interactive=False,
                    elem_id="status-box",
                )

                # ----------------------------------------------------
                # Example files.
                # ----------------------------------------------------
                example_audio_files = _discover_example_audio_files()

                if example_audio_files:
                    gr.Examples(
                        examples=[
                            [path]
                            for path in example_audio_files
                        ],
                        inputs=[uploaded_file],
                        label="Example recordings",
                    )

                else:
                    gr.Markdown(
                        "_Tip: drop a sample recording into the `examples/` "
                        "folder to have it appear here as a quick-start "
                        "example._"
                    )

            # ========================================================
            # Right column: outputs
            # ========================================================
            with gr.Column(
                scale=2,
                min_width=480,
            ):
                with gr.Tab("📋 Meeting Minutes"):
                    minutes_markdown = gr.Markdown(
                        value=(
                            "_Your generated meeting minutes "
                            "will appear here._"
                        ),
                        elem_id="minutes-preview",
                    )

                    minutes_raw_text = gr.Textbox(
                        label="Raw Markdown (for copying)",
                        lines=8,
                        show_copy_button=True,
                        interactive=False,
                    )

                    with gr.Row():
                        minutes_md_file = gr.File(
                            label="Download minutes (.md)",
                            interactive=False,
                            visible=True,
                        )

                        minutes_txt_file = gr.File(
                            label="Download minutes (.txt)",
                            interactive=False,
                            visible=True,
                        )

                with gr.Tab("🎧 Transcript"):
                    transcript_text = gr.Textbox(
                        label="Full Transcript",
                        lines=18,
                        show_copy_button=True,
                        interactive=False,
                        elem_id="transcript-preview",
                    )

                    transcript_file = gr.File(
                        label="Download transcript (.txt)",
                        interactive=False,
                        visible=True,
                    )

        _build_footer()

        # ============================================================
        # Event wiring
        # ============================================================
        generate_button.click(
            fn=generate_meeting_minutes,
            inputs=[
                uploaded_file,
                recorded_audio,
                title_input,
                date_input,
                location_input,
                attendees_input,
            ],
            outputs=[
                status_box,
                transcript_text,
                minutes_markdown,
                minutes_raw_text,
                transcript_file,
                minutes_md_file,
                minutes_txt_file,
            ],
            api_name="generate_minutes",
        )

        clear_button.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                uploaded_file,
                recorded_audio,
                title_input,
                date_input,
                location_input,
                attendees_input,
                status_box,
                transcript_text,
                minutes_markdown,
                minutes_raw_text,
                transcript_file,
                minutes_md_file,
                minutes_txt_file,
            ],
        )

    return demo