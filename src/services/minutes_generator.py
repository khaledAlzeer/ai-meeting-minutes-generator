"""End-to-end orchestration of the meeting-minutes generation pipeline.

:class:`MeetingMinutesPipeline` is the single entry point the UI layer
talks to. It composes the transcription and summarization services,
reports progress via an optional callback, and returns a single
:class:`MeetingMinutesResult` object.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.models.schemas import MeetingMetadata, MeetingMinutesResult
from src.services.summarization.llm_service import LlamaSummarizationService
from src.services.transcription.whisper_service import WhisperTranscriptionService
from src.utils.logger import get_logger

logger = get_logger(__name__)

# A progress callback receives a float in [0, 1] and a short status message,
# mirroring the signature Gradio's gr.Progress() tracker expects.
ProgressCallback = Callable[[float, str], None]


def _noop_progress(_fraction: float, _message: str) -> None:
    """Default no-op progress callback."""


class MeetingMinutesPipeline:
    """Coordinates transcription and summarization into a single workflow."""

    def __init__(
        self,
        transcription_service: WhisperTranscriptionService | None = None,
        summarization_service: LlamaSummarizationService | None = None,
    ) -> None:
        self.transcription_service = transcription_service or WhisperTranscriptionService()
        self.summarization_service = summarization_service or LlamaSummarizationService()

    def generate(
        self,
        audio_path: str | Path,
        metadata: MeetingMetadata | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> MeetingMinutesResult:
        """Run the full pipeline: transcribe audio, then generate minutes.

        Args:
            audio_path: Path to the uploaded/recorded audio file.
            metadata: Optional meeting metadata supplied by the user.
            on_progress: Optional callback invoked with
                ``(fraction_complete, status_message)`` at each pipeline
                stage, useful for driving a UI progress bar.

        Returns:
            A :class:`MeetingMinutesResult` containing the transcript and
            the generated Markdown minutes.
        """
        progress = on_progress or _noop_progress
        metadata = metadata or MeetingMetadata()

        progress(0.05, "Validating audio file...")
        logger.info("Starting meeting minutes pipeline for '%s'.", audio_path)

        progress(0.15, "Transcribing audio with Whisper (this can take a while)...")
        transcription_result = self.transcription_service.transcribe(audio_path)

        progress(0.6, "Transcription complete. Generating meeting minutes with Llama 3.2...")
        minutes_markdown = self.summarization_service.generate_minutes(
            transcript=transcription_result.text,
            metadata=metadata,
        )

        progress(1.0, "Meeting minutes generated successfully.")
        logger.info("Pipeline completed successfully.")

        return MeetingMinutesResult(
            markdown=minutes_markdown,
            transcript=transcription_result.text,
            metadata=metadata,
        )
