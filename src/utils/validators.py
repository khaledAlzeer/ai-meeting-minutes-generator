"""Input validation utilities.

Validation is performed as early as possible in the pipeline so that
failures produce clear, user-facing error messages instead of cryptic
stack traces from deep inside third-party libraries.
"""

from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.utils.exceptions import (
    AudioTooLargeError,
    EmptyTranscriptError,
    InvalidAudioFileError,
    InvalidTranscriptError,
    MissingAudioError,
    UnsupportedAudioFormatError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_audio_file(audio_path: str | Path | None) -> Path:
    """Validate that an uploaded/recorded audio file is usable.

    Args:
        audio_path: Path to the audio file, as provided by the Gradio
            ``Audio`` component (``type="filepath"``).

    Returns:
        The validated :class:`~pathlib.Path`.

    Raises:
        MissingAudioError: If no path was provided at all.
        InvalidAudioFileError: If the path does not point to a real, non-empty file.
        UnsupportedAudioFormatError: If the file extension is not supported.
        AudioTooLargeError: If the file exceeds the configured size limit.
    """
    if not audio_path:
        raise MissingAudioError(
            "No audio was provided. Please upload an audio file or record "
            "one using the microphone before generating minutes."
        )

    path = Path(audio_path)

    if not path.exists() or not path.is_file():
        raise InvalidAudioFileError(
            f"The audio file could not be found on disk: '{path}'. Please "
            "try uploading it again."
        )

    if path.stat().st_size == 0:
        raise InvalidAudioFileError(
            "The uploaded audio file is empty (0 bytes). Please choose a "
            "different file or re-record your audio."
        )

    extension = path.suffix.lower()
    if extension not in settings.limits.allowed_audio_extensions:
        allowed = ", ".join(settings.limits.allowed_audio_extensions)
        raise UnsupportedAudioFormatError(
            f"Unsupported audio format '{extension}'. Supported formats "
            f"are: {allowed}."
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.limits.max_audio_size_mb:
        raise AudioTooLargeError(
            f"The audio file is {size_mb:.1f} MB, which exceeds the "
            f"{settings.limits.max_audio_size_mb} MB limit. Please trim the "
            "recording or increase MAX_AUDIO_SIZE_MB."
        )

    logger.debug("Audio file '%s' (%.2f MB) passed validation.", path, size_mb)
    return path


def validate_transcript(transcript: str | None) -> str:
    """Validate that a transcript is non-empty and long enough to summarize.

    Args:
        transcript: The raw transcript text produced by the transcription service.

    Returns:
        The stripped, validated transcript string.

    Raises:
        EmptyTranscriptError: If the transcript is ``None`` or blank.
        InvalidTranscriptError: If the transcript is shorter than the
            configured minimum length.
    """
    if transcript is None or not transcript.strip():
        raise EmptyTranscriptError(
            "Transcription produced no text. This can happen with silent, "
            "corrupted, or non-speech audio. Please try a different "
            "recording."
        )

    cleaned = transcript.strip()
    if len(cleaned) < settings.limits.min_transcript_characters:
        raise InvalidTranscriptError(
            "The transcript is too short to generate meaningful meeting "
            f"minutes (minimum {settings.limits.min_transcript_characters} "
            "characters). Please provide a longer recording."
        )

    return cleaned
