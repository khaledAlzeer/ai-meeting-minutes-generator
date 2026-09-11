"""Domain-specific exceptions.

Using a dedicated exception hierarchy (rather than bare ``Exception`` or
built-in exception types) lets the UI layer present clear, actionable
error messages to end users while still allowing calling code to catch
broad categories of failure when useful.
"""

from __future__ import annotations


class MeetingMinutesError(Exception):
    """Base class for all application-specific errors."""


# --------------------------------------------------------------------------- #
# Input validation errors
# --------------------------------------------------------------------------- #
class MissingAudioError(MeetingMinutesError):
    """Raised when no audio input was provided by the user."""


class UnsupportedAudioFormatError(MeetingMinutesError):
    """Raised when the uploaded audio file extension is not supported."""


class AudioTooLargeError(MeetingMinutesError):
    """Raised when the uploaded audio file exceeds the configured size limit."""


class InvalidAudioFileError(MeetingMinutesError):
    """Raised when the audio file exists but appears to be corrupt or empty."""


# --------------------------------------------------------------------------- #
# Authentication / environment errors
# --------------------------------------------------------------------------- #
class MissingHFTokenError(MeetingMinutesError):
    """Raised when a required Hugging Face access token is not configured."""


class HuggingFaceAuthenticationError(MeetingMinutesError):
    """Raised when authentication with the Hugging Face Hub fails."""


class GPUNotAvailableError(MeetingMinutesError):
    """Raised (as a soft warning condition) when no CUDA GPU is detected."""


# --------------------------------------------------------------------------- #
# Model lifecycle errors
# --------------------------------------------------------------------------- #
class ModelDownloadError(MeetingMinutesError):
    """Raised when a model or tokenizer fails to download or load."""


class ModelLoadError(MeetingMinutesError):
    """Raised when a model fails to initialize after being downloaded."""


class OutOfMemoryError(MeetingMinutesError):
    """Raised when the accelerator (GPU) runs out of memory during inference."""


class NetworkError(MeetingMinutesError):
    """Raised when a network operation (e.g. model download) fails."""


# --------------------------------------------------------------------------- #
# Pipeline errors
# --------------------------------------------------------------------------- #
class TranscriptionError(MeetingMinutesError):
    """Raised when audio transcription fails for a reason other than the above."""


class EmptyTranscriptError(MeetingMinutesError):
    """Raised when transcription succeeds but produces no usable text."""


class SummarizationError(MeetingMinutesError):
    """Raised when meeting-minutes generation fails."""


class InvalidTranscriptError(MeetingMinutesError):
    """Raised when a transcript is malformed or too short to summarize."""
