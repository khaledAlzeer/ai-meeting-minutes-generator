"""Unit tests for src.utils.validators."""

from __future__ import annotations

import pytest

from src.utils.exceptions import (
    AudioTooLargeError,
    EmptyTranscriptError,
    InvalidTranscriptError,
    MissingAudioError,
    UnsupportedAudioFormatError,
)
from src.utils.validators import validate_audio_file, validate_transcript


def test_validate_audio_file_raises_when_missing():
    with pytest.raises(MissingAudioError):
        validate_audio_file(None)

    with pytest.raises(MissingAudioError):
        validate_audio_file("")


def test_validate_audio_file_raises_for_unsupported_extension(tmp_path):
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("not audio")

    with pytest.raises(UnsupportedAudioFormatError):
        validate_audio_file(bad_file)


def test_validate_audio_file_raises_for_too_large_file(tmp_path):
    from dataclasses import replace

    from src.config import settings

    # `Settings` and `LimitConfig` are frozen dataclasses (by design, so
    # configuration cannot be mutated at runtime by application code).
    # For this test only, we bypass immutability with `object.__setattr__`
    # and restore the original value afterwards.
    original_limits = settings.limits
    try:
        object.__setattr__(settings, "limits", replace(original_limits, max_audio_size_mb=0))

        audio_file = tmp_path / "meeting.mp3"
        audio_file.write_bytes(b"fake audio bytes")

        with pytest.raises(AudioTooLargeError):
            validate_audio_file(audio_file)
    finally:
        object.__setattr__(settings, "limits", original_limits)


def test_validate_audio_file_accepts_valid_file(tmp_path):
    audio_file = tmp_path / "meeting.mp3"
    audio_file.write_bytes(b"fake audio bytes")

    result = validate_audio_file(audio_file)
    assert result == audio_file


def test_validate_transcript_raises_when_empty():
    with pytest.raises(EmptyTranscriptError):
        validate_transcript(None)

    with pytest.raises(EmptyTranscriptError):
        validate_transcript("   ")


def test_validate_transcript_raises_when_too_short():
    with pytest.raises(InvalidTranscriptError):
        validate_transcript("hi")


def test_validate_transcript_returns_stripped_text():
    text = "  This is a sufficiently long transcript for testing purposes.  "
    assert validate_transcript(text) == text.strip()
