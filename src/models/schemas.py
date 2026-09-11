"""Shared data structures used across services, prompts, and the UI.

Keeping these as plain, typed dataclasses (rather than passing raw dicts
or tuples between layers) makes the pipeline self-documenting and easy to
extend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_type


@dataclass
class MeetingMetadata:
    """User-supplied context about the meeting, used to steer the prompt.

    All fields are optional free text — the LLM is instructed to infer
    any missing details directly from the transcript where possible.
    """

    title: str = ""
    date: str = ""
    location: str = ""
    attendees: str = ""

    def has_any_context(self) -> bool:
        """Return ``True`` if the user supplied at least one metadata field."""

        return any(
            [
                self.title.strip(),
                self.date.strip(),
                self.location.strip(),
                self.attendees.strip(),
            ]
        )

    def display_title(self) -> str:
        """A safe, non-empty title suitable for filenames and headers."""

        return self.title.strip() or "Untitled Meeting"


@dataclass
class TranscriptionResult:
    """The output of the transcription service."""

    text: str
    language: str = "en"
    duration_seconds: float | None = None
    chunks: list[dict] = field(default_factory=list)

    @property
    def character_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class MeetingMinutesResult:
    """The final output of the end-to-end pipeline."""

    markdown: str
    transcript: str
    metadata: MeetingMetadata
    generated_on: date_type = field(default_factory=date_type.today)

    @property
    def word_count(self) -> int:
        return len(self.markdown.split())