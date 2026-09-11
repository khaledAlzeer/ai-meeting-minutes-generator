"""Unit tests for src.prompts.templates."""

from __future__ import annotations

from src.models.schemas import MeetingMetadata
from src.prompts.templates import build_chat_messages, build_user_prompt


def test_build_user_prompt_includes_transcript():
    transcript = "Alice: Let's discuss the Q3 roadmap."
    prompt = build_user_prompt(transcript, MeetingMetadata())
    assert transcript in prompt


def test_build_user_prompt_includes_known_metadata():
    metadata = MeetingMetadata(
        title="Q3 Roadmap Sync",
        date="2026-09-10",
        location="Zoom",
        attendees="Alice, Bob",
    )
    prompt = build_user_prompt("Some transcript text.", metadata)

    assert "Q3 Roadmap Sync" in prompt
    assert "2026-09-10" in prompt
    assert "Zoom" in prompt
    assert "Alice, Bob" in prompt


def test_build_user_prompt_handles_missing_metadata_gracefully():
    prompt = build_user_prompt("Some transcript text.", MeetingMetadata())
    assert "infer everything from the transcript" in prompt


def test_build_chat_messages_structure():
    messages = build_chat_messages("Transcript text.", MeetingMetadata())
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Transcript text." in messages[1]["content"]
