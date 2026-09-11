"""Prompt templates for meeting-minutes generation.

Prompts are intentionally isolated from the summarization service so they
can be iterated on independently — tweaking tone, structure, or output
format never requires touching model-loading or inference code.
"""

from __future__ import annotations

from src.models.schemas import MeetingMetadata

SYSTEM_PROMPT: str = (
    "You are an expert executive assistant who specializes in producing "
    "clear, professional meeting minutes from raw meeting transcripts. "
    "You write in clean Markdown (no code blocks, no HTML) and always "
    'follow the requested structure exactly, using level-2 Markdown '
    'headings ("## ") for each section. You infer missing details '
    "(attendees, date, location) from the transcript when they are not "
    'explicitly provided, and you clearly write "Not specified" when a '
    "detail genuinely cannot be determined. You are concise, neutral in "
    "tone, and never invent facts, decisions, or action items that are "
    "not supported by the transcript."
)


MINUTES_OUTPUT_STRUCTURE: str = (
    "## Meeting Title\n"
    "## Meeting Date\n"
    "## Meeting Location\n"
    "## Attendees\n"
    "## Executive Summary\n"
    "## Discussion Points\n"
    "## Decisions\n"
    "## Key Takeaways\n"
    "## Action Items\n"
    "## Next Steps\n"
    "## Notes"
)


def build_user_prompt(transcript: str, metadata: MeetingMetadata) -> str:
    """Construct the user-turn prompt sent to the LLM.

    Args:
        transcript: The raw (validated) transcript text.
        metadata: Optional meeting context supplied by the user via the UI.

    Returns:
        A fully formatted prompt string ready to be placed in the
        ``messages`` list passed to the chat template.
    """

    known_context_lines = []

    if metadata.title.strip():
        known_context_lines.append(
            f"- Meeting Title: {metadata.title.strip()}"
        )

    if metadata.date.strip():
        known_context_lines.append(
            f"- Meeting Date: {metadata.date.strip()}"
        )

    if metadata.location.strip():
        known_context_lines.append(
            f"- Meeting Location: {metadata.location.strip()}"
        )

    if metadata.attendees.strip():
        known_context_lines.append(
            f"- Known Attendees: {metadata.attendees.strip()}"
        )

    known_context_block = (
        "\n".join(known_context_lines)
        if known_context_lines
        else (
            "(No metadata was provided by the user; infer everything from "
            "the transcript.)"
        )
    )

    return (
        "Below is a transcript of a meeting. Write professional meeting "
        "minutes in Markdown, following EXACTLY this section structure "
        "and using these level-2 headings, in this order:\n\n"
        f"{MINUTES_OUTPUT_STRUCTURE}\n\n"
        "Guidance for each section:\n\n"
        "- Meeting Title: A short, descriptive title for the meeting.\n"
        '- Meeting Date: The date of the meeting if mentioned or provided, '
        'otherwise "Not specified".\n'
        '- Meeting Location: The location or platform (e.g. "City Hall", '
        '"Zoom call") if known, otherwise "Not specified".\n'
        "- Attendees: A bulleted list of participants mentioned in the "
        "transcript or provided below.\n"
        "- Executive Summary: 2-4 sentences summarizing the purpose and "
        "outcome of the meeting.\n"
        "- Discussion Points: A bulleted list of the main topics discussed.\n"
        '- Decisions: A bulleted list of concrete decisions that were made. '
        'Write "No formal decisions were recorded." if none.\n'
        "- Key Takeaways: A short bulleted list of the most important "
        "insights.\n"
        '- Action Items: A bulleted list formatted as "- [Owner]: Action '
        'description (due: date if known)". Use "Unassigned" as the owner '
        "if none is mentioned.\n"
        "- Next Steps: A short bulleted list of what happens after this "
        "meeting.\n"
        '- Notes: Any other relevant context, caveats about transcript '
        'quality, or open questions. Write "None." if not applicable.\n\n'
        "Known meeting context supplied by the user (use it verbatim where "
        "it does not conflict with the transcript; otherwise trust the "
        "transcript):\n\n"
        f"{known_context_block}\n\n"
        "Transcript:\n\n"
        '"""\n\n'
        f"{transcript}\n\n"
        '"""\n\n'
        "Respond with ONLY the Markdown meeting minutes. Do not wrap the "
        "response in a code block, and do not add any commentary before "
        "or after the minutes."
    )


def build_chat_messages(
    transcript: str,
    metadata: MeetingMetadata,
) -> list[dict[str, str]]:
    """Build the full chat-formatted message list for the LLM's chat template."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(transcript, metadata),
        },
    ]