"""Export utilities for generating downloadable artifacts.

Gradio's ``File`` / ``DownloadButton`` components need a real path on
disk. These helpers write transcripts and meeting minutes to the
application's output directory using clean, predictable filenames, and
provide a simple Markdown-to-plain-text conversion for the ``.txt``
download variant.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _slugify(value: str, max_length: int = 40) -> str:
    """Turn arbitrary text into a filesystem-safe slug."""

    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")

    return (value or "meeting")[:max_length]


def _unique_path(
    directory: Path,
    base_name: str,
    extension: str,
) -> Path:
    """Build a unique output path, avoiding collisions between runs."""

    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    filename = f"{base_name}-{timestamp}-{short_id}{extension}"

    return directory / filename


def markdown_to_plain_text(markdown_text: str) -> str:
    """Strip common Markdown syntax to produce a readable plain-text version.

    This is intentionally lightweight rather than a full Markdown parser:
    it is good enough to produce a clean ``.txt`` download without pulling
    in an extra dependency.
    """

    text = markdown_text

    # Remove code fences but keep their inner content.
    text = re.sub(r"```[a-zA-Z0-9]*\n?", "", text)

    # Headings: "## Title" -> "TITLE"
    text = re.sub(
        r"^\s{0,3}#{1,6}\s*(.+)$",
        lambda m: m.group(1).strip().upper(),
        text,
        flags=re.MULTILINE,
    )

    # Bold / italic markers.
    text = re.sub(r"(\*\*\*|___)(.+?)\1", r"\2", text)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.+?)\1", r"\2", text)

    # Bullet markers "- item" / "* item" -> "  • item"
    text = re.sub(
        r"^\s*[-*+]\s+",
        "  • ",
        text,
        flags=re.MULTILINE,
    )

    # Numbered list markers stay as-is; just strip extra markdown link syntax.
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Collapse 3+ blank lines down to 2.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip() + "\n"


def export_transcript(
    transcript: str,
    meeting_title: str = "meeting",
) -> Path:
    """Write the raw transcript to a ``.txt`` file and return its path."""

    slug = _slugify(meeting_title)
    path = _unique_path(
        settings.paths.output_dir,
        f"{slug}-transcript",
        ".txt",
    )

    path.write_text(transcript, encoding="utf-8")
    logger.info("Transcript exported to '%s'.", path)

    return path


def export_minutes_markdown(
    markdown_text: str,
    meeting_title: str = "meeting",
) -> Path:
    """Write meeting minutes to a ``.md`` file and return its path."""

    slug = _slugify(meeting_title)
    path = _unique_path(
        settings.paths.output_dir,
        f"{slug}-minutes",
        ".md",
    )

    path.write_text(markdown_text, encoding="utf-8")
    logger.info("Meeting minutes (Markdown) exported to '%s'.", path)

    return path


def export_minutes_plain_text(
    markdown_text: str,
    meeting_title: str = "meeting",
) -> Path:
    """Write meeting minutes as plain text (``.txt``) and return its path."""

    slug = _slugify(meeting_title)
    plain_text = markdown_to_plain_text(markdown_text)
    path = _unique_path(
        settings.paths.output_dir,
        f"{slug}-minutes",
        ".txt",
    )

    path.write_text(plain_text, encoding="utf-8")
    logger.info("Meeting minutes (plain text) exported to '%s'.", path)

    return path