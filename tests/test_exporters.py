"""Unit tests for src.utils.exporters."""

from __future__ import annotations

import pytest

from src.config import settings
from src.utils.exporters import (
    export_minutes_markdown,
    export_minutes_plain_text,
    export_transcript,
    markdown_to_plain_text,
)


@pytest.fixture
def redirected_output_dir(tmp_path):
    """Temporarily redirect the output directory used by the exporters.

    `PathConfig` is a frozen dataclass by design, so we bypass immutability
    with `object.__setattr__` for the duration of the test only, restoring
    the original value afterwards.
    """
    original_paths = settings.paths
    try:
        object.__setattr__(settings, "paths", _with_output_dir(original_paths, tmp_path))
        yield tmp_path
    finally:
        object.__setattr__(settings, "paths", original_paths)


def _with_output_dir(paths, new_output_dir):
    from dataclasses import replace

    return replace(paths, output_dir=new_output_dir)


def test_markdown_to_plain_text_strips_headings_and_emphasis():
    markdown_text = "## Executive Summary\n\nThis was a **great** meeting."
    plain_text = markdown_to_plain_text(markdown_text)

    assert "##" not in plain_text
    assert "**" not in plain_text
    assert "EXECUTIVE SUMMARY" in plain_text
    assert "great" in plain_text


def test_markdown_to_plain_text_converts_bullets():
    markdown_text = "- First item\n- Second item"
    plain_text = markdown_to_plain_text(markdown_text)

    assert "\u2022 First item" in plain_text
    assert "\u2022 Second item" in plain_text


def test_export_transcript_writes_file(redirected_output_dir):
    path = export_transcript("Hello world transcript.", "Team Sync")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == "Hello world transcript."
    assert path.suffix == ".txt"
    assert path.parent == redirected_output_dir


def test_export_minutes_markdown_writes_file(redirected_output_dir):
    path = export_minutes_markdown("## Executive Summary\nAll good.", "Team Sync")
    assert path.exists()
    assert path.suffix == ".md"
    assert path.parent == redirected_output_dir


def test_export_minutes_plain_text_writes_file(redirected_output_dir):
    path = export_minutes_plain_text("## Executive Summary\nAll good.", "Team Sync")
    assert path.exists()
    assert path.suffix == ".txt"
    assert path.parent == redirected_output_dir
    content = path.read_text(encoding="utf-8")
    assert "EXECUTIVE SUMMARY" in content
