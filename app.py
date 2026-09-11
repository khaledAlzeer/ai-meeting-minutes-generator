"""Application entry point for the AI Meeting Minutes Generator.

Run locally with:

    uv run app.py

This file is also the entry point Hugging Face Spaces looks for by
default when deploying a Gradio "Space" (a file named ``app.py`` at the
repository root that launches a ``gr.Blocks`` / ``gr.Interface`` demo).
"""

from __future__ import annotations

import sys

from src.config import settings
from src.ui.gradio_app import build_interface
from src.utils.logger import configure_logging, get_logger

configure_logging(level=settings.log_level)
logger = get_logger(__name__)


def main() -> None:
    """Configure the environment and launch the Gradio application."""
    settings.paths.ensure_directories()

    logger.info("Starting %s...", settings.metadata.title)
    logger.info(
        "Whisper model: %s | LLM: %s | 4-bit quantization: %s",
        settings.model.whisper_model_id,
        settings.model.llm_model_id,
        settings.model.use_4bit_quantization,
    )

    if not settings.hf_token:
        logger.warning(
            "HF_TOKEN is not set. Loading gated models such as '%s' will "
            "fail until a valid token is provided via the environment or "
            "a .env file. See .env.example for details.",
            settings.model.llm_model_id,
        )

    demo = build_interface()
    demo.queue(max_size=settings.server.queue_max_size)

    try:
        demo.launch(
            server_name=settings.server.server_name,
            server_port=settings.server.server_port,
            share=settings.server.share,
            show_error=settings.server.show_error,
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive convenience
        logger.info("Shutting down gracefully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
