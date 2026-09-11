"""Modal deployment entry point for the AI Meeting Minutes Generator."""

from pathlib import Path

import modal


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# Modal image
# ============================================================

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi",
        "torch==2.9.0",
        "transformers==4.57.6",
        "accelerate>=0.30.0",
        "bitsandbytes>=0.43.0",
        "gradio==5.50.0",
        "huggingface_hub>=0.24.0",
        "python-dotenv>=1.0.1",
        "numpy>=1.26.0",
    )
    .add_local_dir(
        PROJECT_ROOT / "src",
        remote_path="/app/src",
    )
    .add_local_dir(
        PROJECT_ROOT / "assets",
        remote_path="/app/assets",
    )
    .add_local_dir(
        PROJECT_ROOT / "examples",
        remote_path="/app/examples",
    )
)


# ============================================================
# Modal application
# ============================================================

app = modal.App("ai-meeting-minutes")


# ============================================================
# Gradio web application
# ============================================================

@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 60,
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def gradio_app():
    """Create and return the Gradio application mounted on FastAPI."""

    import sys

    import gradio as gr
    from fastapi import FastAPI

    # Make the deployed project source importable.
    sys.path.insert(0, "/app")

    from src.config import settings
    from src.ui.gradio_app import build_interface
    from src.utils.logger import configure_logging

    # Prepare required directories.
    settings.paths.ensure_directories()

    # Configure application logging.
    configure_logging(level=settings.log_level)

    # Build the existing production Gradio interface.
    demo = build_interface()

    # Enable Gradio's queue for long-running AI operations.
    demo.queue(max_size=settings.server.queue_max_size)

    # Mount Gradio inside a FastAPI ASGI application.
    return gr.mount_gradio_app(
        app=FastAPI(),
        blocks=demo,
        path="/",
    )