"""Centralized application configuration.

Every model identifier, generation parameter, filesystem path, and
operational limit used across the application is defined here so that the
rest of the codebase never hard-codes configuration values. Configuration
is sourced from environment variables (optionally loaded from a local
``.env`` file via ``python-dotenv``), with sensible defaults for local
development.

Only ``get_settings()`` / the module-level ``settings`` singleton should be
imported by the rest of the application.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a `.env` file if one exists in the working directory
# or any parent directory. Real environment variables always take
# precedence over values defined in `.env`.
load_dotenv(override=False)

# Root of the repository (two levels up from this file: src/config/settings.py).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean-like environment variable."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int) -> int:
    """Parse an integer environment variable, falling back on failure."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Parse a float environment variable, falling back on failure."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class ModelConfig:
    """Model identifiers and inference-time generation parameters."""

    whisper_model_id: str = os.getenv("WHISPER_MODEL_ID", "openai/whisper-medium.en")
    llm_model_id: str = os.getenv("LLM_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")

    max_new_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_NEW_TOKENS", 2000))
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.3))
    top_p: float = field(default_factory=lambda: _env_float("LLM_TOP_P", 0.9))
    repetition_penalty: float = field(
        default_factory=lambda: _env_float("LLM_REPETITION_PENALTY", 1.1)
    )

    use_4bit_quantization: bool = field(
        default_factory=lambda: _env_bool("USE_4BIT_QUANTIZATION", True)
    )
    whisper_return_timestamps: bool = True
    whisper_chunk_length_s: int = field(
        default_factory=lambda: _env_int("WHISPER_CHUNK_LENGTH_S", 30)
    )


@dataclass(frozen=True)
class PathConfig:
    """Filesystem locations used by the application."""

    project_root: Path = PROJECT_ROOT
    output_dir: Path = PROJECT_ROOT / "outputs"
    assets_dir: Path = PROJECT_ROOT / "assets"
    examples_dir: Path = PROJECT_ROOT / "examples"
    logo_path: Path = PROJECT_ROOT / "assets" / "logo.png"

    def ensure_directories(self) -> None:
        """Create runtime-writable directories if they do not yet exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LimitConfig:
    """Validation limits and constraints."""

    max_audio_size_mb: int = field(default_factory=lambda: _env_int("MAX_AUDIO_SIZE_MB", 200))
    allowed_audio_extensions: tuple[str, ...] = (
        ".mp3",
        ".wav",
        ".m4a",
        ".flac",
        ".ogg",
        ".webm",
        ".mp4",
        ".mpeg",
        ".mpga",
    )
    min_transcript_characters: int = field(
        default_factory=lambda: _env_int("MIN_TRANSCRIPT_CHARACTERS", 10)
    )


@dataclass(frozen=True)
class ServerConfig:
    """Gradio server / Hugging Face Spaces launch configuration."""

    server_name: str = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port: int = field(default_factory=lambda: _env_int("GRADIO_SERVER_PORT", 7860))
    share: bool = field(default_factory=lambda: _env_bool("GRADIO_SHARE", False))
    queue_max_size: int = field(default_factory=lambda: _env_int("GRADIO_QUEUE_MAX_SIZE", 20))
    show_error: bool = True


@dataclass(frozen=True)
class AppMetadata:
    """Static application copy used throughout the UI and docs."""

    title: str = "AI Meeting Minutes Generator"
    tagline: str = "From raw audio to polished meeting minutes in minutes."
    description: str = (
        "Upload or record a meeting, and let on-device AI models transcribe "
        "the conversation and draft structured, professional minutes — "
        "complete with a summary, discussion points, decisions, and action "
        "items with owners."
    )
    author: str = os.getenv("APP_AUTHOR", "Your Name")
    repository_url: str = os.getenv(
        "APP_REPOSITORY_URL", "https://github.com/your-username/ai-meeting-minutes"
    )


@dataclass(frozen=True)
class Settings:
    """Aggregate application settings."""

    model: ModelConfig
    paths: PathConfig
    limits: LimitConfig
    server: ServerConfig
    metadata: AppMetadata
    hf_token: str | None
    log_level: str
    device_preference: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build (and memoize) the application :class:`Settings` singleton.

    Using ``lru_cache`` guarantees the settings object — and therefore the
    environment variables backing it — is only read once per process,
    which keeps behaviour predictable even if ``os.environ`` is mutated
    later at runtime (e.g. in tests).
    """
    return Settings(
        model=ModelConfig(),
        paths=PathConfig(),
        limits=LimitConfig(),
        server=ServerConfig(),
        metadata=AppMetadata(),
        hf_token=os.getenv("HF_TOKEN") or None,
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        device_preference=os.getenv("DEVICE_PREFERENCE", "auto").lower(),
    )


# Convenience module-level singleton for ergonomic imports:
#   from src.config import settings
#   settings.model.llm_model_id
settings = get_settings()
