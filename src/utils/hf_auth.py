"""Hugging Face Hub authentication helpers.

Models such as ``meta-llama/Llama-3.2-3B-Instruct`` are gated and require
an authenticated Hugging Face account with accepted usage terms. This
module centralizes the login logic and turns opaque failures into clear,
actionable error messages.
"""

from __future__ import annotations

from src.config import settings
from src.utils.exceptions import HuggingFaceAuthenticationError, MissingHFTokenError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_LOGIN_ATTEMPTED = False
_LOGIN_SUCCEEDED = False


def ensure_hf_login(required: bool = True) -> bool:
    """Authenticate with the Hugging Face Hub using the ``HF_TOKEN`` env var.

    The login is only performed once per process; subsequent calls are
    no-ops that return the cached result.

    Args:
        required: If ``True``, raise :class:`MissingHFTokenError` when no
            token is configured. If ``False``, missing credentials are
            logged as a warning and the function returns ``False``,
            allowing callers to proceed with public, non-gated models.

    Returns:
        ``True`` if authentication succeeded, ``False`` otherwise.

    Raises:
        MissingHFTokenError: If ``required`` is ``True`` and no token is set.
        HuggingFaceAuthenticationError: If the token is set but login fails
            (e.g. it is invalid or revoked).
    """
    global _LOGIN_ATTEMPTED, _LOGIN_SUCCEEDED

    if _LOGIN_ATTEMPTED:
        return _LOGIN_SUCCEEDED

    _LOGIN_ATTEMPTED = True
    token = settings.hf_token

    if not token:
        message = (
            "No Hugging Face access token was found. Set the HF_TOKEN "
            "environment variable (e.g. in a local .env file, or as a "
            "'Repository secret' on Hugging Face Spaces) with a token that "
            "has been granted access to gated models such as "
            f"'{settings.model.llm_model_id}'. Create a token at "
            "https://huggingface.co/settings/tokens and request model "
            f"access at https://huggingface.co/{settings.model.llm_model_id}."
        )
        if required:
            logger.error(message)
            raise MissingHFTokenError(message)
        logger.warning(message)
        return False

    try:
        # Imported lazily so environments that only need transcription
        # (no gated models) are not forced to have huggingface_hub's full
        # dependency chain resolved at import time.
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        logger.info("Successfully authenticated with the Hugging Face Hub.")
        _LOGIN_SUCCEEDED = True
        return True
    except Exception as exc:  # noqa: BLE001 - we deliberately re-wrap any failure
        message = (
            "Failed to authenticate with the Hugging Face Hub using the "
            "configured HF_TOKEN. Please verify the token is valid and has "
            "not expired or been revoked."
        )
        logger.exception(message)
        raise HuggingFaceAuthenticationError(message) from exc
