"""Meeting-minutes generation service backed by a local Llama 3.2 model.

All tokenizer and model management (loading, quantization, device
placement, generation, and decoding) is hidden behind
:class:`LlamaSummarizationService`. The model is loaded lazily on first
use and cached at the class level so it is never loaded more than once
per process, regardless of how many requests the Gradio app serves.
"""

from __future__ import annotations

from src.config import settings
from src.models.schemas import MeetingMetadata
from src.prompts.templates import build_chat_messages
from src.utils.exceptions import (
    ModelDownloadError,
    ModelLoadError,
    NetworkError,
    OutOfMemoryError,
    SummarizationError,
)
from src.utils.hf_auth import ensure_hf_login
from src.utils.logger import get_logger
from src.utils.validators import validate_transcript

logger = get_logger(__name__)

# Marker inserted by Llama 3.2's chat template ahead of the assistant's
# reply; used to strip the echoed prompt out of the decoded output.
_ASSISTANT_HEADER_MARKERS = (
    "<|start_header_id|>assistant<|end_header_id|>",
)
_END_OF_TEXT_TOKENS = ("<|eot_id|>", "<|end_of_text|>")


class LlamaSummarizationService:
    """Generates structured meeting minutes from a transcript using Llama 3.2."""

    _tokenizer = None  # Shared across instances.
    _model = None  # Shared across instances.
    _loaded_model_id: str | None = None

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or settings.model.llm_model_id

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #
    def _load_model(self):
        """Lazily load and cache the tokenizer and model."""
        if (
            LlamaSummarizationService._model is not None
            and LlamaSummarizationService._loaded_model_id == self.model_id
        ):
            return LlamaSummarizationService._tokenizer, LlamaSummarizationService._model

        # Gated models require an authenticated Hugging Face session.
        ensure_hf_login(required=True)

        logger.info("Loading language model '%s'...", self.model_id)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ModelLoadError(
                "Required packages ('transformers', 'torch', "
                "'bitsandbytes') are not installed. Run 'uv sync' to "
                "install project dependencies."
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        except OSError as exc:
            raise ModelDownloadError(
                f"Failed to download or load the tokenizer for "
                f"'{self.model_id}'. Verify your HF_TOKEN has been granted "
                "access to this gated model."
            ) from exc

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict = {"device_map": "auto"}

        gpu_available = torch.cuda.is_available()
        if gpu_available and settings.model.use_4bit_quantization:
            logger.info("CUDA GPU detected. Loading model in 4-bit quantized mode.")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        elif gpu_available:
            logger.info("CUDA GPU detected. Loading model in bfloat16 (quantization disabled).")
            model_kwargs["dtype"] = torch.bfloat16
        else:
            logger.warning(
                "No CUDA GPU detected. Loading the language model on CPU. "
                "4-bit quantization (bitsandbytes) requires a GPU and has "
                "been disabled automatically. Generation will be slow."
            )
            model_kwargs["dtype"] = torch.float32

        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        except OSError as exc:
            raise ModelDownloadError(
                f"Failed to download or load the model weights for "
                f"'{self.model_id}'. Verify your HF_TOKEN has been granted "
                "access to this gated model, and that you have accepted "
                "its license on the Hugging Face Hub."
            ) from exc
        except ConnectionError as exc:
            raise NetworkError(
                "A network error occurred while downloading the language "
                "model. Please check your internet connection and try "
                "again."
            ) from exc

        LlamaSummarizationService._tokenizer = tokenizer
        LlamaSummarizationService._model = model
        LlamaSummarizationService._loaded_model_id = self.model_id
        logger.info("Language model '%s' loaded successfully.", self.model_id)

        return tokenizer, model

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate_minutes(self, transcript: str, metadata: MeetingMetadata) -> str:
        """Generate Markdown meeting minutes from a transcript.

        Args:
            transcript: The validated transcript text.
            metadata: Optional user-supplied meeting context.

        Returns:
            A Markdown string containing the structured meeting minutes.

        Raises:
            EmptyTranscriptError / InvalidTranscriptError: If the
                transcript fails validation.
            MissingHFTokenError / HuggingFaceAuthenticationError: On
                authentication failures.
            ModelDownloadError / NetworkError / ModelLoadError: On model
                loading failures.
            OutOfMemoryError: If the GPU runs out of memory during generation.
            SummarizationError: For any other generation failure.
        """
        transcript = validate_transcript(transcript)
        tokenizer, model = self._load_model()

        import torch

        messages = build_chat_messages(transcript, metadata)

        logger.info("Generating meeting minutes with '%s'...", self.model_id)
        try:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=settings.model.max_new_tokens,
                    do_sample=True,
                    temperature=settings.model.temperature,
                    top_p=settings.model.top_p,
                    repetition_penalty=settings.model.repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
            raise OutOfMemoryError(
                "The GPU ran out of memory while generating meeting "
                "minutes. Try a shorter transcript, reduce "
                "LLM_MAX_NEW_TOKENS, or run on a machine with more GPU "
                "memory."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise SummarizationError(f"Meeting minutes generation failed: {exc}") from exc

        generated_tokens = output_ids[0][input_ids.shape[-1] :]
        raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        markdown = self._clean_output(raw_output)
        if not markdown:
            raise SummarizationError(
                "The language model returned an empty response. Please "
                "try again."
            )

        logger.info("Meeting minutes generated (%d characters).", len(markdown))
        return markdown

    @staticmethod
    def _clean_output(raw_output: str) -> str:
        """Strip any residual special tokens or code-fence wrapping."""
        text = raw_output.strip()

        for marker in _ASSISTANT_HEADER_MARKERS:
            if marker in text:
                text = text.split(marker, 1)[-1]

        for token in _END_OF_TEXT_TOKENS:
            text = text.replace(token, "")

        text = text.strip()

        # Some models occasionally wrap the whole answer in a markdown
        # code fence despite instructions not to; unwrap it if present.
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]  # drop opening fence (with optional language tag)
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]  # drop closing fence
            text = "\n".join(lines).strip()

        return text
