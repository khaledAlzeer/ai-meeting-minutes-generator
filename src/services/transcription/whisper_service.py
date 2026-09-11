"""Speech-to-text transcription service backed by OpenAI's Whisper model.

The service is encapsulated behind a small, stable interface
(:meth:`WhisperTranscriptionService.transcribe`) so the underlying model
can be swapped out in the future (e.g. for ``faster-whisper`` or a hosted
API) without touching any calling code.

Uploaded audio/video files are normalized through FFmpeg into a temporary
16 kHz mono WAV file before being passed to Whisper. This allows the service
to accept formats such as MP4, M4A, WebM, MP3, and other FFmpeg-supported
media formats consistently.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from src.config import settings
from src.models.schemas import TranscriptionResult
from src.utils.exceptions import (
    ModelDownloadError,
    ModelLoadError,
    NetworkError,
    OutOfMemoryError,
    TranscriptionError,
)
from src.utils.logger import get_logger
from src.utils.validators import validate_audio_file, validate_transcript

logger = get_logger(__name__)


class WhisperTranscriptionService:
    """Transcribes audio/video files to text using a Hugging Face Whisper pipeline.

    The underlying ``transformers`` pipeline is loaded lazily on first use
    and cached at the class level so that repeated transcriptions within
    the same process reuse the already-loaded model.

    Before transcription, incoming media is normalized with FFmpeg to a
    temporary 16 kHz mono PCM WAV file. This provides Whisper with a
    consistent input format regardless of the original uploaded format.
    """

    _pipeline = None  # Shared across all instances within the process.
    _loaded_model_id: str | None = None

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or settings.model.whisper_model_id

    # ------------------------------------------------------------------
    # Media preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_audio_to_wav(audio_path: Path) -> Path:
        """Convert an input media file into a temporary 16 kHz mono WAV.

        FFmpeg is used because Transformers/SoundFile may not be able to
        decode container formats such as MP4 directly.

        Args:
            audio_path: Validated source audio/video path.

        Returns:
            Path to the temporary WAV file.

        Raises:
            TranscriptionError: If FFmpeg is unavailable or conversion fails.

        The caller is responsible for deleting the returned temporary file.
        """

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                prefix="whisper_",
                delete=False,
            ) as temp_file:
                output_path = Path(temp_file.name)
        except OSError as exc:
            raise TranscriptionError(
                "Could not create a temporary audio file for transcription."
            ) from exc

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]

        logger.info(
            "Normalizing media file '%s' to 16 kHz mono WAV...",
            audio_path,
        )

        try:
            completed_process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            output_path.unlink(missing_ok=True)
            raise TranscriptionError(
                "FFmpeg is required to process uploaded audio/video files "
                "but was not found on the system PATH. Please install "
                "FFmpeg and make sure the 'ffmpeg' command is available."
            ) from exc
        except OSError as exc:
            output_path.unlink(missing_ok=True)
            raise TranscriptionError(
                f"Could not start FFmpeg while processing '{audio_path}'."
            ) from exc

        if completed_process.returncode != 0:
            output_path.unlink(missing_ok=True)
            error_details = completed_process.stderr.strip()

            logger.error(
                "FFmpeg conversion failed for '%s': %s",
                audio_path,
                error_details,
            )

            raise TranscriptionError(
                "FFmpeg could not convert the uploaded media file into "
                "an audio format suitable for transcription. "
                "Please try another recording."
            )

        if not output_path.exists() or output_path.stat().st_size == 0:
            output_path.unlink(missing_ok=True)
            raise TranscriptionError(
                "FFmpeg completed but produced an empty audio file. "
                "Please try another recording."
            )

        logger.info(
            "Media normalization complete: '%s' -> '%s'.",
            audio_path,
            output_path,
        )

        return output_path

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _resolve_device_and_dtype(self):
        """Determine the best available device and matching dtype.

        Returns:
            A tuple of ``(device, torch_dtype)``.

        Falls back gracefully to CPU with float32 when no GPU is available.
        """

        import torch

        if torch.cuda.is_available():
            logger.info(
                "CUDA GPU detected. Running Whisper on GPU with float16."
            )
            return "cuda:0", torch.float16

        logger.warning(
            "No CUDA GPU detected. Falling back to CPU for transcription. "
            "This will be significantly slower than GPU inference."
        )

        return "cpu", torch.float32

    def _load_pipeline(self):
        """Lazily instantiate and cache the ASR pipeline."""

        if (
            WhisperTranscriptionService._pipeline is not None
            and WhisperTranscriptionService._loaded_model_id == self.model_id
        ):
            return WhisperTranscriptionService._pipeline

        logger.info(
            "Loading Whisper model '%s'...",
            self.model_id,
        )

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ModelLoadError(
                "The 'transformers' package is required for transcription "
                "but is not installed. Run 'uv sync' to install project "
                "dependencies."
            ) from exc

        device, torch_dtype = self._resolve_device_and_dtype()

        try:
            # Do not use chunk_length_s here.
            # The previous experimental chunking path was extremely slow
            # for long-form audio.
            asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                dtype=torch_dtype,
                device=device,
                return_timestamps=settings.model.whisper_return_timestamps,
            )
        except OSError as exc:
            raise ModelDownloadError(
                f"Failed to download or load the Whisper model "
                f"'{self.model_id}'. Check your internet connection and "
                "that the model name is correct."
            ) from exc
        except ConnectionError as exc:
            raise NetworkError(
                "A network error occurred while downloading the Whisper "
                "model. Please check your internet connection and try "
                "again."
            ) from exc

        WhisperTranscriptionService._pipeline = asr_pipeline
        WhisperTranscriptionService._loaded_model_id = self.model_id

        logger.info(
            "Whisper model '%s' loaded successfully on %s.",
            self.model_id,
            device,
        )

        return asr_pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: str | Path,
    ) -> TranscriptionResult:
        """Transcribe an audio/video file to text.

        Args:
            audio_path: Path to the uploaded or recorded media file.

        Returns:
            A :class:`TranscriptionResult` containing the transcript text.

        Raises:
            MissingAudioError: If ``audio_path`` is empty/None.
            InvalidAudioFileError: If the file is missing, empty, or corrupt.
            UnsupportedAudioFormatError: If the file extension is unsupported.
            AudioTooLargeError: If the file exceeds the configured size limit.
            ModelDownloadError / NetworkError / ModelLoadError: On model
                loading failures.
            OutOfMemoryError: If the GPU runs out of memory during inference.
            TranscriptionError: For other transcription failures.
            EmptyTranscriptError: If transcription yields no usable text.
        """

        # --------------------------------------------------------------
        # Step 1: Validate the original uploaded file
        # --------------------------------------------------------------

        validated_path = validate_audio_file(audio_path)

        # --------------------------------------------------------------
        # Step 2: Convert the input media to a Whisper-friendly WAV
        # --------------------------------------------------------------

        temporary_wav_path: Path | None = None

        try:
            temporary_wav_path = self._normalize_audio_to_wav(
                validated_path
            )

            # ----------------------------------------------------------
            # Step 3: Load the cached Whisper model
            # ----------------------------------------------------------

            asr_pipeline = self._load_pipeline()

            logger.info(
                "Transcribing normalized audio file '%s'...",
                temporary_wav_path,
            )

            # ----------------------------------------------------------
            # Step 4: Run Whisper
            # ----------------------------------------------------------

            try:
                result = asr_pipeline(str(temporary_wav_path))
            except Exception as exc:  # noqa: BLE001 - broad on purpose
                self._reraise_as_domain_error(exc)
                raise  # pragma: no cover

            # ----------------------------------------------------------
            # Step 5: Extract transcript text
            # ----------------------------------------------------------

            text = (
                (result.get("text") or "").strip()
                if isinstance(result, dict)
                else str(result).strip()
            )

            validate_transcript(text)

            chunks = (
                result.get("chunks", [])
                if isinstance(result, dict)
                else []
            )

            logger.info(
                "Transcription complete: %d characters, %d chunk(s).",
                len(text),
                len(chunks),
            )

            return TranscriptionResult(
                text=text,
                chunks=chunks,
            )

        finally:
            # ----------------------------------------------------------
            # Step 6: Always remove the temporary WAV
            # ----------------------------------------------------------

            if temporary_wav_path is not None:
                try:
                    temporary_wav_path.unlink(missing_ok=True)
                    logger.debug(
                        "Temporary transcription file removed: '%s'.",
                        temporary_wav_path,
                    )
                except OSError as exc:
                    logger.warning(
                        "Could not remove temporary transcription file "
                        "'%s': %s",
                        temporary_wav_path,
                        exc,
                    )

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @staticmethod
    def _reraise_as_domain_error(exc: Exception) -> None:
        """Translate low-level exceptions into domain-specific ones."""

        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            raise OutOfMemoryError(
                "The GPU ran out of memory while transcribing this audio "
                "file. Try a shorter audio clip, or run on a machine with "
                "more GPU memory."
            ) from exc

        if isinstance(exc, (ConnectionError, TimeoutError)):
            raise NetworkError(
                "A network error occurred during transcription. Please "
                "check your internet connection and try again."
            ) from exc

        raise TranscriptionError(
            f"Transcription failed: {exc}"
        ) from exc