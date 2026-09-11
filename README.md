---
title: AI Meeting Minutes Generator
emoji: 🗒️
colorFrom: indigo
colorTo: teal
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
---

# 🗒️ AI Meeting Minutes Generator

Transcribe meeting audio or video with **OpenAI Whisper** and turn it into structured, professional meeting minutes with **Meta Llama 3.2 3B Instruct**.

The project can run locally on a CUDA-enabled machine or as a cloud-hosted Gradio application on **Modal** using a GPU-backed runtime.

> Upload a recording or video, use your microphone, and get back a polished Markdown meeting-minutes document with a summary, discussion points, decisions, key takeaways, action items with owners, and next steps — ready to copy, share, or download.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation with UV](#installation-with-uv)
  - [Environment Variables](#environment-variables)
  - [Hugging Face Token Setup](#hugging-face-token-setup)
  - [Running Locally](#running-locally)
  - [Using Cursor / VS Code](#using-cursor--vs-code)
- [Usage](#usage)
- [Example Output](#example-output)
- [Testing](#testing)
- [Deployment on Modal](#deployment-on-modal)
  - [Install and Authenticate Modal](#install-and-authenticate-modal)
  - [Create the Hugging Face Secret](#create-the-hugging-face-secret)
  - [Test with Modal Serve](#test-with-modal-serve)
  - [Deploy to Production](#deploy-to-production)
  - [Updating the Deployment](#updating-the-deployment)
- [Configuration Reference](#configuration-reference)
- [Error Handling](#error-handling)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Repository](#repository)

---

## Features

- 🎙️ **Flexible media input** — upload audio/video files or record directly with your microphone, supporting `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.webm`, `.mp4`, `.mpeg`, `.mpga`, and more.

- 📝 **Local, open-source transcription** via `openai/whisper-medium.en`, automatically using your GPU when available and falling back to CPU otherwise.

- 🤖 **Structured meeting-minutes generation** via `meta-llama/Llama-3.2-3B-Instruct`, with 4-bit quantization on CUDA GPUs for efficient memory usage.

- 🧩 **Optional meeting context** — provide the meeting title, date, location, and attendees to guide generation.

- 📋 **Rich Markdown output** covering:
  - Meeting title
  - Meeting date
  - Meeting location
  - Attendees
  - Executive summary
  - Discussion points
  - Decisions
  - Key takeaways
  - Action items with owners
  - Next steps
  - Notes

- 📥 **Multiple export formats** — download the transcript and meeting minutes as `.md` or `.txt` files.

- 📊 **Live progress feedback** with status messages and a progress bar while transcription, model loading, and generation are running.

- 🛡️ **Comprehensive error handling** for missing files, unsupported formats, oversized files, missing GPU, missing Hugging Face tokens, model download failures, network issues, out-of-memory errors, and invalid transcripts.

- 🪵 **Structured logging** throughout the application with no stray `print()` calls.

- 🎨 **Polished Gradio UI** with a custom theme, tabs, accordions, metadata inputs, download controls, and an About section.

- 🧪 **Automated test suite** for validation, prompt building, and export functionality using `pytest`.

- ☁️ **Modal deployment ready** — the project includes a dedicated `modal_app.py` deployment entry point for running the Gradio application on a cloud GPU.

---

## Architecture

The project follows a layered, service-oriented architecture that separates the UI, business logic, model services, configuration, and utilities.

```text
┌─────────────────────────────────────────────────────────────────┐
│                         app.py                                  │
│       Configures logging → builds UI → launches server          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                │ Local execution
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                    src/ui/gradio_app.py                         │
│          Presentation layer: Gradio UI + event wiring           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                src/services/minutes_generator.py               │
│             MeetingMinutesPipeline — orchestrator               │
└──────────────┬───────────────────────────────┬──────────────────┘
               │                               │
┌──────────────▼──────────────┐   ┌────────────▼─────────────────┐
│ services/transcription/     │   │ services/summarization/      │
│ whisper_service.py          │   │ llm_service.py               │
│ WhisperTranscriptionService │   │ LlamaSummarizationService    │
└──────────────┬──────────────┘   └────────────┬─────────────────┘
               │                               │
               │                    ┌──────────▼───────────┐
               │                    │ src/prompts/         │
               │                    │ templates.py         │
               │                    └──────────────────────┘
               │
┌──────────────▼─────────────────────────────────────────────────┐
│                    src/utils/ + src/config/                    │
│ validators · exporters · logger · hf_auth · exceptions          │
│ settings                                                        │
└─────────────────────────────────────────────────────────────────┘


                     Cloud deployment
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       modal_app.py                              │
│       Modal deployment adapter + GPU-backed Gradio server       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
                     src/ui/gradio_app.py
                                │
                                ▼
                     MeetingMinutesPipeline
```

### Key Design Decisions

- **Lazy, cached model loading**  
  Whisper and Llama models are loaded only when first needed and cached for reuse during the application lifetime.

- **Prompts are data, not code**  
  `src/prompts/templates.py` contains prompt templates and formatting logic without model-loading responsibilities.

- **Domain exception hierarchy**  
  Expected application failures are represented through a centralized `MeetingMinutesError` hierarchy and converted into user-friendly UI messages.

- **Separated UI and business logic**  
  `src/ui/gradio_app.py` is responsible for the presentation layer and delegates the actual workflow to `MeetingMinutesPipeline`.

- **GPU-aware inference**  
  The application automatically detects CUDA and uses GPU acceleration when available. Llama can use 4-bit quantization on compatible CUDA systems.

- **Audio/video normalization**  
  Uploaded video files such as `.mp4` are converted to a Whisper-compatible WAV format through FFmpeg before transcription.

- **Deployment adapter separation**  
  `modal_app.py` contains Modal-specific deployment configuration without modifying the core application architecture. This allows the same `src/` application code to run locally and remotely.

---

## Folder Structure

```text
ai-meeting-minutes-generator/
│
├── app.py
├── modal_app.py
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
│
├── assets/
│   └── README.md
│
├── examples/
│   └── README.md
│
├── tests/
│   ├── __init__.py
│   ├── test_validators.py
│   ├── test_prompts.py
│   └── test_exporters.py
│
└── src/
    ├── __init__.py
    │
    ├── config/
    │   ├── __init__.py
    │   └── settings.py
    │
    ├── models/
    │   ├── __init__.py
    │   └── schemas.py
    │
    ├── prompts/
    │   ├── __init__.py
    │   └── templates.py
    │
    ├── services/
    │   ├── __init__.py
    │   ├── minutes_generator.py
    │   │
    │   ├── transcription/
    │   │   ├── __init__.py
    │   │   └── whisper_service.py
    │   │
    │   └── summarization/
    │       ├── __init__.py
    │       └── llm_service.py
    │
    ├── ui/
    │   ├── __init__.py
    │   ├── theme.py
    │   └── gradio_app.py
    │
    └── utils/
        ├── __init__.py
        ├── exceptions.py
        ├── exporters.py
        ├── hf_auth.py
        ├── logger.py
        └── validators.py
```

---

## Getting Started

### Prerequisites

- Python **3.10+**
- [UV](https://docs.astral.sh/uv/) installed
- FFmpeg installed and available on the system `PATH`
- A Hugging Face account
- Access to `meta-llama/Llama-3.2-3B-Instruct`
- A Hugging Face access token with permission to download the model
- **Recommended for local execution:** NVIDIA GPU with CUDA

The application can fall back to CPU, but LLM generation will be significantly slower without GPU acceleration.

---

### Installation with UV

This project is managed with **UV**.

Clone the repository:

```bash
git clone https://github.com/khaledAlzeer/ai-meeting-minutes-generator.git
cd ai-meeting-minutes-generator
```

Install the dependencies:

```bash
uv sync
```

UV will create or update the project's `.venv` and install the dependencies defined by `pyproject.toml` and `uv.lock`.

To add a new dependency later:

```bash
uv add <package-name>
```

---

### Environment Variables

Create your local `.env` file from the provided template:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

At minimum, configure:

```text
HF_TOKEN=your_hugging_face_token
```

See `.env.example` for the complete configuration reference.

> ⚠️ Never commit your real `.env` file or Hugging Face token to GitHub.

---

### Hugging Face Token Setup

`meta-llama/Llama-3.2-3B-Instruct` is a gated model.

1. Create a Hugging Face account at [huggingface.co/join](https://huggingface.co/join).

2. Visit the [Llama 3.2 3B Instruct model page](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) and accept the required Meta license terms.

3. Create a Hugging Face access token from [Hugging Face Settings → Access Tokens](https://huggingface.co/settings/tokens).

4. A **Read** token is sufficient for downloading the model.

5. Add the token to your local `.env` file:

```text
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

The real token should remain local and must never be committed to the repository.

---

### Running Locally

Start the application with:

```bash
uv run app.py
```

The Gradio server will start on:

```text
http://localhost:7860
```

On the first run, the application may download the Whisper and Llama models from Hugging Face. Depending on your connection and hardware, the first startup may take several minutes.

Subsequent runs reuse the local model cache.

---

### Using Cursor / VS Code

The project works with both [Cursor](https://cursor.com/) and [Visual Studio Code](https://code.visualstudio.com/).

1. Open the `ai-meeting-minutes-generator` project folder.
2. Allow UV to manage the project's `.venv`.
3. Use the integrated terminal to run:

```bash
uv run app.py
```

Run the tests with:

```bash
uv run pytest
```

Add dependencies with:

```bash
uv add <package-name>
```

---

## Usage

1. Launch the application:

```bash
uv run app.py
```

2. Open the Gradio interface in your browser.

3. Either:
   - Upload an audio/video file, or
   - Record a meeting using your microphone.

4. Optionally provide:
   - Meeting title
   - Meeting date
   - Meeting location
   - Attendees

5. Click:

```text
✨ Generate Minutes
```

6. The application will:
   - Validate the uploaded media.
   - Convert supported video/audio formats when required.
   - Transcribe the recording using Whisper.
   - Analyze the transcript using Llama 3.2 3B Instruct.
   - Generate structured meeting minutes.
   - Display the transcript and minutes.
   - Prepare Markdown and text downloads.

7. Review the results in:
   - 📋 **Meeting Minutes**
   - 🎧 **Transcript**

8. Use the download controls to save the generated documents.

9. Click **🗑️ Clear** to reset the interface.

---

## Example Output

For a meeting discussing a project budget, the generated output may look like:

```markdown
# Meeting Minutes

## Meeting Title

Project Budget Review

## Meeting Date

Not specified

## Meeting Location

Not specified

## Attendees

- Project Manager
- Finance Lead
- Engineering Lead

## Executive Summary

The team reviewed the current project budget, discussed
remaining expenses, and agreed on adjustments to upcoming
development priorities.

## Discussion Points

- Current project spending
- Remaining development costs
- Resource allocation
- Upcoming project priorities

## Decisions

- The remaining budget will be prioritized for critical
  development tasks.

## Key Takeaways

- Budget usage is currently within the expected range.
- Several upcoming expenses require monitoring.

## Action Items

- [Finance Lead]&#58; Prepare an updated budget report.
- [Engineering Lead]&#58; Review upcoming development costs.

## Next Steps

- Review the updated budget during the next project meeting.

## Notes

None.
```

> **Note:** Exact output varies depending on the transcript, meeting context, and model generation.

---

## Testing

Run the complete test suite:

```bash
uv run pytest
```

The current test suite covers:

- Input validation
- Prompt construction
- Export functionality

Expected result:

```text
16 passed
```

For coverage:

```bash
uv run pytest --cov=src
```

For code quality:

```bash
uv run ruff check .
```

For Python compilation checks:

```bash
uv run python -m compileall src app.py modal_app.py
```

---

# Deployment on Modal

The production deployment uses **Modal** to host the Gradio interface on a GPU-backed cloud runtime.

The core application remains inside `src/`, while `modal_app.py` acts as the deployment adapter.

### Deployment Architecture

```text
Browser
   │
   ▼
Modal Web Function
   │
   ▼
Gradio UI
   │
   ▼
MeetingMinutesPipeline
   ├── Whisper Medium English
   └── Llama 3.2 3B Instruct
          │
          ▼
     Hugging Face
```

The deployed Modal function uses a GPU runtime and stores the Hugging Face token through a Modal Secret rather than exposing it in the source code.

---

### Install and Authenticate Modal

Install Modal through UV:

```bash
uv add modal
```

Authenticate the local Modal CLI:

```bash
uv run modal setup
```

Verify the active Modal profile:

```bash
uv run modal profile current
```

---

### Create the Hugging Face Secret

The deployment expects a Modal Secret named:

```text
huggingface-secret
```

Create it from your local `.env` file:

```bash
uv run modal secret create huggingface-secret --from-dotenv .env
```

The secret should contain:

```text
HF_TOKEN
```

> ⚠️ Never commit `.env`, your Hugging Face token, or Modal credentials to GitHub.

---

### Test with Modal Serve

Before deploying the application permanently, test the deployment locally through Modal:

```bash
uv run modal serve modal_app.py
```

Modal provides a temporary development URL.

Open the URL in your browser and verify:

- The Gradio interface loads.
- Audio/video uploads work.
- Microphone input works.
- Meeting minutes generation works.
- Transcript generation works.
- Export downloads work.

Stop the development server with:

```text
Ctrl+C
```

> `modal serve` is intended for development and testing. The development URL is not the production deployment.

---

### Deploy to Production

After the application has been tested successfully:

```bash
uv run modal deploy modal_app.py
```

Modal will create a persistent deployment and provide a public web URL.

The production application can then be accessed directly from that URL without running `modal serve` on your local computer.

---

### Updating the Deployment

After making changes to the application:

1. Test locally.
2. Run the automated tests.
3. Commit and push the changes to GitHub.
4. Deploy the updated version:

```bash
uv run modal deploy modal_app.py
```

The Modal deployment uses the code from the current project directory at deployment time.

---

## Configuration Reference

Configuration is centralized in:

```text
src/config/settings.py
```

Environment variables are documented in `.env.example`.

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | None | Hugging Face access token required for the LLM |
| `WHISPER_MODEL_ID` | `openai/whisper-medium.en` | Whisper transcription model |
| `LLM_MODEL_ID` | `meta-llama/Llama-3.2-3B-Instruct` | LLM used for meeting-minutes generation |
| `LLM_MAX_NEW_TOKENS` | `2000` | Maximum generated tokens |
| `LLM_TEMPERATURE` | `0.3` | Generation temperature |
| `LLM_TOP_P` | `0.9` | Nucleus sampling parameter |
| `LLM_REPETITION_PENALTY` | `1.1` | Repetition penalty |
| `USE_4BIT_QUANTIZATION` | `true` | Enables 4-bit quantization on compatible CUDA GPUs |
| `WHISPER_CHUNK_LENGTH_S` | `30` | Whisper processing window |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Server bind address |
| `GRADIO_SERVER_PORT` | `7860` | Server port |
| `GRADIO_SHARE` | `false` | Whether to create a public Gradio share link |
| `GRADIO_QUEUE_MAX_SIZE` | `20` | Maximum Gradio queue size |
| `MAX_AUDIO_SIZE_MB` | `200` | Maximum accepted media size |
| `MIN_TRANSCRIPT_CHARACTERS` | `10` | Minimum transcript length |
| `LOG_LEVEL` | `INFO` | Application logging level |
| `DEVICE_PREFERENCE` | `auto` | Device selection strategy |
| `APP_AUTHOR` | `Khaled Alzeer` | Application author |
| `APP_REPOSITORY_URL` | `https://github.com/khaledAlzeer/ai-meeting-minutes-generator` | Repository URL |

---

## Error Handling

The application handles common failure scenarios including:

- Missing media input
- Unsupported media formats
- Oversized files
- Invalid or empty media files
- FFmpeg conversion failures
- Missing GPU
- Missing or invalid Hugging Face tokens
- Model access restrictions
- Model download failures
- Network errors
- CUDA out-of-memory errors
- Empty transcripts
- Transcripts that are too short
- LLM generation failures

Expected failures are converted into clear, human-readable messages in the UI.

Detailed technical information is written to the application logs for debugging.

---

## Troubleshooting

### "No Hugging Face access token was found"

Make sure your local `.env` contains:

```text
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Never commit this token.

---

### Llama model access error

Make sure your Hugging Face account has access to:

```text
meta-llama/Llama-3.2-3B-Instruct
```

Check the model page:

https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

---

### Generation is extremely slow

The application may be running on CPU.

Check CUDA availability:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

If the result is:

```text
True
```

CUDA is available to PyTorch.

---

### FFmpeg is not found

Make sure FFmpeg is installed and available through the system `PATH`.

Verify it with:

```bash
ffmpeg -version
```

The application uses FFmpeg to normalize supported media formats before Whisper transcription.

---

### "CUDA is required but not available for bitsandbytes"

The application automatically disables 4-bit quantization when CUDA is unavailable and falls back to CPU-compatible loading where supported.

---

### MP4 or video upload fails

Make sure:

- FFmpeg is installed.
- The uploaded file is not corrupted.
- The file size does not exceed the configured limit.
- The file extension is supported by the application.

Video files are converted to a Whisper-compatible WAV representation before transcription.

---

## Future Improvements

Potential future enhancements include:

- 🎤 Speaker diarization to identify individual speakers
- 🌍 Multi-language transcription and meeting-minutes generation
- ⚡ Streaming token-by-token LLM generation
- 🔌 Pluggable hosted transcription and LLM providers
- 🗂️ Persistent meeting history
- 🔎 Search across previous meetings
- 📄 DOCX and PDF exports
- 🧑‍🤝‍🧑 Improved speaker/attendee attribution
- 📊 Meeting analytics and statistics
- 🔐 Authentication and multi-user support

---

## License

This project is released under the [MIT License](LICENSE).

---

## Repository

GitHub:

https://github.com/khaledAlzeer/ai-meeting-minutes-generator