"""AI Meeting Minutes Generator.

A production-ready application that transcribes audio recordings of
meetings using OpenAI's Whisper model and generates structured,
professional meeting minutes using Meta's Llama 3.2 instruction-tuned
language model.

Package layout
--------------
config          Centralized application configuration.
models          Plain data structures (schemas) shared across the app.
services        Business logic: transcription, summarization, orchestration.
prompts         LLM prompt templates, kept separate from business logic.
utils           Cross-cutting utilities: logging, validation, exporting, etc.
ui              Gradio-based user interface.
"""

__version__ = "1.0.0"
