# Examples

Drop sample audio files here (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`,
`.webm`) to have them automatically appear as quick-start examples in the
Gradio UI's "Example recordings" section — see
`src/ui/gradio_app.py::_discover_example_audio_files`.

Note: `.gitignore` excludes `*.mp3` / `*.wav` by default everywhere
*except* this folder, so committed example recordings here will not be
accidentally excluded. Keep sample files small (a minute or two of audio
is plenty) to keep the repository lightweight.
