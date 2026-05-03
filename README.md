# Voice Agent

The main page has tabs for two separate voice agent modes:

Gemini keeps the current native realtime path:

`Microphone -> Gemini Live -> Speaker`

Groq adds a separate test path:

`Detected utterance WAV -> Groq STT -> streaming Groq LLM -> browser speech`

Groq API TTS can be enabled with `GROQ_TTS_MODE=api` after accepting Orpheus TTS terms in Groq Console.

The UI shows:

- Connection state
- Your microphone level
- When your voice is detected
- When the agent is speaking
- Recent call events

The terminal prints debug logs for browser connection, microphone audio sent to Gemini, agent audio received from Gemini, speech-state changes, errors, and disconnects.
The Groq test logs utterance uploads, STT text, streamed LLM text, TTS chunks, errors, and disconnects.

## Environment

Create `.env` with:

```bash
GEMINI_API_KEY=...
GROQ_API_KEY=...

# Optional
GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview
SYSTEM_PROMPT=You are a warm, concise real-time phone call voice agent.
GROQ_LLM_MODEL=llama-3.1-8b-instant
GROQ_STT_MODEL=whisper-large-v3-turbo
GROQ_TTS_MODEL=canopylabs/orpheus-v1-english
GROQ_TTS_VOICE=troy
WEB_TEST_PORT=7860
```

## Run

```bash
uv sync
uv run python voice_agent_server.py
```

Open:

```text
http://127.0.0.1:7860
```

Direct routes:

```text
http://127.0.0.1:7860/gemini
http://127.0.0.1:7860/groq
```

Or run `START.bat`.
