# Voice Agent

One FastAPI server with two separate browser voice modes:

- `Gemini`: native realtime audio-to-audio conversation.
- `Groq`: staged voice test using STT, streaming LLM, and speech output.

Open the main tabbed UI:

```text
http://127.0.0.1:7860
```

## File Layout

```text
voice_agent_server.py  # main FastAPI app: pages, health, websocket route wiring
gemini_agent.py        # Gemini Live websocket pipeline
groq_agent.py          # Groq STT, streaming LLM, and speech pipeline
ui/app.html            # main tabbed page
ui/voice_test.html     # Gemini UI
ui/groq_test.html      # Groq UI
```

Direct routes:

```text
http://127.0.0.1:7860/gemini
http://127.0.0.1:7860/groq
```

## How Gemini Works

Gemini mode is the closest to a phone call.

```text
Browser microphone -> /ws/gemini -> Gemini Live -> browser speaker
```

The browser streams `16 kHz PCM` microphone chunks to the server. The server forwards them to Gemini Live with audio response enabled. Gemini returns `24 kHz PCM` audio chunks, and the browser plays them immediately.

Use this mode when you want the most realtime behavior.

## How Groq Works

Groq mode is separate from Gemini and uses a staged pipeline.

```text
Browser microphone -> WAV utterance -> Groq Whisper STT -> Groq streaming LLM -> browser speech
```

The browser records a short voice clip and sends it to `/ws/groq`. The server uploads that WAV to Groq STT using `whisper-large-v3-turbo`. The transcript is sent to Groq chat completions with streaming enabled. The browser speaks the streamed answer using browser TTS by default.

Groq API TTS can be enabled later with:

```bash
GROQ_TTS_MODE=api
```

That requires accepting Groq Orpheus TTS terms in the Groq Console.

## Groq Voice Testing

For the most reliable Groq voice test:

1. Open the `Groq` tab.
2. Click `Start Groq call`.
3. Click `Record`.
4. Speak your question.
5. Click `Send voice`.

There is also a typed question box in Groq mode. Use it to confirm the Groq LLM and speech output work even if microphone/STT capture needs debugging.

## Environment

Create `.env` from `.env.example`:

```bash
GEMINI_API_KEY=
GROQ_API_KEY=

GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview
GROQ_LLM_MODEL=llama-3.1-8b-instant
GROQ_STT_MODEL=whisper-large-v3-turbo
GROQ_TTS_MODE=browser
WEB_TEST_PORT=7860
```

Optional Groq API TTS settings:

```bash
GROQ_TTS_MODEL=canopylabs/orpheus-v1-english
GROQ_TTS_VOICE=troy
```

## Run

```bash
uv sync
uv run python voice_agent_server.py
```

Or run:

```text
START.bat
```

## Debug Logs

The terminal prints debug logs for both modes.

Gemini logs:

- Browser connection and disconnect
- Mic chunks sent to Gemini
- Audio chunks received from Gemini
- Gemini turn complete and interruption events

Groq logs:

- Browser connection and disconnect
- Voice clip size
- Groq STT transcript
- Streaming Groq LLM output
- Browser/API TTS chunks
- Groq API errors
