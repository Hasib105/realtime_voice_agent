import asyncio
import logging
import json
import os
import time
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from google import genai
from google.genai import types
import httpx
from websockets.exceptions import ConnectionClosed

load_dotenv()

INPUT_RATE = 16000
DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
APP_UI_FILE = Path(__file__).parent / "ui" / "app.html"
UI_FILE = Path(__file__).parent / "ui" / "voice_test.html"
GROQ_UI_FILE = Path(__file__).parent / "ui" / "groq_test.html"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"
GROQ_DEFAULT_STT_MODEL = "whisper-large-v3-turbo"
GROQ_DEFAULT_TTS_MODEL = "canopylabs/orpheus-v1-english"
GROQ_DEFAULT_VOICE = "troy"
SYSTEM_PROMPT = (
    "You are a warm, concise real-time phone call voice agent. "
    "Reply naturally in short spoken turns. Do not mention that you are an AI unless asked."
)
GROQ_SYSTEM_PROMPT = (
    "You are a warm, concise phone-call assistant. Keep answers short, natural, "
    "and easy to speak aloud. Prefer one or two brief sentences."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("voice-agent")

app = FastAPI()


class PhraseBuffer:
    def __init__(self, min_chars: int = 90) -> None:
        self._min_chars = min_chars
        self._parts: list[str] = []
        self._char_count = 0

    def push(self, token: str) -> str | None:
        if not token:
            return None
        self._parts.append(token)
        self._char_count += len(token)
        text = "".join(self._parts)
        if text.endswith((".", "!", "?", "\n")):
            return self.flush()
        if self._char_count >= self._min_chars and text.endswith((" ", ",", ";", ":")):
            return self.flush()
        return None

    def flush(self) -> str | None:
        if not self._parts:
            return None
        out = "".join(self._parts).strip()
        self._parts.clear()
        self._char_count = 0
        return out or None


def split_for_tts(text: str, max_chars: int = 180) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        chunks.append(" ".join(current))
    return chunks


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(APP_UI_FILE)


@app.get("/gemini")
async def gemini_index() -> FileResponse:
    return FileResponse(UI_FILE)


@app.get("/groq")
async def groq_index() -> FileResponse:
    return FileResponse(GROQ_UI_FILE)


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "gemini_api_key": bool(os.getenv("GEMINI_API_KEY")),
            "groq_api_key": bool(os.getenv("GROQ_API_KEY")),
            "model": os.getenv("GEMINI_LIVE_MODEL", DEFAULT_MODEL),
            "groq_model": os.getenv("GROQ_LLM_MODEL", GROQ_DEFAULT_LLM_MODEL),
        }
    )


@app.websocket("/ws/gemini")
async def gemini_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = uuid4().hex[:8]
    logger.info("[%s] browser connected from %s", session_id, websocket.client)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        await websocket.send_json(
            {"type": "Error", "message": "GEMINI_API_KEY is not set in .env"}
        )
        logger.error("[%s] GEMINI_API_KEY missing", session_id)
        await websocket.close(code=1011)
        return

    model = os.getenv("GEMINI_LIVE_MODEL", DEFAULT_MODEL)
    system_prompt = os.getenv("SYSTEM_PROMPT", SYSTEM_PROMPT)
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})

    async with client.aio.live.connect(
        model=model,
        config={
            "response_modalities": ["AUDIO"],
            "system_instruction": system_prompt,
        },
    ) as session:
        await websocket.send_json({"type": "Session", "message": model})
        logger.info("[%s] Gemini Live session started: %s", session_id, model)

        upstream = asyncio.create_task(browser_to_gemini(websocket, session, session_id))
        downstream = asyncio.create_task(gemini_to_browser(websocket, session, session_id))

        try:
            done, pending = await asyncio.wait(
                {upstream, downstream},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                task.result()
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        except (WebSocketDisconnect, ConnectionClosed):
            logger.info("[%s] browser disconnected", session_id)
        except Exception:
            logger.exception("[%s] session crashed", session_id)
            with suppress(Exception):
                await websocket.send_json({"type": "Error", "message": "Server error. Check terminal debug."})
        finally:
            upstream.cancel()
            downstream.cancel()
            await asyncio.gather(upstream, downstream, return_exceptions=True)
            logger.info("[%s] session closed", session_id)


async def browser_to_gemini(websocket: WebSocket, session, session_id: str) -> None:
    mic_chunks = 0
    mic_bytes = 0
    last_report = time.monotonic()

    while True:
        message = await websocket.receive()

        if "bytes" in message and message["bytes"]:
            mic_chunks += 1
            mic_bytes += len(message["bytes"])
            await session.send_realtime_input(
                audio=types.Blob(
                    data=message["bytes"],
                    mime_type=f"audio/pcm;rate={INPUT_RATE}",
                )
            )
            now = time.monotonic()
            if now - last_report >= 1:
                logger.info(
                    "[%s] mic -> Gemini: %s chunks, %.1f KB",
                    session_id,
                    mic_chunks,
                    mic_bytes / 1024,
                )
                last_report = now
            continue

        if "text" in message and message["text"]:
            with suppress(json.JSONDecodeError):
                data = json.loads(message["text"])
                if data.get("type") == "stop":
                    logger.info("[%s] stop requested by browser", session_id)
                    await session.send_realtime_input(audio_stream_end=True)
                    return
                if data.get("type") == "debug":
                    logger.info("[%s] browser: %s", session_id, data.get("message", "debug"))


async def gemini_to_browser(websocket: WebSocket, session, session_id: str) -> None:
    audio_chunks = 0
    audio_bytes = 0
    last_report = time.monotonic()

    while True:
        async for message in session.receive():
            server_content = message.server_content
            if server_content:
                if getattr(server_content, "interrupted", False):
                    logger.info("[%s] Gemini interrupted current answer", session_id)
                    await websocket.send_json({"type": "Agent", "message": "Interrupted"})

                if server_content.model_turn:
                    for part in server_content.model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            audio_chunks += 1
                            audio_bytes += len(part.inline_data.data)
                            await websocket.send_bytes(part.inline_data.data)
                            now = time.monotonic()
                            if now - last_report >= 1:
                                logger.info(
                                    "[%s] Gemini -> speaker: %s chunks, %.1f KB",
                                    session_id,
                                    audio_chunks,
                                    audio_bytes / 1024,
                                )
                                last_report = now

                if getattr(server_content, "turn_complete", False):
                    logger.info("[%s] Gemini turn complete; waiting for next question", session_id)
                    await websocket.send_json({"type": "Agent", "message": "Ready for next question"})

            if message.text:
                logger.info("[%s] Gemini text: %s", session_id, message.text)
                await websocket.send_json({"type": "Text", "message": message.text})


@app.websocket("/ws/groq")
async def groq_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = uuid4().hex[:8]
    logger.info("[%s] Groq browser connected from %s", session_id, websocket.client)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        await websocket.send_json({"type": "Error", "message": "GROQ_API_KEY is not set in .env"})
        logger.error("[%s] GROQ_API_KEY missing", session_id)
        await websocket.close(code=1011)
        return

    history: list[dict[str, str]] = [
        {"role": "system", "content": os.getenv("GROQ_SYSTEM_PROMPT", GROQ_SYSTEM_PROMPT)}
    ]
    turn_lock = asyncio.Lock()
    send_lock = asyncio.Lock()
    await safe_send_json(
        websocket,
        send_lock,
        {"type": "Session", "message": os.getenv("GROQ_LLM_MODEL", GROQ_DEFAULT_LLM_MODEL)},
    )

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=None)) as http:
            while True:
                message = await websocket.receive()

                if "bytes" in message and message["bytes"]:
                    audio_bytes = message["bytes"]
                    logger.info("[%s] Groq utterance received: %.1f KB", session_id, len(audio_bytes) / 1024)
                    if turn_lock.locked():
                        await safe_send_json(
                            websocket,
                            send_lock,
                            {"type": "Busy", "message": "Still answering. Try again in a moment."},
                        )
                        continue
                    asyncio.create_task(
                        handle_groq_turn(
                            websocket,
                            http,
                            api_key,
                            history,
                            audio_bytes,
                            session_id,
                            turn_lock,
                            send_lock,
                        )
                    )
                    continue

                if "text" in message and message["text"]:
                    with suppress(json.JSONDecodeError):
                        data = json.loads(message["text"])
                        if data.get("type") == "stop":
                            logger.info("[%s] Groq stop requested by browser", session_id)
                            return
                        if data.get("type") == "debug":
                            logger.info("[%s] Groq browser: %s", session_id, data.get("message", "debug"))
                        if data.get("type") == "ask_text":
                            user_text = str(data.get("message", "")).strip()
                            if not user_text:
                                continue
                            if turn_lock.locked():
                                await safe_send_json(
                                    websocket,
                                    send_lock,
                                    {"type": "Busy", "message": "Still answering. Try again in a moment."},
                                )
                                continue
                            asyncio.create_task(
                                handle_groq_text_turn(
                                    websocket,
                                    http,
                                    api_key,
                                    history,
                                    user_text,
                                    session_id,
                                    turn_lock,
                                    send_lock,
                                )
                            )
    except WebSocketDisconnect:
        logger.info("[%s] Groq browser disconnected", session_id)
    except Exception:
        logger.exception("[%s] Groq session crashed", session_id)
        with suppress(Exception):
            await safe_send_json(
                websocket,
                send_lock,
                {"type": "Error", "message": "Groq server error. Check terminal debug."},
            )
    finally:
        logger.info("[%s] Groq session closed", session_id)


async def safe_send_json(websocket: WebSocket, send_lock: asyncio.Lock, payload: dict[str, str]) -> None:
    async with send_lock:
        await websocket.send_json(payload)


async def safe_send_bytes(websocket: WebSocket, send_lock: asyncio.Lock, payload: bytes) -> None:
    async with send_lock:
        await websocket.send_bytes(payload)


async def handle_groq_turn(
    websocket: WebSocket,
    http: httpx.AsyncClient,
    api_key: str,
    history: list[dict[str, str]],
    audio_bytes: bytes,
    session_id: str,
    turn_lock: asyncio.Lock,
    send_lock: asyncio.Lock,
) -> None:
    async with turn_lock:
        try:
            await safe_send_json(websocket, send_lock, {"type": "STT", "message": "Transcribing your voice"})
            user_text = await groq_transcribe(http, api_key, audio_bytes)
            if not user_text:
                await safe_send_json(
                    websocket,
                    send_lock,
                    {"type": "NoSpeech", "message": "I did not catch speech in that clip."},
                )
                return

            logger.info("[%s] Groq STT: %s", session_id, user_text)
            await run_groq_answer(websocket, http, api_key, history, user_text, session_id, send_lock)
        except Exception as exc:
            logger.exception("[%s] Groq turn failed", session_id)
            await safe_send_json(websocket, send_lock, {"type": "Error", "message": str(exc)})


async def handle_groq_text_turn(
    websocket: WebSocket,
    http: httpx.AsyncClient,
    api_key: str,
    history: list[dict[str, str]],
    user_text: str,
    session_id: str,
    turn_lock: asyncio.Lock,
    send_lock: asyncio.Lock,
) -> None:
    async with turn_lock:
        try:
            logger.info("[%s] Groq typed input: %s", session_id, user_text)
            await run_groq_answer(websocket, http, api_key, history, user_text, session_id, send_lock)
        except Exception as exc:
            logger.exception("[%s] Groq typed turn failed", session_id)
            await safe_send_json(websocket, send_lock, {"type": "Error", "message": str(exc)})


async def run_groq_answer(
    websocket: WebSocket,
    http: httpx.AsyncClient,
    api_key: str,
    history: list[dict[str, str]],
    user_text: str,
    session_id: str,
    send_lock: asyncio.Lock,
) -> None:
    await safe_send_json(websocket, send_lock, {"type": "UserText", "message": user_text})
    history.append({"role": "user", "content": user_text})

    await safe_send_json(websocket, send_lock, {"type": "LLM", "message": "Streaming answer"})
    assistant_text = await groq_stream_llm_and_tts(
        websocket,
        http,
        api_key,
        history,
        session_id,
        send_lock,
    )
    if assistant_text:
        history.append({"role": "assistant", "content": assistant_text})
        del history[1:-10]
    await safe_send_json(websocket, send_lock, {"type": "Done", "message": "Ready for next question"})


async def groq_transcribe(http: httpx.AsyncClient, api_key: str, audio_bytes: bytes) -> str:
    response = await http.post(
        f"{GROQ_BASE_URL}/audio/transcriptions",
        headers={"Authorization": f"Bearer {api_key}"},
        data={
            "model": os.getenv("GROQ_STT_MODEL", GROQ_DEFAULT_STT_MODEL),
            "response_format": "json",
            "language": os.getenv("GROQ_STT_LANGUAGE", "en"),
            "temperature": "0",
        },
        files={"file": ("utterance.wav", audio_bytes, "audio/wav")},
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:400]
        raise RuntimeError(f"Groq STT failed {exc.response.status_code}: {detail}") from exc
    payload = response.json()
    return str(payload.get("text", "")).strip()


async def groq_stream_llm_and_tts(
    websocket: WebSocket,
    http: httpx.AsyncClient,
    api_key: str,
    history: list[dict[str, str]],
    session_id: str,
    send_lock: asyncio.Lock,
) -> str:
    text_queue: asyncio.Queue[str | None] = asyncio.Queue()
    tts_task = asyncio.create_task(groq_tts_worker(websocket, http, api_key, text_queue, session_id, send_lock))
    phrase_buffer = PhraseBuffer()
    assistant_parts: list[str] = []

    try:
        async with http.stream(
            "POST",
            f"{GROQ_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("GROQ_LLM_MODEL", GROQ_DEFAULT_LLM_MODEL),
                "messages": history,
                "temperature": float(os.getenv("GROQ_LLM_TEMPERATURE", "0.35")),
                "max_completion_tokens": int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "220")),
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content") or ""
                if not delta:
                    continue
                assistant_parts.append(delta)
                await safe_send_json(websocket, send_lock, {"type": "TextDelta", "message": delta})
                phrase = phrase_buffer.push(delta)
                if phrase:
                    await text_queue.put(phrase)

        remaining = phrase_buffer.flush()
        if remaining:
            await text_queue.put(remaining)
        await text_queue.put(None)
        await tts_task
        assistant_text = "".join(assistant_parts).strip()
        logger.info("[%s] Groq LLM: %s", session_id, assistant_text)
        return assistant_text
    except Exception:
        await text_queue.put(None)
        tts_task.cancel()
        await asyncio.gather(tts_task, return_exceptions=True)
        raise


async def groq_tts_worker(
    websocket: WebSocket,
    http: httpx.AsyncClient,
    api_key: str,
    text_queue: asyncio.Queue[str | None],
    session_id: str,
    send_lock: asyncio.Lock,
) -> None:
    while True:
        text = await text_queue.get()
        if text is None:
            return
        for chunk in split_for_tts(text):
            if os.getenv("GROQ_TTS_MODE", "browser").lower() != "api":
                logger.info("[%s] Groq browser TTS chunk: %r", session_id, chunk[:80])
                await safe_send_json(websocket, send_lock, {"type": "TTSFallback", "message": chunk})
                continue

            await safe_send_json(websocket, send_lock, {"type": "TTS", "message": chunk})
            try:
                audio = await groq_speech(http, api_key, chunk)
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text[:300]
                logger.warning("[%s] Groq TTS failed, using browser TTS fallback: %s", session_id, detail)
                await safe_send_json(
                    websocket,
                    send_lock,
                    {
                        "type": "TTSFallback",
                        "message": chunk,
                    },
                )
                continue
            logger.info("[%s] Groq TTS audio: %.1f KB for %r", session_id, len(audio) / 1024, chunk[:80])
            await safe_send_bytes(websocket, send_lock, audio)


async def groq_speech(http: httpx.AsyncClient, api_key: str, text: str) -> bytes:
    response = await http.post(
        f"{GROQ_BASE_URL}/audio/speech",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": os.getenv("GROQ_TTS_MODEL", GROQ_DEFAULT_TTS_MODEL),
            "voice": os.getenv("GROQ_TTS_VOICE", GROQ_DEFAULT_VOICE),
            "input": text,
            "response_format": "wav",
        },
    )
    response.raise_for_status()
    return response.content


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("WEB_TEST_PORT", "7860"))
    uvicorn.run(app, host="127.0.0.1", port=port)
