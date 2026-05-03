import asyncio
import json
import logging
import os
import time
from contextlib import suppress
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
from websockets.exceptions import ConnectionClosed

INPUT_RATE = 16000
DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
SYSTEM_PROMPT = (
    "You are a warm, concise real-time phone call voice agent. "
    "Reply naturally in short spoken turns. Do not mention that you are an AI unless asked."
)

logger = logging.getLogger("voice-agent")


async def gemini_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = uuid4().hex[:8]
    logger.info("[%s] browser connected from %s", session_id, websocket.client)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        await websocket.send_json({"type": "Error", "message": "GEMINI_API_KEY is not set in .env"})
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
