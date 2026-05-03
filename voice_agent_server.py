import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse

load_dotenv()

from gemini_agent import DEFAULT_MODEL as GEMINI_DEFAULT_MODEL
from gemini_agent import gemini_socket
from groq_agent import GROQ_DEFAULT_LLM_MODEL, groq_socket

BASE_DIR = Path(__file__).parent
APP_UI_FILE = BASE_DIR / "ui" / "app.html"
GEMINI_UI_FILE = BASE_DIR / "ui" / "voice_test.html"
GROQ_UI_FILE = BASE_DIR / "ui" / "groq_test.html"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(title="Voice Agent")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(APP_UI_FILE)


@app.get("/gemini")
async def gemini_index() -> FileResponse:
    return FileResponse(GEMINI_UI_FILE)


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
            "model": os.getenv("GEMINI_LIVE_MODEL", GEMINI_DEFAULT_MODEL),
            "groq_model": os.getenv("GROQ_LLM_MODEL", GROQ_DEFAULT_LLM_MODEL),
        }
    )


@app.websocket("/ws/gemini")
async def gemini_websocket(websocket: WebSocket) -> None:
    await gemini_socket(websocket)


@app.websocket("/ws/groq")
async def groq_websocket(websocket: WebSocket) -> None:
    await groq_socket(websocket)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("WEB_TEST_PORT", "7860"))
    uvicorn.run(app, host="127.0.0.1", port=port)
