#!/usr/bin/env python3
"""Verify the single browser voice agent setup."""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

print("Voice Agent Setup Verification")
print("=" * 48)
print(f"Python {sys.version.split()[0]}")
print()


def masked(value: str) -> str:
    if len(value) < 8:
        return "***"
    return value[:6] + "..." + value[-4:]


def check_env(name: str, required: bool = False) -> bool:
    value = os.getenv(name)
    if value:
        print(f"[ok] {name}: {masked(value)}")
        return True
    if required:
        print(f"[missing] {name}")
        return False
    print(f"[optional] {name}: not set")
    return True


ok = True
print("Environment:")
ok &= check_env("GEMINI_API_KEY", required=True)
check_env("GROQ_API_KEY")
check_env("GEMINI_LIVE_MODEL")
check_env("GROQ_LLM_MODEL")
check_env("GROQ_STT_MODEL")
check_env("GROQ_TTS_MODEL")
check_env("GROQ_TTS_VOICE")
check_env("SYSTEM_PROMPT")
check_env("WEB_TEST_PORT")

print()
print("Imports:")
imports = [
    ("dotenv", "python-dotenv"),
    ("fastapi", "fastapi"),
    ("google.genai", "google-genai"),
    ("httpx", "httpx"),
    ("uvicorn", "uvicorn"),
    ("websockets", "websockets"),
]

for module_name, package_name in imports:
    try:
        __import__(module_name)
        print(f"[ok] {package_name}")
    except ImportError:
        ok = False
        print(f"[missing] {package_name}")

print("=" * 48)
if ok:
    print("Setup check completed.")
    print("Run: uv run python voice_agent_server.py")
else:
    print("Setup check failed. Install missing dependencies first.")
    sys.exit(1)
