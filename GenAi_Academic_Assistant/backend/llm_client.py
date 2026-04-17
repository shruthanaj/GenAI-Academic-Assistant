"""
LLM Client
==========
Thin wrapper around the OpenAI SDK.  Swap the model string or base_url
to use LLaMA / Mistral / any OpenAI-compatible endpoint.

Set your key in .env:
    OPENAI_API_KEY=sk-...
Or for a local Ollama server:
    OPENAI_BASE_URL=http://localhost:11434/v1
    OPENAI_API_KEY=ollama
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env from project root (use absolute path for reliability)
project_root = Path(__file__).parent.parent.resolve()
env_path = project_root / ".env"

from dotenv import load_dotenv

# Load environment on module import
load_dotenv(env_path, override=False)

try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False


def call_llm(system: str, user: str, max_tokens: int = 1024) -> str:
    """Call the configured LLM and return the response text."""

    if not _openai_available:
        return (
            "[LLM not configured] Install openai: `pip install openai` "
            "and set OPENAI_API_KEY in your .env file."
        )

    # Reload .env at runtime (ensure fresh env vars)
    load_dotenv(env_path, override=True)

    # Get config at runtime
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

    if not api_key or api_key == "your-api-key-here":
        return (
            "[LLM not configured] Install openai: `pip install openai` "
            "and set OPENAI_API_KEY in your .env file."
        )

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        return f"[LLM Error] {exc}"
