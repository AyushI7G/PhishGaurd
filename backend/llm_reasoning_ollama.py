"""Ollama-based LLM reasoning helper.

This module calls a local Ollama server (default: http://localhost:11434)
using the /api/chat endpoint.

Expected to be used by backend/app.py.
"""

from __future__ import annotations

import json
from typing import Optional

import requests


def ollama_chat(
    prompt: str,
    *,
    model: str = "llama3",
    base_url: str = "http://localhost:11434",
    timeout_s: int = 60,
) -> str:
    """Send prompt to Ollama and return assistant text.

    Raises RuntimeError on HTTP/parse failures.
    """

    url = f"{base_url.rstrip('/')}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a cybersecurity assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")

    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Ollama response was not valid JSON: {e}. Raw: {resp.text[:500]}")

    # Ollama chat response: {"message": {"role":"assistant","content":"..."}, ...}
    message = data.get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    # Fallback: try older shape
    content = data.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    raise RuntimeError(f"Ollama response missing content. Raw: {json.dumps(data)[:1000]}")

