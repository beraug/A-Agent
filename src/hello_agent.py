"""
Phase 0: Minimal "hello world" for a generative agent.

This script:
  1. Loads environment variables from .env
  2. Gets a reply from an LLM using (in order of preference):
     - Ollama (default, offline) — set LLM_BACKEND=ollama or leave unset
     - OpenAI — set LLM_BACKEND=openai and OPENAI_API_KEY in .env
     - Mock — set MOCK_LLM=1 or leave Ollama/OpenAI unavailable
  3. Prints the model's response

Run from project root: python src/hello_agent.py
Ollama (default): install from ollama.ai, run "ollama pull llama3.2", then run this script.
"""

import os
import sys
from pathlib import Path

# Ensure src is on path so we can import llm when run as script.
_src = Path(__file__).resolve().parent
if _src not in sys.path:
    sys.path.insert(0, str(_src))

from dotenv import load_dotenv

from llm import OllamaError, complete as ollama_complete

# Optional: OpenAI only used when LLM_BACKEND=openai and key is set.
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

MOCK_AGENT_REPLY = (
    "Hello! I am a generative agent — I remember, reflect, and plan like the "
    "agents in the paper. (This is a mock reply; no API was called.)"
)

HELLO_PROMPT = (
    "Say hello in one short sentence as a generative agent "
    "(a software agent that simulates believable human behavior)."
)


def _run_ollama() -> str | None:
    """Use Ollama; return reply or None on failure."""
    try:
        return ollama_complete(HELLO_PROMPT)
    except OllamaError:
        return None


def _run_openai(api_key: str) -> str | None:
    """Use OpenAI; return reply or None on failure."""
    if OpenAI is None:
        return None
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": HELLO_PROMPT}],
            max_tokens=150,
        )
        return response.choices[0].message.content
    except Exception:
        return None


def main() -> None:
    load_dotenv()

    mock_env = os.environ.get("MOCK_LLM", "").strip().lower()
    use_mock = mock_env in ("1", "true", "yes", "on")
    backend = (os.environ.get("LLM_BACKEND") or "ollama").strip().lower()
    api_key = os.environ.get("OPENAI_API_KEY")

    if use_mock:
        text = MOCK_AGENT_REPLY
        print("Agent says:", text)
        return

    # Prefer Ollama (default for course, works offline).
    if backend in ("ollama", ""):
        text = _run_ollama()
        if text is not None:
            print("Agent says (Ollama):", text)
            return
        print(
            "Ollama not available (is it running? try: ollama serve). "
            "Falling back to OpenAI or mock."
        )

    # Fallback: OpenAI if requested and key set.
    if backend == "openai" and api_key:
        text = _run_openai(api_key)
        if text is not None:
            print("Agent says (OpenAI):", text)
            return
        print("OpenAI call failed. Using mock reply.")

    # Last resort: mock.
    if backend == "openai" and not api_key:
        print(
            "LLM_BACKEND=openai but OPENAI_API_KEY is not set. "
            "Set OPENAI_API_KEY in .env or use Ollama (default)."
        )
    print("Agent says:", MOCK_AGENT_REPLY)


if __name__ == "__main__":
    main()
