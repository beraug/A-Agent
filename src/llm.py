"""
Phase 1: Ollama as the LLM backend.

Single module for all agent reasoning. Uses the Ollama API (localhost:11434)
so the course runs fully offline with no API keys.

Usage:
    from llm import complete, complete_stream

    text = complete("What is 2+2?")
    for chunk in complete_stream("Tell me a joke"):
        print(chunk, end="")
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Generator, Optional

# Default base URL for Ollama (override with OLLAMA_HOST env if needed).
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


class OllamaError(Exception):
    """Raised when Ollama is unreachable or returns an error."""

    pass


def _get_base_url() -> str:
    import os
    return os.environ.get("OLLAMA_HOST", DEFAULT_BASE_URL).rstrip("/")


def complete(
    prompt: str,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
) -> str:
    """
    Call Ollama with a prompt and optional system message; return the full reply.

    Args:
        prompt: User message (or full prompt if no system).
        system: Optional system prompt (e.g. "You are a helpful assistant.").
        model: Ollama model name (e.g. "llama3.2", "mistral").
        base_url: Ollama API base URL (default: OLLAMA_HOST or localhost:11434).

    Returns:
        The assistant's reply as a single string.

    Raises:
        OllamaError: If Ollama is not running or returns an error.
    """
    url = (base_url or _get_base_url()) + "/api/chat"
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        if "Connection refused" in str(e) or "nodename nor servname" in str(e):
            raise OllamaError(
                "Ollama is not running. Start it with: ollama serve (or run the Ollama app)."
            ) from e
        raise OllamaError(f"Cannot reach Ollama: {e}") from e
    except json.JSONDecodeError as e:
        raise OllamaError(f"Invalid response from Ollama: {e}") from e

    if "message" not in data or "content" not in data["message"]:
        raise OllamaError(f"Unexpected Ollama response: {data}")

    return data["message"]["content"].strip()


def complete_stream(
    prompt: str,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
) -> Generator[str, None, None]:
    """
    Call Ollama with streaming; yield text chunks as they arrive.

    Same arguments as complete(). Yields partial content strings.
    """
    url = (base_url or _get_base_url()) + "/api/chat"
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    content = data["message"]["content"]
                    if content:
                        yield content
                if data.get("done"):
                    break
    except urllib.error.URLError as e:
        if "Connection refused" in str(e) or "nodename nor servname" in str(e):
            raise OllamaError(
                "Ollama is not running. Start it with: ollama serve (or run the Ollama app)."
            ) from e
        raise OllamaError(f"Cannot reach Ollama: {e}") from e


def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_BASE_URL).rstrip("/")


def pull_model(host: str, model: str, timeout_s: int) -> Dict[str, Any]:
    """
    Pull/download a model using Ollama's /api/pull endpoint.
    stream=False => response is a single JSON object (not a stream).
    """
    url = f"{host.rstrip('/')}/api/pull"
    payload = {"model": model, "stream": False}

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            status_code = resp.status
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot connect to Ollama at {host}. Is Ollama installed and running? ({e})"
        ) from e

    if status_code < 200 or status_code >= 300:
        detail: Optional[str] = None
        try:
            detail = json.loads(raw).get("error")
        except Exception:
            detail = raw.strip() or None
        raise RuntimeError(f"Pull failed (HTTP {status_code}): {detail}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Pull returned non-JSON response: {raw[:200]}") from e

    data.setdefault("model", model)
    data.setdefault("status", "ok")
    return data


def list_models(host: str, timeout_s: int) -> Dict[str, Any]:
    """
    List available models via /api/tags.
    """
    url = f"{host.rstrip('/')}/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            status_code = resp.status
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to reach {url} ({e})") from e

    if status_code < 200 or status_code >= 300:
        raise RuntimeError(f"Tags failed (HTTP {status_code}): {raw[:200]}")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Tags returned non-JSON response: {raw[:200]}") from e


def is_model_present(tags_json: Dict[str, Any], model: str) -> bool:
    """
    Ollama tags payload typically has: {"models": [{"name": "...", ...}, ...]}
    We'll match the name exactly; if user passes "llama3.2" but Ollama lists "llama3.2:latest",
    we also allow prefix match up to ':' if that seems reasonable.
    """
    models = tags_json.get("models") or []
    names = [m.get("name") for m in models if isinstance(m, dict)]
    if model in names:
        return True

    if ":" not in model:
        for n in names:
            if isinstance(n, str) and n.startswith(model + ":"):
                return True

    return False


def warm_model(host: str, model: str, timeout_s: int) -> None:
    """
    Force the model to load by issuing a tiny chat request.
    """
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            status_code = resp.status
    except urllib.error.URLError as e:
        raise RuntimeError(f"Warm-up failed calling {url} ({e})") from e

    if status_code < 200 or status_code >= 300:
        detail: Optional[str] = None
        try:
            detail = json.loads(raw).get("error")
        except Exception:
            detail = raw.strip() or None
        raise RuntimeError(f"Warm-up failed (HTTP {status_code}): {detail}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"Warm-up returned non-JSON response: {raw[:200]}")

    msg = (data.get("message") or {}).get("content")
    if not isinstance(msg, str) or not msg.strip():
        raise RuntimeError("Warm-up succeeded but response content was empty/unexpected.")


def bootstrap_ollama(
    model: str = DEFAULT_MODEL,
    host: str | None = None,
    pull_timeout_s: int = 900,
    tags_timeout_s: int = 10,
    warm_timeout_s: int = 60,
) -> None:
    """
    Bootstrap Ollama model (extracted from test_ollama.py):
      1) Pull via POST /api/pull (stream=False)
      2) Verify availability via GET /api/tags
      3) Warm via POST /api/chat ("ping")
    """
    host = host or _ollama_host()

    pull_result = pull_model(host=host, model=model, timeout_s=pull_timeout_s)
    tags = list_models(host=host, timeout_s=tags_timeout_s)
    present = is_model_present(tags, model)
    if not present:
        names = [m.get("name") for m in (tags.get("models") or []) if isinstance(m, dict)]
        raise RuntimeError(f"Model '{model}' not found in /api/tags. Available models: {names}")

    warm_model(host=host, model=model, timeout_s=warm_timeout_s)

    # Keep pull_result referenced to match the original flow (useful for debugging)
    _ = pull_result
