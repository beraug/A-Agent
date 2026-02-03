from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pydantic import BaseModel
from typing import Optional

# Keep phases 0–4 unchanged: add src/ to sys.path and import modules as top-level.
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent import agent_chat, get_memory_stream  # noqa: E402
from llm import DEFAULT_MODEL, bootstrap_ollama  # noqa: E402
from ingest import extract_text_from_upload  # noqa: E402


app = FastAPI()

@app.get("/")
def index():
    return FileResponse("app/static/index.html")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

class ChatIn(BaseModel):
    message: str


class ChatOut(BaseModel):
    reply: str

@app.on_event("startup")
def _startup() -> None:
    # Bootstrap Ollama model at startup (no shell-out).
    bootstrap_ollama(model=DEFAULT_MODEL)


@app.get("/health")
def health() -> dict:
    return {"ollama": "ok", "model": DEFAULT_MODEL}


@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn) -> ChatOut:
    message = (body.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    try:
        reply = agent_chat(message)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ChatOut(reply=reply)

@app.post("/chat-file", response_model=ChatOut)
async def chat_file(
    message: str = Form(""),
    files: list[UploadFile] = File(default=[]),
) -> ChatOut:
    message = (message or "").strip()
    if not message and not files:
        raise HTTPException(status_code=400, detail="message and files cannot both be empty")

    # Bigger caps (override with env vars if desired)
    # - TOTAL_UPLOAD_BYTES: hard cap on bytes read from all uploads combined
    # - MAX_EXTRACT_CHARS_PER_FILE: cap on extracted text per file
    # - MAX_EXTRACT_CHARS_TOTAL: cap on extracted text total across files
    total_upload_bytes_cap = int(os.environ.get("TOTAL_UPLOAD_BYTES", "8000000"))  # 8 MB
    max_chars_per_file = int(os.environ.get("MAX_EXTRACT_CHARS_PER_FILE", "250000"))  # 250k chars
    max_chars_total = int(os.environ.get("MAX_EXTRACT_CHARS_TOTAL", "800000"))  # 800k chars

    used_bytes = 0
    used_chars = 0
    extracted_parts: list[str] = []
    file_summaries: list[str] = []

    for f in files:
        name = f.filename or "upload"
        ctype = (f.content_type or "").lower()

        raw = await f.read()
        if not raw:
            file_summaries.append(f"{name} ({ctype or 'unknown'}) bytes=0")
            continue

        # Enforce total upload bytes cap
        remaining_bytes = max(0, total_upload_bytes_cap - used_bytes)
        raw = raw[:remaining_bytes]
        used_bytes += len(raw)

        file_summaries.append(f"{name} ({ctype or 'unknown'}) bytes={len(raw)}")

        if used_bytes >= total_upload_bytes_cap:
            extracted_parts.append("[Note: uploads truncated due to TOTAL_UPLOAD_BYTES limit]")
            break

        if used_chars >= max_chars_total:
            extracted_parts.append("[Note: extracted text truncated due to MAX_EXTRACT_CHARS_TOTAL limit]")
            break

        text = extract_text_from_upload(
            filename=name,
            content_type=ctype,
            data=raw,
            max_chars=max_chars_per_file,
        ).strip()

        if text:
            # Enforce total extracted chars cap
            remaining_chars = max(0, max_chars_total - used_chars)
            text = text[:remaining_chars]
            used_chars += len(text)
            extracted_parts.append(f"[Extracted: {name}]\n{text}\n")

    context = "\n".join(extracted_parts).strip()
    combined = message
    if context:
        combined = (
            "User provided uploaded files. Use their extracted text as authoritative context.\n\n"
            f"FILES:\n{context}\n\n"
            f"USER MESSAGE:\n{message}"
        )

    # Optionally: store a lightweight note in memory via agent_chat loop (keeps architecture frozen).
    # We only pass it as part of the message; agent loop already stores observations.
    try:
        reply = agent_chat(combined)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ChatOut(reply=reply)

@app.get("/history")
def history(n: int = 50) -> list[dict]:
    # Last N memory stream items (global, no sessions).
    n = max(0, min(500, n))
    stream = get_memory_stream()
    return stream.get_recent(k=n)

