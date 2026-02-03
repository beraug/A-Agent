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

from agent import agent_chat, get_memory_stream, reset_memory_stream, reply_with_ephemeral_context  # noqa: E402
from llm import DEFAULT_MODEL, bootstrap_ollama  # noqa: E402
from ingest import extract_text_from_upload  # noqa: E402

MEMORY_DB_PATH = Path(os.environ.get("MEMORY_DB_PATH", "data/memory.db"))

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

@app.post("/memory/clear")
def clear_memory() -> dict:
    """
    Hard reset: close DB, delete SQLite files (db/wal/shm), recreate empty DB.
    """
    try:
        stream = get_memory_stream()
        db_path = Path(stream.db_path)

        # Close current connection first
        try:
            stream.close()
        except Exception:
            pass

        # Delete db + WAL sidecars
        for p in [
            db_path,
            Path(str(db_path) + "-wal"),
            Path(str(db_path) + "-shm"),
        ]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

        # Recreate empty DB (schema created in MemoryStream.__init__)
        reset_memory_stream(db_path=str(db_path))
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    tagged_notes: list[tuple[str, str]] = []  # (type, content)

    for f in files:
        name = f.filename or "upload"
        ctype = (f.content_type or "").lower()

        raw = await f.read()
        if not raw:
            tagged_notes.append(("upload", f"User uploaded empty file: {name} ({ctype or 'unknown'})"))
            continue

        # Enforce total upload bytes cap
        remaining_bytes = max(0, total_upload_bytes_cap - used_bytes)
        raw = raw[:remaining_bytes]
        used_bytes += len(raw)

        # Tag per file type for later filtered retrieval.
        # NOTE: We store ONLY lightweight notes here, not full extracted text (avoids memory bloat).
        if ctype.startswith("image/") or name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            tagged_notes.append(("image_ocr", f"User uploaded image for OCR: {name} ({ctype}) bytes={len(raw)}"))
        elif ctype == "application/pdf" or name.lower().endswith(".pdf"):
            tagged_notes.append(("document", f"User uploaded PDF: {name} bytes={len(raw)}"))
        elif name.lower().endswith(".docx") or "wordprocessingml.document" in ctype:
            tagged_notes.append(("document", f"User uploaded DOCX: {name} bytes={len(raw)}"))
        else:
            tagged_notes.append(("upload", f"User uploaded file: {name} ({ctype or 'unknown'}) bytes={len(raw)}"))

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

    extra_context = "\n".join(extracted_parts).strip()


    # Optionally: store a lightweight note in memory via agent_chat loop (keeps architecture frozen).
    # We only pass it as part of the message; agent loop already stores observations.
    try:
        stream = get_memory_stream()
        reply = reply_with_ephemeral_context(
            stream,
            message,
            extra_context=extra_context,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ChatOut(reply=reply)

@app.get("/history")
def history(n: int = 50) -> list[dict]:
    # Last N memory stream items (global, no sessions).
    n = max(0, min(500, n))
    stream = get_memory_stream()
    return stream.get_recent(k=n)
