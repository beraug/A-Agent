from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pydantic import BaseModel

# Keep phases 0–4 unchanged: add src/ to sys.path and import modules as top-level.
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent import agent_chat, get_memory_stream  # noqa: E402
from llm import DEFAULT_MODEL, bootstrap_ollama  # noqa: E402


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


@app.get("/history")
def history(n: int = 50) -> list[dict]:
    # Last N memory stream items (global, no sessions).
    n = max(0, min(500, n))
    stream = get_memory_stream()
    return stream.get_recent(k=n)

