# Practical Course: Build a Website Running an Agentic AI System

A step-by-step course for junior coders to build a **fully functional website** that runs a **generative agent** (simulacra-style) using **offline Ollama models**. The architecture follows the [Generative Agents paper](PAPER_SUMMARY.md) (Park et al., UIST 2023): **Memory**, **Reflection**, and **Planning**.

---

## Course Overview

| Goal | Outcome |
|------|--------|
| **Paper** | Implement the three pillars: Memory stream, Reflection, Planning |
| **Backend** | Python API (FastAPI) that runs the agent logic |
| **Frontend** | Simple web UI to chat with the agent and see its “thoughts” |
| **LLM** | Ollama (local models, no API key, works offline) |

**Prerequisites:** Basic Python, minimal HTML/JS, terminal. No prior AI/ML experience required.

---

## Tech Stack

- **Backend:** Python 3.10+, FastAPI, `ollama` Python client (or HTTP calls to `localhost:11434`)
- **LLM:** Ollama with a small model (e.g. `llama3.2`, `mistral`, `phi3`) — runs fully offline
- **Storage:** Start with in-memory / JSON files; later SQLite for memory and reflections
- **Frontend:** Simple HTML + vanilla JS or a lightweight framework; or a minimal React/Vue app that talks to the API

---

## Phase 0: Environment and “Hello Agent” (Already in Repo)

**Goal:** Run a minimal script that gets a reply from an LLM.

- [x] Install Ollama: [ollama.ai](https://ollama.ai) — pull a model, e.g. `ollama pull llama3.2`
- [x] Run existing `src/hello_agent.py` (OpenAI/mock). Same script now uses **Ollama by default**; set `LLM_BACKEND=openai` or `MOCK_LLM=1` to override.
- [x] **Deliverable:** One script that prints an agent reply using either OpenAI (optional) or Ollama (default for course).

**Concepts:** Environment variables, one LLM call, no memory yet.

---

## Phase 1: Ollama as the LLM Backend ✅

**Goal:** All agent reasoning uses Ollama; no cloud API required.

- [x] Install `ollama` (CLI) and pull a model (e.g. `llama3.2:latest` or `mistral`).
- [x] Use the Ollama API: `POST http://localhost:11434/api/chat` with JSON body (model, messages, stream).
- [x] Create `src/llm.py`: `def complete(prompt: str, system: str | None = None) -> str` that calls Ollama and returns the generated text. Handle “Ollama not running” with a clear error.
- [x] Optional: `complete_stream()` yields text chunks for the UI.
- [x] **Deliverable:** `src/llm.py` with `complete()` and `complete_stream()`, used by the rest of the course.

**Concepts:** Local LLM, prompt engineering, sync vs stream.

---

## Phase 2: Memory Stream (Paper Pillar 1)

**Goal:** The agent has a **time-ordered stream of observations** (memory stream).

- [ ] Define a **memory record**: e.g. `{ "id": "...", "timestamp": "ISO8601", "content": "User said: ...", "importance": 0.0–1.0 }`. Importance can be a fixed value at first, or later inferred by the LLM.
- [ ] Implement **storage**: in-memory list or append-only JSON file. Later replace with SQLite.
- [ ] **Add memory:** When the user sends a message, append an observation (e.g. “User said: …”) with timestamp.
- [ ] **Retrieve memory:** Function `retrieve(query: str, k: int = 5)` that returns the top-k memories. Start with **recency only** (last k). Then add **relevance**: use Ollama to embed or score memories (or simple keyword match) and combine with recency (as in the paper).
- [ ] **Deliverable:** `src/memory.py` with `add_observation(content, importance?)`, `retrieve(query, k)`, and a clear representation of the memory stream. Unit tests with fixed timestamps.

**Concepts:** Time-ordered stream, retrieval (recency, later relevance/importance).

---

## Phase 3: Reflection (Paper Pillar 2)

**Goal:** Periodically **synthesize recent memories into higher-level reflections** (beliefs, patterns).

- [ ] **Trigger:** After every N new observations, or on a timer, run a “reflection” step.
- [ ] **Prompt:** Give the LLM the last K memories and ask: “Summarize patterns, beliefs, or important facts about this agent/user.” One reflection = one new “memory” of type `reflection` stored in the same stream (or a separate reflection store).
- [ ] Store reflections with timestamp and maybe `type: "reflection"` so they can be retrieved like normal memories.
- [ ] **Deliverable:** `src/reflection.py` (or inside `memory.py`) that runs reflection using `src/llm.py` and appends results to the memory stream. Integrate into the agent loop (e.g. after each user message, check “should we reflect?”).

**Concepts:** Compression of experience, identity/preferences (paper’s “believable behavior”).

---

## Phase 4: Planning (Paper Pillar 3)

**Goal:** Use **retrieved memories (and reflections) to plan** the agent’s response and behavior.

- [ ] When the user sends a message:
  1. **Retrieve** relevant memories + reflections (`memory.retrieve(user_message, k=10)`).
  2. **Plan:** Build a prompt that includes: current message, retrieved context, and an instruction like “Given this context, decide what the agent should do or say next. Be concise.”
  3. Call Ollama with that prompt; the model’s output is the “plan” or direct reply.
- [ ] Optionally split into two steps: (a) “internal plan” (what to do) and (b) “response to user” (what to say). Start with (b) only for simplicity.
- [ ] **Store:** Append the agent’s “action” or “reply” as a new observation (e.g. “Agent replied: …”) so future retrieval sees it.
- [ ] **Deliverable:** `src/agent.py` (or `src/planning.py`) that wires Memory → Retrieve → Plan (LLM) → Response, and appends new observations. The agent loop is: user message → retrieve → plan → reply → add memories.

**Concepts:** Retrieval-augmented response, plan-as-prompt, closing the loop (paper’s “planning ties memory to action”).

---

## Phase 5: REST API (Backend for the Website)

**Goal:** A **FastAPI** app that the website can call.

- [ ] Create `src/api.py` or `app/main.py`: FastAPI app, CORS enabled for the frontend origin.
- [ ] Endpoints:
  - `POST /chat` — body: `{ "message": "..." }`. Runs the agent loop (retrieve → reflect if needed → plan → reply), returns `{ "reply": "...", "memories_used": [...] }` (optional).
  - `GET /history` — returns recent memory stream (or last N messages) for the UI.
  - `GET /health` — returns `{ "ollama": "ok" }` if Ollama is reachable (optional).
- [ ] Use a single global agent instance (or per-session if you add session IDs later).
- [ ] **Deliverable:** Run with `uvicorn`; `curl` and browser can hit `/chat` and `/history`.

**Concepts:** REST, request/response, CORS.

---

## Phase 6: Simple Web Frontend

**Goal:** A **fully functional website** that talks to your API.

- [ ] Create a static page (e.g. `static/index.html` or a small SPA in `frontend/`).
- [ ] **UI:** Text input for the user message, “Send” button, area that displays the conversation (user + agent messages). Optional: “Show memories used” or “Show last reflection” for learning.
- [ ] **JS:** On Send, `fetch('http://localhost:8000/chat', { method: 'POST', body: JSON.stringify({ message }) })`, then append the reply to the conversation.
- [ ] Serve the static files from FastAPI (`app.mount("/", StaticFiles(...))`) or any simple HTTP server; ensure the API URL in the frontend matches your backend (e.g. relative `/chat` if same origin).
- [ ] **Deliverable:** Open the page in a browser; type a message; see the agent reply. No login required for the minimal version.

**Concepts:** Fetch API, DOM updates, same-origin vs CORS.

---

## Phase 7: Polish and Offline-First

**Goal:** Robust, runnable entirely offline with Ollama.

- [ ] **Startup:** Document: “Start Ollama first (`ollama serve` or run the app), then start the API, then open the frontend.”
- [ ] **Errors:** If Ollama is down, return a clear JSON error and show “Agent unavailable (is Ollama running?)” in the UI.
- [ ] **Persistence:** Replace in-memory/JSON memory with SQLite so conversations survive restarts. Optional: export/import of memory stream for debugging.
- [ ] **Deliverable:** One-command (or two-command) run: e.g. `ollama serve & uvicorn app.main:app --reload`, and a README that says exactly how to run the full stack offline.

---

## Suggested Order and Time (Rough)

| Phase | Focus | Suggested time |
|-------|--------|-----------------|
| 0 | Hello Agent + Ollama path | 0.5–1 day |
| 1 | Ollama backend (`llm.py`) | 1 day |
| 2 | Memory stream | 1–2 days |
| 3 | Reflection | 1 day |
| 4 | Planning / agent loop | 1–2 days |
| 5 | FastAPI | 0.5–1 day |
| 6 | Web frontend | 1 day |
| 7 | Polish, SQLite, docs | 1 day |

Total: about **1–2 weeks** for a junior coder moving step by step.

---

## Mapping to the Paper

| Paper concept | Course implementation |
|---------------|------------------------|
| **Memory stream** | Phase 2: time-ordered observations, `add_observation`, `retrieve` (recency + relevance) |
| **Reflection** | Phase 3: periodic synthesis of memories into reflections, stored as special memories |
| **Planning** | Phase 4: retrieve memories → prompt LLM → plan/reply → store new observations |
| **Believable behavior** | Emerges from using all three: memory gives context, reflection gives identity, planning gives coherent actions |

---

## Resources

- **Ollama:** [ollama.ai](https://ollama.ai), `ollama pull llama3.2`, API: `http://localhost:11434`
- **Paper summary:** [PAPER_SUMMARY.md](PAPER_SUMMARY.md) in this repo
- **Existing code:** `src/hello_agent.py` — extend with Ollama in Phase 0/1

---

## Next Step

Start with **Phase 0 + Phase 1:** get Ollama running and implement `src/llm.py` so that every subsequent phase uses local models only. Then implement Phase 2 (memory) and build upward.
