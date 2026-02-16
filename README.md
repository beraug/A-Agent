# Agentic AI Playground (Offline Generative Agents)
This repository contains a **complete, offline-first implementation of a Generative Agent** inspired by the paper:

**“Generative Agents: Interactive Simulacra of Human Behavior” (Park et al., UIST 2023)**
The project demonstrates how to build an **agentic AI system** with:
- Memory
- Reflection
- Planning

The agent is exposed through a **FastAPI backend** and a **simple web frontend**, running entirely **locally** using **Ollama**.

This project also serves as a **research exploration into agent cognition under constrained, safety-aligned**, and offline execution environments.

No cloud APIs. No API keys. Fully reproducible.

## Academic Credit
This work is inspired by:

Park, J. S., O’Brien, J., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023).  
*Generative Agents: Interactive Simulacra of Human Behavior.*  
Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST 2023).  
https://arxiv.org/abs/2304.03442

This repository is **not** the authors’ implementation and makes no claim of authorship.  
It is a **learning-focused, minimal implementation** intended for education and experimentation.

## Agent Architecture Overview
The agent follows the three core pillars described in the paper.

### 1. Memory Stream
- A time-ordered stream of observations
- Stores:
  - User messages
  - Agent replies
  - Reflections
- Persisted using SQLite (`data/memory.db`)

### 2. Reflection
- Triggered periodically after a configurable number of observations
- Summarizes recent experiences into higher-level beliefs or patterns
- Reflections are stored back into memory as special records

### 3. Planning
- For every user message:
  1. Relevant memories (and reflections) are retrieved
  2. Context is assembled
  3. The LLM decides what the agent should say next
- The agent’s reply is added back into memory

### Agent Loop (Simplified)
User message  
→ Retrieve memories + reflections  
→ Plan response (LLM)  
→ Reply to user  
→ Store new observation  

## Repository Structure
.
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── static/
│       └── index.html
├── data/
│   └── memory.db            # auto-created at runtime (gitignored)
├── src/
│   ├── agent.py
│   ├── hello_agent.py
│   ├── llm.py
│   ├── memory.py
│   └── reflection.py
├── tests/
│   ├── test_agent.py
│   └── test_reflection.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE


## Requirements

### System
- Python 3.10+
- Linux - Ubuntu 22.04.5 LTS
- Internet required only once to download the model

### Python Dependencies
From the repository root:

pip install -r requirements.txt

## Install Ollama (Local LLM Runtime)
Ollama runs the language model locally and exposes it over HTTP.

Download and install from:
https://ollama.com
Verify installation:
ollama --version

## Download a Model

Run once to download a model (example): ollama pull llama3.2 

## >> The model is stored locally.

Note: On most systems you do **not** need to run `ollama serve` explicitly.  
Ollama starts automatically when accessed.

## Run the Project (Offline)

### 1. Start the FastAPI Backend
From the repository root: uvicorn app.main:app --reload --port 8000

On startup, the application will:
- Ensure the Ollama model is available
- Warm the model
- Fail fast if Ollama is not running

### 2. Open the Web Frontend
Open a browser and navigate to:

http://localhost:8000/

You should see a simple chat interface.  
Type a message and the agent will reply.

## API Usage (Optional)

### Chat Endpoint
curl -X POST http://localhost:8000/chat

-H "Content-Type: application/json"
-d '{"message":"Hello! What do you remember about me?"}'

### History Endpoint
curl http://localhost:8000/history?n=10

### Health Check
curl http://localhost:8000/health

## Running Tests
Tests use an **in-memory SQLite database** for isolation.

From the repository root: pytest

## Offline-First Design
- No cloud APIs
- No API keys
- Local LLM execution via Ollama
- Persistent memory using SQLite
- Fully runnable without internet after model download

## Learning Goals
This project is designed to demonstrate:
- How agentic systems are structured
- How memory, reflection, and planning interact
- How to expose an AI agent via a web API
- How to build an offline-first AI application

## License
See the LICENSE file.

## Acknowledgements
- Park et al. (2023) for the Generative Agents concept
- Ollama for local LLM infrastructure
- FastAPI for the backend framework

