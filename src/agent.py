"""
Phase 4: Planning (paper pillar 3).

Agent loop: user message → add to memory → (optionally reflect) → retrieve
context → plan with LLM → reply → store reply in memory. Ties memory to action.

Usage:
    from memory import MemoryStream
    from agent import reply

    stream = MemoryStream()
    response = reply(stream, "What did we talk about last time?")
"""

from typing import TYPE_CHECKING, Callable

from llm import OllamaError, complete

from reflection import run_reflection, should_reflect

if TYPE_CHECKING:
    from memory import MemoryStream
from memory import MemoryStream

# Single global agent instance (no sessions)
_STREAM = MemoryStream()


def get_memory_stream() -> MemoryStream:
    return _STREAM


def agent_chat(message: str) -> str:
    # Phase 5 contract: callable agent_chat(message) -> reply
    return reply(_STREAM, message)


AGENT_SYSTEM = (
    "You are a generative agent: you remember past interactions and use that "
    "context to respond naturally. Given recent observations and the current "
    "user message, reply as the agent in 1–3 short sentences. Be concise and "
    "conversational. Do not repeat the context back; just respond."
)

PROMPT_TEMPLATE = """Recent context (observations and reflections):

{context}

Current user message: {message}

Respond as the agent."""


def _format_context(records: list[dict]) -> str:
    """Format retrieved memory records for the planning prompt."""
    lines = []
    for r in records:
        content = r.get("content", "")
        kind = r.get("type", "observation")
        lines.append(f"- [{kind}] {content}")
    return "\n".join(lines) if lines else "(No prior context.)"


def reply(
    memory_stream: "MemoryStream",
    user_message: str,
    retrieve_k: int = 10,
    reflect_every_n: int = 5,
    complete_fn: Callable[[str, str | None], str] | None = None,
) -> str:
    """
    Run the agent loop: store user message, optionally reflect, retrieve context,
    plan (LLM), store and return the reply.

    Args:
        memory_stream: The memory stream (add_observation, retrieve).
        user_message: The current user message.
        retrieve_k: Number of memories to pass to the LLM as context.
        reflect_every_n: Run reflection when this many observations exist since last reflection (0 = skip).
        complete_fn: Optional (prompt, system) -> text; used instead of llm.complete for tests.

    Returns:
        The agent's reply text.

    Raises:
        OllamaError: If the LLM call fails (Ollama not running, etc.).
    """
    # Store user message
    memory_stream.add_observation(f"User said: {user_message.strip()}")

    # Optionally run reflection
    if reflect_every_n > 0 and should_reflect(memory_stream, every_n=reflect_every_n):
        run_reflection(memory_stream, last_k=retrieve_k, complete_fn=complete_fn)

    # Retrieve context (memories + reflections)
    context_records = memory_stream.retrieve(user_message, k=retrieve_k)
    context_text = _format_context(context_records)

    # Plan: build prompt and call LLM
    prompt = PROMPT_TEMPLATE.format(context=context_text, message=user_message.strip())

    if complete_fn is not None:
        reply_text = complete_fn(prompt, AGENT_SYSTEM)
    else:
        reply_text = complete(prompt, system=AGENT_SYSTEM)

    reply_text = reply_text.strip()
    if not reply_text:
        reply_text = "(No reply generated.)"

    # Store agent reply so future retrieval sees it
    memory_stream.add_observation(f"Agent replied: {reply_text}")

    return reply_text


def main() -> None:
    """Demo: one turn with Ollama (or mock). Run from project root: python src/agent.py."""
    import sys
    from pathlib import Path
    if Path(__file__).resolve().parent not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))

    from memory import MemoryStream

    stream = MemoryStream()
    message = sys.argv[1] if len(sys.argv) > 1 else "Hello, who are you?"
    print("User:", message)
    try:
        response = reply(stream, message, reflect_every_n=0)
        print("Agent:", response)
    except OllamaError as e:
        print("Agent unavailable:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
