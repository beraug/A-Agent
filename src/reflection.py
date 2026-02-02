"""
Phase 3: Reflection (paper pillar 2).

Periodically synthesize recent memories into higher-level reflections
(beliefs, patterns). Uses the LLM to summarize; stores result in the
same memory stream with type "reflection" so retrieval sees it.

Usage:
    from memory import MemoryStream
    from reflection import run_reflection, should_reflect

    stream = MemoryStream()
    stream.add_observation("User said: I love hiking")
    stream.add_observation("Agent replied: Me too, we should go sometime")
    if should_reflect(stream, every_n=2):
        run_reflection(stream, last_k=5)
"""

from typing import TYPE_CHECKING, Callable

from llm import OllamaError, complete

if TYPE_CHECKING:
    from memory import MemoryStream


REFLECTION_SYSTEM = (
    "You summarize patterns, beliefs, and important facts from a stream of "
    "observations about an agent and its user. Output 1–3 short sentences. "
    "Be concrete and specific. Do not add preamble or meta-commentary."
)

REFLECTION_PROMPT_TEMPLATE = """Here are recent observations (oldest first):

{observations}

Summarize patterns, beliefs, or important facts about this agent and the user."""


def _format_observations(records: list[dict]) -> str:
    """Turn memory records into a single text block for the prompt."""
    lines = []
    for r in records:
        ts = r.get("timestamp", "")[:19]  # trim to datetime part
        content = r.get("content", "")
        kind = r.get("type", "observation")
        lines.append(f"- [{ts}] ({kind}) {content}")
    return "\n".join(lines) if lines else "(No observations yet.)"


def run_reflection(
    memory_stream: "MemoryStream",
    last_k: int = 10,
    importance: float = 0.8,
    complete_fn: Callable[[str, str | None], str] | None = None,
) -> dict | None:
    """
    Run one reflection step: get last K memories, ask the LLM to summarize
    patterns/beliefs, append the result as a reflection to the stream.

    Args:
        memory_stream: The memory stream (must have get_recent and add_observation).
        last_k: How many recent records to feed into the reflection prompt.
        importance: Importance score for the new reflection record (0–1).
        complete_fn: Optional (prompt, system) -> text; used instead of llm.complete for tests.

    Returns:
        The new reflection record, or None if the LLM call failed.
    """
    recent = memory_stream.get_recent(k=last_k)
    if not recent:
        return None

    observations_text = _format_observations(recent)
    prompt = REFLECTION_PROMPT_TEMPLATE.format(observations=observations_text)

    if complete_fn is not None:
        summary = complete_fn(prompt, REFLECTION_SYSTEM)
    else:
        try:
            summary = complete(prompt, system=REFLECTION_SYSTEM)
        except OllamaError:
            return None

    summary = summary.strip()
    if not summary:
        return None

    record = memory_stream.add_observation(
        content=summary,
        importance=importance,
        type_="reflection",
    )
    return record


def should_reflect(memory_stream: "MemoryStream", every_n: int = 5) -> bool:
    """
    Return True if the agent should run a reflection step: there are at least
    every_n observations (non-reflection) since the last reflection.

    The agent loop can call this after each user message and then call
    run_reflection(stream, ...) when it returns True.
    """
    return memory_stream.count_since_last_reflection() >= every_n
