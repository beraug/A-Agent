"""Phase 4: Unit tests for the agent loop (no Ollama)."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from memory import MemoryStream
from agent import reply, _format_context


def test_format_context():
    records = [
        {"content": "User said: Hi", "type": "observation"},
        {"content": "Agent replied: Hello", "type": "observation"},
        {"content": "Both are greeting.", "type": "reflection"},
    ]
    text = _format_context(records)
    assert "User said: Hi" in text
    assert "Agent replied: Hello" in text
    assert "Both are greeting" in text
    assert "reflection" in text


def test_reply_stores_user_and_agent():
    stream = MemoryStream(db_path=":memory:")

    def mock_complete(prompt: str, system: str | None) -> str:
        return "I remember we just said hello!"

    out = reply(stream, "What did we just say?", complete_fn=mock_complete, reflect_every_n=0)
    assert "remember" in out.lower() or "hello" in out.lower()

    all_records = stream.get_all()
    assert len(all_records) >= 2
    assert any("User said:" in r.get("content", "") for r in all_records)
    assert any("Agent replied:" in r.get("content", "") for r in all_records)


def test_reply_uses_retrieved_context():
    stream = MemoryStream(db_path=":memory:")
    stream.add_observation("User said: My name is Alex")
    stream.add_observation("Agent replied: Nice to meet you, Alex")

    def mock_complete(prompt: str, system: str | None) -> str:
        assert "Alex" in prompt
        return "Hello again, Alex!"

    out = reply(stream, "Do you remember my name?", complete_fn=mock_complete, reflect_every_n=0)
    assert "Alex" in out


if __name__ == "__main__":
    test_format_context()
    test_reply_stores_user_and_agent()
    test_reply_uses_retrieved_context()
    print("All 3 agent tests passed.")
