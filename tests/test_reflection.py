"""Phase 3: Unit tests for reflection (fixed behavior, no Ollama)."""

import sys
from pathlib import Path

# Project root on path so we can import src modules.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from memory import MemoryStream
from reflection import run_reflection, should_reflect


def test_should_reflect_after_n_observations():
    stream = MemoryStream(db_path=":memory:")
    assert not should_reflect(stream, every_n=3)
    stream.add_observation("User said: A")
    stream.add_observation("Agent said: B")
    assert not should_reflect(stream, every_n=3)
    stream.add_observation("User said: C")
    assert should_reflect(stream, every_n=3)


def test_should_reflect_resets_after_reflection():
    stream = MemoryStream(db_path=":memory:")
    stream.add_observation("A")
    stream.add_observation("B")
    stream.add_observation("C")
    assert should_reflect(stream, every_n=3)

    def mock_complete(prompt: str, system: str | None) -> str:
        return "User and agent are chatting. Three messages so far."

    run_reflection(stream, last_k=5, complete_fn=mock_complete)
    assert not should_reflect(stream, every_n=3)
    assert stream.count_since_last_reflection() == 0


def test_run_reflection_appends_reflection_record():
    stream = MemoryStream(db_path=":memory:")
    stream.add_observation("User said: I love hiking")
    stream.add_observation("Agent replied: Same here")

    def mock_complete(prompt: str, system: str | None) -> str:
        return "Both user and agent enjoy hiking."

    record = run_reflection(stream, last_k=5, complete_fn=mock_complete)
    assert record is not None
    assert record.get("type") == "reflection"
    assert "hiking" in record.get("content", "").lower()
    assert "timestamp" in record and "id" in record

    all_records = stream.get_all()
    assert len(all_records) == 3
    assert all_records[-1]["type"] == "reflection"


if __name__ == "__main__":
    test_should_reflect_after_n_observations()
    test_should_reflect_resets_after_reflection()
    test_run_reflection_appends_reflection_record()
    print("All 3 reflection tests passed.")
