"""Checkpointer backend selection: memory / sqlite / postgres."""

from unittest.mock import patch

from src.graph import runtime


def test_memory_backend_for_offline_tests(monkeypatch, tmp_path):
    monkeypatch.setenv("COACH_CHECKPOINTER", "memory")
    saver = runtime._make_checkpointer(str(tmp_path / "x.db"))
    # langgraph >=1.0 renamed MemorySaver -> InMemorySaver; accept both
    assert type(saver).__name__ in ("MemorySaver", "InMemorySaver")


def test_sqlite_is_the_default(monkeypatch, tmp_path):
    monkeypatch.delenv("COACH_CHECKPOINTER", raising=False)
    monkeypatch.delenv("GRAPH_CHECKPOINT_BACKEND", raising=False)
    saver = runtime._make_checkpointer(str(tmp_path / "ckpt.db"))
    assert type(saver).__name__ == "SqliteSaver"
    assert (tmp_path / "ckpt.db").exists()


def test_postgres_backend_selected_by_env(monkeypatch):
    monkeypatch.delenv("COACH_CHECKPOINTER", raising=False)
    monkeypatch.setenv("GRAPH_CHECKPOINT_BACKEND", "postgres")
    sentinel = object()
    with patch.object(runtime, "_make_postgres_checkpointer", return_value=sentinel) as mk:
        assert runtime._make_checkpointer() is sentinel
    mk.assert_called_once()
