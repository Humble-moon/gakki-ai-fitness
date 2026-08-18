from types import SimpleNamespace

import pytest

from src.storage.document_store import DocumentStore


class FakeTransaction:
    def __init__(self, db):
        self.db = db
        self.snapshot = (list(db.documents), list(db.chunks))

    def fetch_one(self, query, params=None):
        if "COUNT(*)" in query:
            return SimpleNamespace(cnt=len(self.db.documents))
        if "RETURNING id" in query:
            doc = {"id": self.db.next_id, **(params or {})}
            self.db.next_id += 1
            self.db.documents.append(doc)
            return SimpleNamespace(id=doc["id"])
        raise AssertionError(query)

    def fetch_all(self, query, params=None):
        if "SELECT id FROM user_documents" in query:
            n = (params or {}).get("n", 0)
            return [SimpleNamespace(id=d["id"]) for d in self.db.documents[:n]]
        raise AssertionError(query)

    def execute(self, query, params=None):
        params = params or {}
        if self.db.fail_cleanup and query.lstrip().startswith("DELETE"):
            raise RuntimeError("quota cleanup failed")
        if "document_chunks" in query and query.lstrip().startswith("INSERT"):
            if self.db.fail_chunk_insert:
                raise RuntimeError("chunk insert failed")
            self.db.chunks.append({"document_id": params["did"], "content": params["ct"]})
        elif "document_chunks" in query:
            self.db.chunks[:] = [c for c in self.db.chunks if c["document_id"] != params["did"]]
        elif "user_documents" in query:
            self.db.documents[:] = [d for d in self.db.documents if d["id"] != params["did"]]
        else:
            raise AssertionError(query)


class FakeDB:
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.next_id = 1
        self.transaction_state = None
        self.transaction_begin_count = 0
        self.fail_chunk_insert = False
        self.fail_cleanup = False

    class _Context:
        def __init__(self, db):
            self.db = db

        def __enter__(self):
            self.db.transaction_begin_count += 1
            self.tx = FakeTransaction(self.db)
            return self.tx

        def __exit__(self, exc_type, exc, tb):
            if exc_type:
                self.db.documents[:] = self.tx.snapshot[0]
                self.db.chunks[:] = self.tx.snapshot[1]
                self.db.transaction_state = "rolled_back"
            else:
                self.db.transaction_state = "committed"
            return False

    def transaction(self):
        return self._Context(self)


@pytest.fixture
def store(monkeypatch):
    db = FakeDB()
    store = DocumentStore()
    store.db = db
    monkeypatch.setattr("src.storage.document_store.chunk_text", lambda text: text.split("\n\n"))
    store.embedder.embed = lambda text: [float(len(text))]
    return store


def test_embedding_failure_rolls_back_document_and_chunks(store):
    calls = {"count": 0}

    def fail_on_second_chunk(_text):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("embedding failed")
        return [1.0]

    store.embedder.embed = fail_on_second_chunk
    with pytest.raises(RuntimeError, match="embedding failed"):
        store.save("s", "x.txt", "txt", 10, "chunk one\n\nchunk two", 1, "x", True, "")
    assert store.db.transaction_state == "rolled_back"
    assert store.db.documents == []
    assert store.db.chunks == []


def test_chunk_insert_failure_rolls_back(store):
    store.db.fail_chunk_insert = True
    with pytest.raises(RuntimeError, match="chunk insert failed"):
        store.save("s", "x.txt", "txt", 10, "safe text", 1, "x", True, "")
    assert store.db.transaction_state == "rolled_back"
    assert store.db.documents == []
    assert store.db.chunks == []


def test_success_commits_once(store):
    doc_id = store.save("s", "x.txt", "txt", 10, "safe text", 1, "x", True, "")
    assert doc_id == 1
    assert store.db.transaction_state == "committed"
    assert store.db.transaction_begin_count == 1
    assert len(store.db.documents) == 1
    assert len(store.db.chunks) == 1


def test_quota_cleanup_is_inside_same_transaction(store):
    store.db.documents = [{"id": i} for i in range(1, 6)]
    store.db.next_id = 6
    store.save("s", "new.txt", "txt", 10, "safe text", 1, "new", True, "")
    assert store.db.transaction_begin_count == 1
    assert store.db.transaction_state == "committed"
    assert [doc["id"] for doc in store.db.documents] == [2, 3, 4, 5, 6]


def test_quota_cleanup_failure_rolls_back(store):
    store.db.documents = [{"id": i} for i in range(1, 6)]
    store.db.next_id = 6
    store.db.fail_cleanup = True
    with pytest.raises(RuntimeError, match="quota cleanup failed"):
        store.save("s", "new.txt", "txt", 10, "safe text", 1, "new", True, "")
    assert store.db.transaction_state == "rolled_back"
    assert [doc["id"] for doc in store.db.documents] == [1, 2, 3, 4, 5]
    assert store.db.chunks == []


def test_transaction_context_rolls_back_and_commits():
    db = FakeDB()
    with pytest.raises(ValueError):
        with db.transaction():
            raise ValueError("boom")
    assert db.transaction_state == "rolled_back"
    with db.transaction():
        pass
    assert db.transaction_state == "committed"
    assert db.transaction_begin_count == 2
