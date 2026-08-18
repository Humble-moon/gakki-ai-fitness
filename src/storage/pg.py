"""pg.py - PostgreSQL 数据库操作客户端"""

from contextlib import contextmanager

from sqlalchemy import text
from src.models.db_models import engine, SessionLocal


class _Transaction:
    """Small transaction-bound query facade used by storage code and tests."""

    def __init__(self, connection):
        self.connection = connection

    def execute(self, query: str, params: dict = None):
        return self.connection.execute(text(query), params or {})

    def fetch_all(self, query: str, params: dict = None):
        return self.execute(query, params).fetchall()

    def fetch_one(self, query: str, params: dict = None):
        return self.execute(query, params).fetchone()


class PGClient:
    """PostgreSQL 客户端封装类。"""

    def __init__(self):
        self.engine = engine

    def execute(self, query: str, params: dict = None):
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            conn.commit()
            return result

    def fetch_all(self, query: str, params: dict = None):
        with self.engine.connect() as conn:
            return conn.execute(text(query), params or {}).fetchall()

    def fetch_one(self, query: str, params: dict = None):
        with self.engine.connect() as conn:
            return conn.execute(text(query), params or {}).fetchone()

    @contextmanager
    def transaction(self):
        """Run multiple operations on one connection and commit exactly once."""
        with self.engine.connect() as conn:
            transaction = conn.begin()
            tx = _Transaction(conn)
            try:
                yield tx
            except Exception:
                transaction.rollback()
                raise
            else:
                transaction.commit()

    def get_session(self):
        return SessionLocal()

    def close(self):
        pass
