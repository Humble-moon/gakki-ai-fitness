from sqlalchemy import text
from src.models.db_models import engine, SessionLocal

class PGClient:
    def __init__(self):
        self.engine = engine

    def execute(self, query: str, params: dict = None):
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            conn.commit()
            return result

    def fetch_all(self, query: str, params: dict = None):
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            return result.fetchall()

    def fetch_one(self, query: str, params: dict = None):
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            return result.fetchone()

    def get_session(self):
        return SessionLocal()

    def close(self):
        pass
