from src.storage.pg import PGClient


class KeywordSearch:
    def __init__(self):
        self.pg = PGClient()

    def search(self, query: str, top_k: int = 10) -> list:
        sql = """
            SELECT name, exercise_type, difficulty, equipment,
                   target_muscles, description, common_errors,
                   similarity(name, :query) AS sim
            FROM exercises
            WHERE name % :query OR name ILIKE '%' || :query || '%'
            ORDER BY sim DESC
            LIMIT :limit
        """
        rows = self.pg.fetch_all(sql, {"query": query, "limit": top_k})
        return [
            {"name": r[0], "type": r[1], "difficulty": r[2], "equipment": r[3],
             "target_muscles": r[4], "description": r[5], "common_errors": r[6],
             "similarity": float(r[7]) if r[7] else 0.0, "source": "keyword"}
            for r in rows
        ]
