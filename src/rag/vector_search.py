from src.storage.pg import PGClient
from src.rag.embedding import EmbeddingService


class VectorSearch:
    def __init__(self):
        self.pg = PGClient()
        self.emb = EmbeddingService()

    def search(self, query: str, top_k: int = 10, filters: dict = None) -> list:
        vec = self.emb.embed(query)
        vec_str = f"[{','.join(str(v) for v in vec)}]"
        filter_clause = ""
        if filters:
            if "equipment" in filters:
                filter_clause = f"AND equipment = '{filters['equipment']}'"
        sql = f"""
            SELECT name, name_en, exercise_type, difficulty, equipment,
                   target_muscles, description, common_errors,
                   1 - (embedding <=> '{vec_str}'::vector) AS similarity
            FROM exercises
            WHERE embedding IS NOT NULL {filter_clause}
            ORDER BY embedding <=> '{vec_str}'::vector
            LIMIT {top_k}
        """
        rows = self.pg.fetch_all(sql)
        return [
            {"name": r[0], "name_en": r[1], "type": r[2], "difficulty": r[3],
             "equipment": r[4], "target_muscles": r[5], "description": r[6],
             "common_errors": r[7], "similarity": float(r[8]),
             "source": "vector"}
            for r in rows
        ]
