from src.storage.neo4j_client import Neo4jClient
from src.llm.provider import LLMProvider


class GraphBuilder:
    def __init__(self):
        self.neo4j = Neo4jClient()
        self.llm = LLMProvider()

    def init_schema(self):
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Muscle) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Exercise) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (eq:Equipment) REQUIRE eq.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Injury) REQUIRE i.name IS UNIQUE",
        ]
        for c in constraints:
            try:
                self.neo4j.run(c)
            except Exception:
                pass

    def build_from_seed(self, exercises: list):
        self.init_schema()
        for ex in exercises:
            self.neo4j.run(
                """
                MERGE (e:Exercise {name: $name})
                SET e.difficulty = $difficulty, e.type = $type
            """,
                {
                    "name": ex["name"],
                    "difficulty": ex.get("difficulty", "\u4e2d\u7ea7"),
                    "type": ex.get("exercise_type", "\u590d\u5408"),
                },
            )

            for muscle in ex.get("target_muscles", []):
                self.neo4j.run(
                    """
                    MERGE (m:Muscle {name: $muscle})
                    MERGE (e:Exercise {name: $ex_name})
                    MERGE (e)-[:TARGETS]->(m)
                """,
                    {"muscle": muscle, "ex_name": ex["name"]},
                )

            if ex.get("equipment"):
                self.neo4j.run(
                    """
                    MERGE (eq:Equipment {name: $equip})
                    MERGE (e:Exercise {name: $ex_name})
                    MERGE (e)-[:REQUIRES]->(eq)
                """,
                    {"equip": ex["equipment"], "ex_name": ex["name"]},
                )

    def extract_triples_with_llm(self, text: str) -> list:
        prompt = (
            f"从以下健身文本中提取（动作-关系-实体）三元组。\n"
            f"实体类型：Muscle, Exercise, Equipment, Injury\n"
            f"关系类型：TARGETS, REQUIRES, MAY_CAUSE, RECOVERED_BY\n\n"
            f"文本：{text}\n\n"
            f'输出 JSON 数组：[{{"subject":"","relation":"","object":"","obj_type":""}}]'
        )
        result = self.llm.chat_with_json_mode(
            [{"role": "user", "content": prompt}]
        )
        return result if isinstance(result, list) else []
