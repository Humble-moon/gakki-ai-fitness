from src.storage.neo4j_client import Neo4jClient


class GraphSearch:
    def __init__(self):
        self.neo4j = Neo4jClient()

    def find_exercises_by_muscle(self, muscle: str, limit: int = 10) -> list:
        results = self.neo4j.query(
            """
            MATCH (e:Exercise)-[:TARGETS]->(m:Muscle)
            WHERE m.name CONTAINS $muscle
            RETURN e.name AS name, e.difficulty AS difficulty,
                   e.type AS type, m.name AS muscle
            LIMIT $limit
        """,
            {"muscle": muscle, "limit": limit},
        )
        return results

    def find_equipment_for_exercise(self, exercise: str) -> list:
        results = self.neo4j.query(
            """
            MATCH (e:Exercise {name: $name})-[:REQUIRES]->(eq:Equipment)
            RETURN eq.name AS equipment
        """,
            {"name": exercise},
        )
        return results

    def multi_hop_search(self, equipment: str, target: str) -> list:
        results = self.neo4j.query(
            """
            MATCH (e:Exercise)-[:REQUIRES]->(eq:Equipment)
            WHERE eq.name CONTAINS $equipment
            MATCH (e)-[:TARGETS]->(m:Muscle)
            WHERE m.name CONTAINS $target
            RETURN e.name AS name, e.difficulty AS difficulty,
                   collect(DISTINCT m.name) AS muscles,
                   collect(DISTINCT eq.name) AS equipment
        """,
            {"equipment": equipment, "target": target},
        )
        return results

    def find_injury_risks(self, exercise: str) -> list:
        results = self.neo4j.query(
            """
            MATCH (e:Exercise {name: $name})-[:MAY_CAUSE]->(i:Injury)
            RETURN i.name AS injury
        """,
            {"name": exercise},
        )
        return results

    def find_rehab_exercises(self, injury: str) -> list:
        results = self.neo4j.query(
            """
            MATCH (i:Injury {name: $injury})<-[:MAY_CAUSE]-(bad:Exercise)
            MATCH (i)-[:RECOVERED_BY]->(rehab:Exercise)
            RETURN rehab.name AS rehab_exercise,
                   collect(DISTINCT bad.name) AS avoid_exercises
        """,
            {"injury": injury},
        )
        return results

    def reason_about_pain(self, exercise: str, symptom: str) -> dict:
        risks = self.find_injury_risks(exercise)
        causes = []
        solutions = []
        for r in risks:
            rehab = self.find_rehab_exercises(r["injury"])
            for item in rehab:
                causes.append(
                    {
                        "injury": r["injury"],
                        "avoid": item.get("avoid_exercises", []),
                    }
                )
                rehab_ex = item.get("rehab_exercise")
                if rehab_ex:
                    solutions.append(rehab_ex)
        return {
            "exercise": exercise,
            "symptom": symptom,
            "possible_causes": causes,
            "suggested_rehab": list(set(solutions)),
            "source": "graph",
        }
