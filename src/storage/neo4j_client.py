from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def run(self, query: str, params: dict = None):
        with self.driver.session() as session:
            return session.run(query, params or {})

    def query(self, query: str, params: dict = None):
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def close(self):
        self.driver.close()
