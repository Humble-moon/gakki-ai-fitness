import pytest
from src.storage.neo4j_client import Neo4jClient

@pytest.fixture
def neo4j():
    client = Neo4jClient()
    yield client
    client.run("MATCH (n:TestNeo4jNode) DETACH DELETE n")
    client.close()

class TestNeo4jClient:
    def test_connection_and_query(self, neo4j):
        neo4j.run("CREATE (:TestNeo4jNode {name: 'test_exercise'})")
        results = neo4j.query("MATCH (n:TestNeo4jNode) RETURN n.name AS name")
        assert len(results) > 0
        assert results[0]["name"] == "test_exercise"
