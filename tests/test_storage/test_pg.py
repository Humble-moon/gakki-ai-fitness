import pytest
from src.storage.pg import PGClient

@pytest.fixture
def pg():
    client = PGClient()
    yield client
    client.close()

class TestPGClient:
    def test_connection(self, pg):
        assert pg.engine is not None

    def test_insert_and_search(self, pg):
        from src.models.db_models import init_db
        init_db()
        pg.execute("DELETE FROM exercises WHERE name = '测试动作'")
        result = pg.execute(
            "INSERT INTO exercises (name, exercise_type, difficulty, equipment, target_muscles) "
            "VALUES ('测试动作', '复合', '初级', '哑铃', '[\"胸大肌\"]')"
        )
        assert result is not None
        rows = pg.fetch_all("SELECT * FROM exercises WHERE name = '测试动作'")
        assert len(rows) == 1
        assert rows[0][1] == '测试动作'
