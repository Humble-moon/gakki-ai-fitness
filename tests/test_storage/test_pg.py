import pytest
from src.storage.pg import PGClient

pytestmark = pytest.mark.integration

@pytest.fixture
def pg():
    client = PGClient()
    yield client
    # 下面的用例会往真实库（不是替身）插入示例动作，收尾必须清掉。
    # 此前只在用例开头清理上次的残留、跑完不清理，于是每执行一次就多一条
    # "测试动作"，库里的动作数悄悄超过 data/seed_exercises.json 的条数。
    try:
        client.execute("DELETE FROM exercises WHERE name = '测试动作'")
    finally:
        client.close()

class TestPGClient:
    def test_connection(self, pg):
        assert pg.engine is not None

    def test_insert_and_search(self, pg):
        from src.models.db_models import init_db
        init_db()
        # 仍保留开头这次清理：上一次若因断言失败等异常中途退出，
        # fixture 的收尾可能没跑到，残留会挡住这次的 INSERT。
        pg.execute("DELETE FROM exercises WHERE name = '测试动作'")
        result = pg.execute(
            "INSERT INTO exercises (name, exercise_type, difficulty, equipment, target_muscles) "
            "VALUES ('测试动作', '复合', '初级', '哑铃', '[\"胸大肌\"]')"
        )
        assert result is not None
        rows = pg.fetch_all("SELECT * FROM exercises WHERE name = '测试动作'")
        assert len(rows) == 1
        assert rows[0][1] == '测试动作'
