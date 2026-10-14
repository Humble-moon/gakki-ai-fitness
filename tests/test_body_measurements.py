"""体测记录接口。

体测记录是个人健康数据，因此这批用例重点钉三件事：
  1. 身份缺失时不得写入，也不得读到别人的数据
  2. 全空提交不落库——那不是一次体测，是误触
  3. 删除限定在本人名下，自增 id 可枚举，不能成为越权通道
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class _FakeBodyMeasureStore:
    """体测记录存储替身：记录调用，供断言归属与写入行为。"""

    def __init__(self):
        self.enabled = True
        self.rows: list[dict] = []
        self._next_id = 1

    def ensure_table(self):
        pass

    def save(self, athlete_key, measure_date=None, **fields):
        row = {"id": self._next_id, "athlete_key": athlete_key, "fields": fields}
        self._next_id += 1
        self.rows.append(row)
        return row["id"]

    def get_measurements(self, athlete_key, weeks=None, limit=200):
        return [r for r in self.rows if r["athlete_key"] == athlete_key]

    def delete_one(self, athlete_key, measurement_id):
        before = len(self.rows)
        self.rows = [
            r for r in self.rows
            if not (r["id"] == measurement_id and r["athlete_key"] == athlete_key)
        ]
        return len(self.rows) < before


@pytest.fixture
def rig(monkeypatch):
    from app import server

    store = _FakeBodyMeasureStore()
    monkeypatch.setattr(server, "_get_body_measure_store", lambda: store)
    return TestClient(server.app), store


class TestCreate:
    def test_requires_identity(self, rig):
        """没有身份就不该写入——否则这条记录不属于任何人。"""
        client, store = rig
        resp = client.post("/api/body-measurements", json={"weight": 70})
        assert resp.status_code == 400
        assert store.rows == []

    def test_rejects_all_empty_payload(self, rig):
        """全空提交是一次误触，不是体测。落库只会在趋势图上留个空洞。"""
        client, store = rig
        resp = client.post("/api/body-measurements", json={"athlete_key": "a-1"})
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "NO_MEASUREMENT"
        assert store.rows == []

    def test_saves_single_field(self, rig):
        """只量了体重也该能记——现实中很少有人每次把八项量齐。"""
        client, store = rig
        resp = client.post("/api/body-measurements",
                           json={"athlete_key": "a-1", "weight": 70.5})
        assert resp.status_code == 200
        assert len(store.rows) == 1
        assert store.rows[0]["fields"]["weight"] == 70.5

    def test_rejects_out_of_range(self, rig):
        client, store = rig
        resp = client.post("/api/body-measurements",
                           json={"athlete_key": "a-1", "weight": 5})
        assert resp.status_code == 422
        assert store.rows == []


class TestRead:
    def test_lists_only_own_records(self, rig):
        """按身份隔离：A 的记录不出现在 B 的列表里。"""
        client, store = rig
        client.post("/api/body-measurements",
                    json={"athlete_key": "a-1", "weight": 70})
        client.post("/api/body-measurements",
                    json={"athlete_key": "b-2", "weight": 80})

        a = client.get("/api/body-measurements?athlete_key=a-1").json()
        b = client.get("/api/body-measurements?athlete_key=b-2").json()
        assert a["count"] == 1 and b["count"] == 1
        assert a["measurements"][0]["athlete_key"] == "a-1"
        assert b["measurements"][0]["athlete_key"] == "b-2"

    def test_requires_identity(self, rig):
        client, _ = rig
        resp = client.get("/api/body-measurements")
        assert resp.status_code == 400


class TestDelete:
    def test_deletes_own_record(self, rig):
        client, store = rig
        client.post("/api/body-measurements",
                    json={"athlete_key": "a-1", "weight": 70})
        resp = client.delete("/api/body-measurements/1?athlete_key=a-1")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        assert store.rows == []

    def test_cannot_delete_others_record(self, rig):
        """自增 id 可枚举，删除必须带上归属过滤，否则越权删别人的数据。

        返回 deleted=false 而不是 404：不暴露"这个 id 存在但不属于你"。
        """
        client, store = rig
        client.post("/api/body-measurements",
                    json={"athlete_key": "a-1", "weight": 70})
        resp = client.delete("/api/body-measurements/1?athlete_key=b-2")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is False
        assert len(store.rows) == 1, "别人的记录被删掉了"
