"""训练计划重命名的接口契约。

名字存在 ``plan_data.plan_name`` 里——表里没有独立的 title 列，列表展示名
正是从这个字段派生的。所以"改名"改的是正文里的一个字段，本组测试锁住
这条链路的三条边界：

* 校验在路由层：非法名字必须得到 400 与具体原因，而不是被存储层翻成
  "计划不存在"；
* 归属不匹配与不存在都归 404：不泄露"这个 id 存在但不是你的"；
* 校验没过就不该碰存储层。

存储层被替成假的：这组测试验证的是路由分支，不需要 PostgreSQL。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import server  # noqa: E402

KEY = "deebb161-7d2f-4155-8eb6-35dfe76e7918"


class _FakeStore:
    """记录调用、按需返回命中与否。"""

    enabled = True

    def __init__(self, found: bool = True):
        self.found = found
        self.calls: list[tuple] = []

    def ensure_table(self) -> None:
        pass

    def rename_plan(self, owner, plan_id, title):
        self.calls.append((owner, plan_id, title))
        return self.found


@pytest.fixture()
def client():
    return TestClient(server.app, raise_server_exceptions=False)


@pytest.fixture()
def fake(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(server, "_get_plan_store", lambda: store)
    return store


def test_rename_returns_new_title(client, fake):
    r = client.patch("/api/plans/7", json={"title": "周四下肢", "athlete_key": KEY})
    assert r.status_code == 200, r.text
    assert r.json() == {"renamed": True, "plan_id": 7, "title": "周四下肢"}
    assert fake.calls and fake.calls[0][1] == 7


def test_rename_trims_surrounding_whitespace(client, fake):
    """前后空白不该被当成名字的一部分存进去。"""
    r = client.patch("/api/plans/7", json={"title": "  周四下肢  ", "athlete_key": KEY})
    assert r.status_code == 200
    assert r.json()["title"] == "周四下肢"
    assert fake.calls[0][2] == "周四下肢"


@pytest.mark.parametrize("title,why", [("", "空字符串"), ("   ", "纯空白")])
def test_rename_rejects_blank(client, fake, title, why):
    r = client.patch("/api/plans/7", json={"title": title, "athlete_key": KEY})
    assert r.status_code == 400, f"{why}应被拒，实际 {r.status_code}"
    assert r.json()["error"]["code"] == "PLAN_TITLE_EMPTY"
    assert fake.calls == [], "校验没过就不该碰存储层"


def test_rename_rejects_too_long(client, fake):
    r = client.patch("/api/plans/7", json={"title": "字" * 41, "athlete_key": KEY})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "PLAN_TITLE_TOO_LONG"
    assert fake.calls == []


def test_rename_accepts_exactly_max_length(client, fake):
    """边界值本身是合法的——只有超过才拒。"""
    title = "字" * server.MAX_PLAN_TITLE_LEN
    r = client.patch("/api/plans/7", json={"title": title, "athlete_key": KEY})
    assert r.status_code == 200


def test_rename_missing_plan_is_404(client, monkeypatch):
    monkeypatch.setattr(server, "_get_plan_store", lambda: _FakeStore(found=False))
    r = client.patch("/api/plans/999", json={"title": "随便", "athlete_key": KEY})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "PLAN_NOT_FOUND"


def test_rename_requires_valid_athlete_key(client, fake):
    """没有身份标识时不该落到存储层——否则等于任何人都能改任何计划。"""
    r = client.patch("/api/plans/7", json={"title": "偷偷改"})
    assert r.status_code == 400
    assert fake.calls == []
