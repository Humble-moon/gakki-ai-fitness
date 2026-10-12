"""请求入口的参数校验回归测试。

背景：app/server.py 里的 PlanRequest / AnalysisRequest / QuestionRequest 是从
src/models/schemas.py 复制出来、按扁平结构重写的第二个版本。复制时丢掉了
Field 约束，导致 days_per_week=0 / height=0 / 体重负数这类值穿透到
generate_plan 内部构造 UserProfileInput 时才抛 ValidationError——用户看到
的是 500 Internal Server Error，而不是一个可读的 422。

这组测试锁住「非法输入必须在入口被拒」这条线，防止将来再复制模型时重演。
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


@pytest.fixture()
def client():
    return TestClient(server.app, raise_server_exceptions=False)


# ---------------------------------------------------------------- PlanRequest
@pytest.mark.parametrize("payload,why", [
    ({"days_per_week": 0}, "每周 0 天"),
    ({"days_per_week": -5}, "每周负数天"),
    ({"days_per_week": 99}, "每周 99 天"),
    ({"height": 0}, "身高 0"),
    ({"height": 999}, "身高 999cm"),
    ({"weight": -10}, "体重负数"),
    ({"weight": 0}, "体重 0"),
    ({"training_years": -1}, "训练年限负数"),
    ({"height": "abc"}, "身高非数值"),
])
def test_plan_request_rejects_bad_numbers(client, payload, why):
    """非法数值必须在入口返回 422，而不是穿透到下游变 500。"""
    body = {"height": 175, "weight": 75}
    body.update(payload)
    r = client.post("/api/generate-plan", json=body)
    assert r.status_code == 422, f"{why} 应被拒，实际 {r.status_code}: {r.text[:200]}"


def test_plan_request_allows_zero_equipment():
    """零器械是合法场景（徒手训练），不能因为加了校验被误伤。

    这里直接校验模型而不发 HTTP 请求——参数合法时会真的走到计划生成，
    那要花一次真实 LLM 调用。校验层的边界用模型构造来断言就够了。
    """
    server.PlanRequest(height=175, weight=75, available_equipment=[],
                       days_per_week=3)


def test_plan_request_rejects_overlong_query(client):
    body = {"height": 175, "weight": 75, "query": "啊" * 3000}
    r = client.post("/api/generate-plan", json=body)
    assert r.status_code == 422, f"超长 query 应被拒，实际 {r.status_code}"


def test_plan_request_accepts_boundary_values():
    """边界值必须放行：1 天、7 天、100cm、250cm、30kg、200kg。"""
    server.PlanRequest(height=100, weight=30, days_per_week=1)
    server.PlanRequest(height=250, weight=200, days_per_week=7)


# ------------------------------------------------------------ AnalysisRequest
def test_analyze_exercise_requires_name(client):
    """没有动作名就无从分析——空值过去会跑到检索阶段白烧一次 LLM。"""
    r = client.post("/api/analyze-exercise", json={})
    assert r.status_code == 422, f"缺动作名应被拒，实际 {r.status_code}"

    r = client.post("/api/analyze-exercise", json={"exercise_name": ""})
    assert r.status_code == 422, f"空动作名应被拒，实际 {r.status_code}"


def test_analyze_exercise_rejects_bad_numbers(client):
    r = client.post("/api/analyze-exercise",
                    json={"exercise_name": "深蹲", "days_per_week": 0})
    assert r.status_code == 422


# ------------------------------------------------------------ QuestionRequest
def test_ask_question_requires_content(client):
    r = client.post("/api/ask-question", json={})
    assert r.status_code == 422, f"缺问题应被拒，实际 {r.status_code}"

    r = client.post("/api/ask-question", json={"question": ""})
    assert r.status_code == 422, f"空问题应被拒，实际 {r.status_code}"


def test_ask_question_rejects_overlong(client):
    r = client.post("/api/ask-question", json={"question": "啊" * 100000})
    assert r.status_code == 422, f"10 万字符问题应被拒，实际 {r.status_code}"


def test_ask_question_rejects_bad_numbers(client):
    r = client.post("/api/ask-question",
                    json={"question": "怎么练", "height": 0})
    assert r.status_code == 422


# ----------------------------------------------------------------------
# 请求体 → 用户档案的字段映射
#
# 这一层曾经是四个调用点各手写一份字段清单。age/sex 加进两个模型之后，
# 映射没跟上，于是用户在表单里填了、前端也传了、请求模型也收下了，
# 到了流水线里却是 None——热量目标悄悄退回粗估，界面上写着"缺身高或年龄"，
# 没有任何一个环节报错。这类静默丢字段只有契约测试拦得住。
# ----------------------------------------------------------------------

def test_profile_mapping_carries_every_shared_field():
    """两个模型的**交集字段**必须逐一映射过去，一个都不能漏。"""
    from src.models.schemas import UserProfileInput

    request_fields = set(server.PlanRequest.model_fields)
    profile_fields = set(UserProfileInput.model_fields)
    shared = request_fields & profile_fields

    # 除了有默认值的可选字段，公共字段都应实际出现在档案模型上
    assert shared, "两个模型没有交集字段，映射测试本身失效了"

    req = server.PlanRequest(
        height=180, weight=80, age=30, sex="男", training_years=1,
        goal="增肌", days_per_week=4, injuries=["左肩袖损伤"],
        habits=["晨练", "跑步"], available_equipment=["哑铃"],
    )
    profile = server._profile_from_request(req)

    for name in shared:
        assert getattr(profile, name) == getattr(req, name), (
            f"{name} 在请求体到档案的映射中丢失了——"
            "用户填了却不生效，且不会有任何报错"
        )


def test_age_and_sex_survive_the_api_boundary():
    """点名守卫：这两个字段就是上述静默丢失的当事人。"""
    req = server.PlanRequest(height=180, weight=80, age=30, sex="女",
                             training_years=1, goal="减脂", days_per_week=3)
    profile = server._profile_from_request(req)

    assert profile.age == 30
    assert profile.sex == "女"


def test_requests_without_age_or_sex_still_build_a_profile():
    """AnalysisRequest 没有 habits 字段，映射不能因此炸掉。"""
    req = server.AnalysisRequest(exercise_name="深蹲")
    profile = server._profile_from_request(req)

    assert profile.age is None
    assert profile.sex == "不填"
    assert profile.habits == []
