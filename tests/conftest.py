"""测试会话的全局准备。

## 为什么需要这个文件

部分测试会在构造 Agent 时创建 OpenAI 客户端（`src/llm/agent` → `LLMProvider`
→ `OpenAI(...)`），而 `openai>=2.x` 在 `api_key` 为空时会**直接抛**
`OpenAIError: Missing credentials`。同理 `app/server.py` 在**模块级**就
`orch = Orchestrator()`，任何 `from app import server` 的测试都会踩到。

开发机上因为有 `.env`，本地跑得通——但刚 clone 仓库、还没配 key 的人
（以及 CI）会看到一片失败。**那不是测试想验证的东西**：离线套件全部走
mock/fake，不发真实请求；真正联网的评测由 `integration` / `live` 标记
隔离，默认不跑（见 `pytest.ini`）。

## 设计：不改变开发机行为

先 `load_dotenv()` 把开发机上的 `.env` 读进来，再用 `setdefault` **只补
缺失项**。因此：

* 有 `.env`（开发机）：真实 key 生效，行为与以前完全一致；
* 无 `.env`（CI / 新克隆）：补占位值，测试得以运行。

占位值刻意写成一眼可辨的字符串，且指向 `.invalid` 域名——万一某条测试
真的发起了网络请求，它会立刻失败而不是悄悄打到真实服务。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# 先把开发机的 .env 灌进环境（若存在）。src/config.py 里也会调用一次，
# 重复调用无副作用。
load_dotenv()

# 测试不配置 API 令牌，显式打开本地放行——生产默认是失败关闭的。
os.environ.setdefault("LOCAL_OPEN_MODE", "1")
# 既有用例大量直接传裸 athlete_key；生产默认要求服务端令牌。
os.environ.setdefault("ALLOW_LEGACY_GUEST_KEY", "1")
# 离线套件不保证本机有 PostgreSQL/Redis，就绪探针只检查配置。
os.environ.setdefault("HEALTH_PROBE_DEPENDENCIES", "0")
# 测试不跑启动迁移：它需要真实数据库，且与用例验证的业务逻辑无关。
os.environ.setdefault("AUTO_MIGRATE", "0")

#: 一眼可辨的占位值；不是真实凭据。
_PLACEHOLDER_KEY = "test-placeholder-key-not-a-real-credential"
#: RFC 2606 保留域名，保证任何真实请求都会立即失败。
_PLACEHOLDER_URL = "https://placeholder.invalid/v1"

# 缺失时才补，不覆盖 .env 里的真实值。
for _var in (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_BASE_URL",
    "LLM_JUDGE_API_KEY",
    "LLM_JUDGE_BASE_URL",
    "EVAL_LLM_API_KEY",
    "EVAL_LLM_BASE_URL",
    # openai SDK 自身的兜底变量：即使代码里没显式传 api_key，
    # 只要它存在于环境里，客户端就能构造成功。
    "OPENAI_API_KEY",
):
    os.environ.setdefault(
        _var, _PLACEHOLDER_URL if _var.endswith("BASE_URL") else _PLACEHOLDER_KEY
    )

del _var


# --------------------------------------------------------------------------
# 每个测试重置限流窗口
# --------------------------------------------------------------------------
import pytest  # noqa: E402


@pytest.fixture
def manual_review(monkeypatch):
    """把审核模式切回 manual（送审 → 等人放行）。

    默认模式是 auto：安全类问题就地拒绝，不产生"等待审核"状态——这个项目
    没有审核人这个角色，扣下等人放行只会让用户永远拿不到计划。

    manual 仍然保留，用于开发期核对规则引擎有没有误报。用到这个 fixture
    的用例验证的正是那条备用通道：它必须还能跑通，但不参与默认流程。
    """
    monkeypatch.setattr("src.core.plan_finalization.MANUAL_REVIEW_ENABLED", True)
    monkeypatch.setattr("src.graph.routing.MANUAL_REVIEW_ENABLED", True)


@pytest.fixture(autouse=True)
def _isolated_audit_db(tmp_path, monkeypatch):
    """审计事件写临时库，不污染运行目录里的 data/audit.db。

    覆盖路径后必须把单例置空：``get_audit_store()`` 是进程级缓存，
    不清掉的话第一个用例的临时路径会被后面所有用例继续使用。
    """
    monkeypatch.setenv("AUDIT_DB_PATH", str(tmp_path / "audit.db"))
    from src.audit import store as audit_store

    audit_store._default_store = None
    yield
    audit_store._default_store = None


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """清空 RateLimitMiddleware 的滑动窗口。

    限流是 per-IP 的（默认 60 请求/分钟），而整套测试都从同一个 TestClient
    地址发请求。跑到文件末尾时窗口早已填满，于是 test_user_auth 里的
    claim 那组测试会拿到 429——那不是被测行为，是测试之间共享了状态。
    单独跑该文件能过、跑全量就挂，正是这类耦合的典型症状。
    """
    try:
        from app import server as _server
        from src.security.api_guard import RateLimitMiddleware

        node = getattr(_server.app, "middleware_stack", None)
        seen: set[int] = set()
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            if isinstance(node, RateLimitMiddleware):
                with node._lock:
                    node._hits.clear()
            node = getattr(node, "app", None)
    except Exception:  # noqa: BLE001 — 重置失败不该让测试本身报错
        pass
    yield


@pytest.fixture(autouse=True)
def _legacy_guest_key_window(monkeypatch):
    """测试沿用裸 athlete_key。

    生产默认已关闭这个窗口（游客身份必须由服务端签发）。既有用例大量
    直接传 ``athlete_key``，它们验证的是业务逻辑而不是身份来源；逐个
    换成真令牌既嘈杂又会掩盖原本的断言。覆盖身份来源的用例会自己把
    这个开关改回去。
    """
    from src.security import session as session_mod
    monkeypatch.setattr(session_mod, "ALLOW_LEGACY_GUEST_KEY", True)
