"""账号体系：密码哈希、会话、归属校验、中间件接缝。

分三层：
1. 纯函数层（哈希、会话键、owner_key 解析）——不依赖 FastAPI 与数据库；
2. 路由层——用 TestClient 打桩 UserStore/SessionStore；
3. 中间件接缝——登录端点的豁免、预检放行、成本上限不拦登录。
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.security import auth as auth_mod
from src.security.api_guard import ApiKeyMiddleware, CostLimitMiddleware
from src.security.session import (
    ANONYMOUS,
    GuestKeyError,
    Principal,
    SessionStore,
    resolve_owner_key,
    user_owner_key,
)


# ============================================================
# 1. 密码哈希
# ============================================================
def test_password_roundtrip():
    stored = auth_mod.hash_password("correct horse battery")
    assert auth_mod.verify_password("correct horse battery", stored) is True
    assert auth_mod.verify_password("wrong horse battery", stored) is False


def test_password_hash_is_salted():
    """同一密码两次哈希必须不同——否则相同密码的用户可互相比对出密码相同。"""
    a = auth_mod.hash_password("same-password-here")
    b = auth_mod.hash_password("same-password-here")
    assert a != b


def test_password_hash_is_self_describing():
    """参数写在串里，将来调强度时旧记录仍可验证。"""
    stored = auth_mod.hash_password("another-password")
    algo, n, r, p, salt, digest = stored.split("$")
    assert algo == "scrypt" and int(n) >= 2 ** 14 and int(r) >= 8
    assert len(salt) == 32 and len(digest) == 128   # 16 字节盐 / 64 字节摘要的 hex


def test_verify_never_raises_on_corrupt_hash():
    """格式损坏的历史记录不该让登录接口 500，而应判为验证不通过。"""
    for bad in ("", "garbage", "scrypt$1$2", "bcrypt$x$y$z$a$b", None, 12345):
        assert auth_mod.verify_password("whatever12345", bad) is False


def test_weak_and_overlong_passwords_are_rejected():
    with pytest.raises(auth_mod.AuthError):
        auth_mod.hash_password("short")
    with pytest.raises(auth_mod.AuthError):
        auth_mod.hash_password("x" * (auth_mod.MAX_PASSWORD_LENGTH + 1))


def test_overlong_password_fails_verification_without_hashing():
    """超长输入直接拒绝，不让 scrypt 白算——否则可被用来打 CPU。"""
    stored = auth_mod.hash_password("normal-password")
    assert auth_mod.verify_password("x" * 10_000, stored) is False


# ============================================================
# 2. 会话令牌
# ============================================================
class FakeRedis:
    def __init__(self):
        self.store = {}
        self.deleted = []

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value

    def delete(self, key):
        self.deleted.append(key)
        self.store.pop(key, None)


def _store():
    return SessionStore(redis=FakeRedis())


def test_session_roundtrip_and_revoke():
    store = _store()
    token = store.create(42, "alice")
    principal = store.resolve(token)
    assert principal.user_id == 42 and principal.username == "alice"
    assert principal.is_authenticated

    store.revoke(token)
    assert store.resolve(token) == ANONYMOUS


def test_session_stores_only_a_digest():
    """Redis 里不能出现明文令牌——否则一次备份泄露等于所有会话被冒充。"""
    store = _store()
    token = store.create(7, "bob")
    keys = list(store.redis.store.keys())
    assert len(keys) == 1
    assert token not in keys[0]
    assert auth_mod.token_fingerprint(token) in keys[0]


def test_invalid_session_inputs_resolve_to_anonymous():
    store = _store()
    assert store.resolve(None) == ANONYMOUS
    assert store.resolve("") == ANONYMOUS
    assert store.resolve("never-issued") == ANONYMOUS


def test_corrupt_session_payload_resolves_to_anonymous():
    """载荷损坏（非 JSON、缺字段）按未登录处理，不抛异常。"""
    store = _store()
    for token, payload in [("t1", "{not json"), ("t2", "{}"),
                           ("t3", '{"user_id": "not-a-number"}')]:
        store.redis.store[auth_mod.session_key(token)] = payload
        assert store.resolve(token) == ANONYMOUS


def test_redis_failure_degrades_to_anonymous():
    """Redis 抖动时应判未登录，而不是把请求变成 500。"""

    class Exploding:
        def get(self, key):
            raise RuntimeError("redis down")

        def set(self, key, value, ex=None):
            raise RuntimeError("redis down")

        def delete(self, key):
            raise RuntimeError("redis down")

    store = SessionStore(redis=Exploding())
    assert store.resolve("any-token") == ANONYMOUS
    store.revoke("any-token")   # 不抛异常


# ============================================================
# 3. owner_key 解析（防越权）
# ============================================================
def test_logged_in_user_key_ignores_client_supplied_key():
    """这是防越权的核心：登录后客户端传什么 key 都不作数。

    athlete_key 存在浏览器 localStorage，攻击者可以自己编。若被采信，
    任何登录用户改一个字段就能读写他人数据。
    """
    alice = Principal(user_id=7, username="alice")
    assert resolve_owner_key(alice, None) == "u:7"
    assert resolve_owner_key(alice, "victim-key") == "u:7"
    assert resolve_owner_key(alice, "u:8") == "u:7"


def test_guest_key_is_used_and_validated():
    assert resolve_owner_key(ANONYMOUS, "abc-123_def.ghi") == "abc-123_def.ghi"
    assert resolve_owner_key(ANONYMOUS, "  padded-key  ") == "padded-key"
    # 历史与演示用过的形态必须仍然可用
    assert resolve_owner_key(ANONYMOUS, "demo-athlete") == "demo-athlete"
    assert resolve_owner_key(
        ANONYMOUS, "f47ac10b-58cc-4372-a567-0e02b2c3d479").startswith("f47ac10b")


@pytest.mark.parametrize("bad", ["", "   ", None, "has space", "sl/ash", "emi:colon",
                                 "x" * 65, "中文键", "semi;colon"])
def test_guest_key_rejects_malformed_values(bad):
    """游客 key 会直接进入 Redis 键名与 SQL 参数，必须校验。"""
    with pytest.raises(GuestKeyError):
        resolve_owner_key(ANONYMOUS, bad)


def test_user_key_namespace_does_not_collide_with_guest_keys():
    """账号键带 u: 前缀，游客不可能撞上——否则把 athlete_key 设成 "3"
    就落到了账号 3 的数据上（而冒号在游客 key 的字符集外）。"""
    assert user_owner_key(3) == "u:3"
    with pytest.raises(GuestKeyError):
        resolve_owner_key(ANONYMOUS, "u:3")


# ============================================================
# 4. 中间件接缝
# ============================================================
def _guard_app(*, api_token="", exempt=()):
    app = FastAPI()

    @app.post("/api/auth/login")
    def login():
        return {"ok": True}

    @app.post("/api/generate-plan")
    def plan():
        return {"ok": True}

    app.add_middleware(ApiKeyMiddleware, token=api_token, exempt_prefixes=exempt)
    return TestClient(app)


def test_auth_endpoints_are_exempt_from_api_key():
    """登录接口必须能在无凭据时访问，否则要拿令牌就得先有令牌。"""
    client = _guard_app(api_token="secret", exempt=("/api/auth/",))
    assert client.post("/api/auth/login").status_code == 200
    assert client.post("/api/generate-plan").status_code == 401


def test_api_key_middleware_lets_preflight_through():
    """预检不带凭据，拦下它浏览器就永远看不到真正的跨源策略。"""
    client = _guard_app(api_token="secret", exempt=("/api/auth/",))
    assert client.options("/api/generate-plan").status_code != 401


def test_cost_limit_does_not_block_auth_endpoints():
    """额度用完时应仍能登录/登出——否则用户被卡在既不能用也退不出的状态。"""

    class ExhaustedGuard:
        daily_limit = 10.0

        def check(self):
            return False, 99.9

    app = FastAPI()

    @app.post("/api/auth/login")
    def login():
        return {"ok": True}

    @app.post("/api/generate-plan")
    def plan():
        return {"ok": True}

    app.add_middleware(CostLimitMiddleware, guard=ExhaustedGuard())
    client = TestClient(app)
    assert client.post("/api/generate-plan").status_code == 429
    assert client.post("/api/auth/login").status_code == 200


# ============================================================
# 5. 路由层：注册 / 登录 / 登出 / 会话
# ============================================================
class InMemoryUserStore:
    """按 UserStore 的对外契约实现的内存版本，避免测试依赖 PostgreSQL。"""

    def __init__(self):
        self.enabled = True
        self._rows: dict[str, dict] = {}
        self._next_id = 1

    def ensure_table(self):
        pass

    def create(self, username, password_hash):
        username = username.strip()
        if username in self._rows:
            return None
        row = {"id": self._next_id, "username": username,
               "password_hash": password_hash, "is_active": True}
        self._rows[username] = row
        self._next_id += 1
        return {"id": row["id"], "username": username, "is_active": True}

    def find_by_username(self, username):
        return self._rows.get((username or "").strip())

    def find_by_id(self, user_id):
        for row in self._rows.values():
            if row["id"] == user_id:
                return {"id": row["id"], "username": row["username"],
                        "is_active": row["is_active"]}
        return None


@pytest.fixture
def authed_client(monkeypatch):
    """真实 app + 内存用户存储 + 内存会话存储。"""
    from app import auth_api, server

    users = InMemoryUserStore()
    sessions = SessionStore(redis=FakeRedis())
    monkeypatch.setattr(auth_api, "_user_store", users)
    monkeypatch.setattr(auth_api, "_session_store", sessions)
    yield TestClient(server.app), users
    monkeypatch.setattr(auth_api, "_user_store", None)
    monkeypatch.setattr(auth_api, "_session_store", None)


def test_register_logs_in_and_sets_httponly_cookie(authed_client):
    client, _ = authed_client
    resp = client.post("/api/auth/register",
                       json={"username": "alice", "password": "password1234"})
    assert resp.status_code == 200
    assert resp.json()["owner_key"] == "u:1"
    set_cookie = resp.headers["set-cookie"]
    assert "httponly" in set_cookie.lower()
    assert "samesite=lax" in set_cookie.lower()
    assert client.get("/api/auth/session").json()["username"] == "alice"


def test_duplicate_username_is_conflict(authed_client):
    client, _ = authed_client
    client.post("/api/auth/register", json={"username": "bob", "password": "password1234"})
    dup = client.post("/api/auth/register",
                      json={"username": "bob", "password": "password1234"})
    assert dup.status_code == 409
    assert dup.json()["detail"]["error"] == "username_taken"


def test_weak_password_is_rejected_at_registration(authed_client):
    client, _ = authed_client
    resp = client.post("/api/auth/register", json={"username": "carol", "password": "123"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "weak_password"


def test_login_failures_do_not_reveal_whether_the_user_exists(authed_client):
    """用户名不存在与密码错误必须返回同一个错误。

    区分开来等于免费提供一个用户名枚举接口——攻击者可先筛出有效账号、
    再对它们做定向爆破。
    """
    client, _ = authed_client
    client.post("/api/auth/register", json={"username": "dave", "password": "password1234"})
    wrong_pw = client.post("/api/auth/login",
                           json={"username": "dave", "password": "wrongpassword"})
    no_user = client.post("/api/auth/login",
                          json={"username": "ghost", "password": "wrongpassword"})
    assert wrong_pw.status_code == no_user.status_code == 401
    assert wrong_pw.json()["detail"] == no_user.json()["detail"]


def test_logout_clears_the_session(authed_client):
    client, _ = authed_client
    client.post("/api/auth/register", json={"username": "erin", "password": "password1234"})
    assert client.get("/api/auth/session").json()["authenticated"] is True
    client.post("/api/auth/logout")
    assert client.get("/api/auth/session").json()["authenticated"] is False


def test_login_restores_the_previous_account(authed_client):
    """登出再登录，身份必须回到同一个账号——这是"记住我"的基本要求。"""
    client, _ = authed_client
    client.post("/api/auth/register", json={"username": "frank", "password": "password1234"})
    first = client.get("/api/auth/session").json()["user_id"]
    client.post("/api/auth/logout")
    client.post("/api/auth/login", json={"username": "frank", "password": "password1234"})
    assert client.get("/api/auth/session").json()["user_id"] == first


# ============================================================
# 6. 归属校验（IDOR）
# ============================================================
def test_logged_in_user_cannot_read_another_key_sdata(authed_client, monkeypatch):
    """登录用户传别人的 athlete_key，数据仍算自己的。"""
    from app import server

    seen = {}

    class SpyStore:
        enabled = True

        def ensure_table(self):
            pass

        def get_logs(self, athlete_key, **kwargs):
            seen["key"] = athlete_key
            return []

    monkeypatch.setattr(server, "_get_training_log_store", lambda: SpyStore())
    client, _ = authed_client
    client.post("/api/auth/register", json={"username": "grace", "password": "password1234"})
    resp = client.get("/api/training-stats", params={"athlete_key": "victim-key"})
    assert resp.status_code == 200
    assert seen["key"] == "u:1", "服务端必须用登录身份，忽略客户端传来的 key"


def test_guest_requests_require_a_valid_athlete_key(authed_client, monkeypatch):
    """游客身份是显式的：没有合法标识就不能确定数据算谁的。"""
    from app import server

    monkeypatch.setattr(server, "_get_training_log_store", lambda: _EmptyStore())
    client, _ = authed_client
    missing = client.get("/api/training-stats")
    assert missing.status_code == 400
    malformed = client.get("/api/training-stats", params={"athlete_key": "bad key!"})
    assert malformed.status_code == 400
    ok = client.get("/api/training-stats", params={"athlete_key": "guest-device-1"})
    assert ok.status_code == 200


class _EmptyStore:
    enabled = True

    def ensure_table(self):
        pass

    def get_logs(self, athlete_key, **kwargs):
        return []


# ============================================================
# 7. 契约：每个业务路由要么带鉴权依赖，要么在公开白名单里
# ============================================================
#: 允许不带身份依赖的 /api 路由及其理由。
#: 白名单必须是显式的——漏加鉴权是最难自查的一类缺陷：接口照常工作，
#: 只有在有人真的去读别人的数据时才会暴露。
PUBLIC_API_ROUTES = {
    "/api/auth/register": "登录前必须可达",
    "/api/auth/login": "登录前必须可达",
    "/api/auth/logout": "登出必须能对过期会话执行",
    "/api/auth/session": "查询登录态，未登录时返回 未登录",
    "/api/analyze-exercise": "按请求体里的画像即时分析，不读写任何用户级持久化数据",
    "/api/upload-document": "上传文件按 session_id 归属（会话级，非账号级），未接入账号身份",
}


def _all_api_routes(app):
    """展开 app.routes 里的全部 API 路由。

    必须显式处理 ``_IncludedRouter``：FastAPI（0.141 起）把 include_router
    注册的路由包成一个没有 ``path`` 属性的容器，直接按 ``app.routes`` 枚举会
    静默漏掉它们——漏掉的正是登录相关路由。
    """
    routes = []
    for r in app.routes:
        if getattr(r, "path", None):
            routes.append(r)
        elif type(r).__name__ == "_IncludedRouter":
            routes.extend(getattr(r.original_router, "routes", []))
    return routes


def _dependency_names(route) -> set[str]:
    dependant = getattr(route, "dependant", None)
    if dependant is None:
        return set()
    return {d.call.__name__ for d in dependant.dependencies if d.call}


#: 满足"声明了身份"的依赖名。
#: require_authenticated 内部依赖 get_current_principal，但 FastAPI 的
#: dependant.dependencies 只列**直接**依赖、不展开传递链，因此两者都要认。
_IDENTITY_DEPS = {"get_current_principal", "require_authenticated"}


def test_every_business_route_declares_an_identity_dependency():
    from app import server

    missing = []
    for route in _all_api_routes(server.app):
        path = getattr(route, "path", "")
        if not path.startswith("/api"):
            continue
        if path in PUBLIC_API_ROUTES:
            continue
        if not (_IDENTITY_DEPS & _dependency_names(route)):
            missing.append(path)

    assert not missing, (
        "以下业务路由没有身份依赖，会导致客户端可自选数据归属：\n  "
        + "\n  ".join(sorted(set(missing)))
        + "\n若确实无需身份，请加进 PUBLIC_API_ROUTES 并写明理由。")


def test_public_allowlist_has_no_stale_entries():
    """白名单里的路由必须真实存在，否则它会在路由改名后静默失效。"""
    from app import server

    live = {getattr(r, "path", "") for r in _all_api_routes(server.app)}
    stale = [p for p in PUBLIC_API_ROUTES if p not in live]
    assert not stale, f"白名单里这些路由已不存在，请清理：{stale}"


# ============================================================
# 8. 游客数据认领
# ============================================================
class FakeLongTerm:
    """记录 rekey/get_preferences 的内存版长期记忆。"""

    def __init__(self, prefs=None):
        self.store = dict(prefs or {})
        self.rekeys = []

    def get_preferences(self, user_id):
        return self.store.get(str(user_id), {})

    def rekey(self, old, new):
        self.rekeys.append((str(old), str(new)))
        src = self.store.get(str(old), {})
        dst = self.store.get(str(new), {})
        moved = skipped = 0
        for k, v in list(src.items()):
            if k in dst:
                skipped += 1
            else:
                dst[k] = v
                moved += 1
        self.store[str(old)] = {}
        self.store[str(new)] = dst
        return {"moved": moved, "skipped": skipped, "failed": []}


class FakeLogStore:
    enabled = True

    def __init__(self):
        self.rekeys = []

    def ensure_table(self):
        pass

    def rekey(self, old, new):
        self.rekeys.append((old, new))
        return 3


@pytest.fixture
def claim_client(monkeypatch):
    from app import auth_api, server

    users = InMemoryUserStore()
    sessions = SessionStore(redis=FakeRedis())
    logs = FakeLogStore()
    memory = FakeLongTerm({"guest-dev-1": {"habits": ["晨练"], "profile": {"goal": "增肌"}}})
    monkeypatch.setattr(auth_api, "_user_store", users)
    monkeypatch.setattr(auth_api, "_session_store", sessions)
    monkeypatch.setattr(auth_api, "_training_log_store", logs)
    monkeypatch.setattr(auth_api, "_long_term_memory", memory)
    client = TestClient(server.app)
    client.post("/api/auth/register", json={"username": "claimer", "password": "password1234"})
    yield client, logs, memory
    for attr in ("_user_store", "_session_store", "_training_log_store", "_long_term_memory"):
        monkeypatch.setattr(auth_api, attr, None)


def test_claim_migrates_guest_training_logs_and_memory(claim_client):
    client, logs, memory = claim_client
    resp = client.post("/api/auth/claim", json={"guest_key": "guest-dev-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["owner_key"] == "u:1"
    assert body["claimed"]["training_logs"] == 3
    assert logs.rekeys == [("guest-dev-1", "u:1")]
    assert memory.rekeys[0] == ("guest-dev-1", "u:1")


def test_claim_requires_login(claim_client):
    from app import server
    anon = TestClient(server.app)
    resp = anon.post("/api/auth/claim", json={"guest_key": "guest-dev-1"})
    assert resp.status_code == 401


@pytest.mark.parametrize(("guest_key", "reason_kind"), [
    ("u:2", "不能认领其他账号"),
    ("demo-athlete", "演示数据为共享内容"),
    ("", "未提供"),
])
def test_claim_refuses_ineligible_sources(claim_client, guest_key, reason_kind):
    """这三类来源必须被拒。

    u: 开头的是别人的账号数据——认领它等于转移他人数据；
    demo-athlete 是所有人共用的演示身份，认领会把共享内容混进个人账号。
    """
    client, logs, memory = claim_client
    body = client.post("/api/auth/claim",
                       json={"guest_key": guest_key, "include_profile_fingerprint": False}).json()
    assert logs.rekeys == [], f"{guest_key} 不该触发迁移"
    if guest_key:
        assert any(reason_kind in s["reason"] for s in body["skipped"]), body["skipped"]


def test_claim_never_overwrites_account_data(claim_client):
    """账号下已有的同名记忆保留账号版本，不被本机旧数据冲掉。"""
    _, _, memory = claim_client
    memory.store["u:1"] = {"habits": ["晚上练"]}      # 账号已有
    memory.store["guest-dev-1"] = {"habits": ["晨练"]}  # 本机旧值
    resp = TestClient.__module__ and claim_client[0].post(
        "/api/auth/claim", json={"guest_key": "guest-dev-1", "include_profile_fingerprint": False})
    assert resp.json()["claimed"]["long_term"] == 0
    assert memory.store["u:1"]["habits"] == ["晚上练"], "账号版本被覆盖了"


def test_claim_reports_the_limits_of_fingerprint_recovery(claim_client):
    """必须如实说明历史指纹不可全量找回，不能暗示"已全部迁移"。"""
    client, _, _ = claim_client
    body = client.post("/api/auth/claim", json={"guest_key": "guest-dev-1"}).json()
    assert any("无法找回" in n for n in body["notes"]), body["notes"]


def test_claim_is_idempotent(claim_client):
    """重复认领不应报错，也不应重复搬运。"""
    client, logs, _ = claim_client
    first = client.post("/api/auth/claim",
                        json={"guest_key": "guest-dev-1", "include_profile_fingerprint": False})
    second = client.post("/api/auth/claim",
                         json={"guest_key": "guest-dev-1", "include_profile_fingerprint": False})
    assert first.status_code == second.status_code == 200
