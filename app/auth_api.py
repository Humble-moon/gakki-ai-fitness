"""账号相关接口：注册、登录、登出、会话查询。

【为什么单独成模块而不是塞进 server.py】
    server.py 已近千行，且这些路由有一条特殊约束——它们必须能在
    **没有会话的情况下**访问（否则要登录就得先登录）。把这条约束
    限制在一个小模块里，比散落在业务路由中间更容易看清。

【会话为什么走 HttpOnly Cookie 而不是 Authorization 头】
    前端的流式问答用的是 fetch + ReadableStream（不是 EventSource），
    同源 fetch 默认就会带上 Cookie。选 Cookie 意味着前端现有的裸 fetch
    一处都不用改；选 Bearer 则要逐个补 Authorization 头，将来若改用
    EventSource 还得退化成把令牌写进 URL——那样令牌会进访问日志。

    HttpOnly 让 JS 读不到令牌，XSS 也偷不走；SameSite=Lax 挡掉跨站
    POST，因此登录、注册、登出这些改状态的端点不需要额外的 CSRF 令牌。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from src.security import auth as auth_mod
from src.security.session import (
    ANONYMOUS,
    GuestKeyError,
    Principal,
    SessionStore,
    is_safe_guest_key,
    resolve_owner_key,
    user_owner_key,
)
from src.storage.user_store import UserStore


def _owner_of(principal: Principal) -> str:
    return user_owner_key(principal.user_id)

logger = logging.getLogger(__name__)

COOKIE_NAME = "ironmind_session"

router = APIRouter(prefix="/api/auth", tags=["auth"])

# 惰性单例。不在模块导入时构造，是为了让测试能替换掉它们而不影响
# 其他测试——项目的接口层测试普遍用 monkeypatch 打桩，这里沿用同一手法。
_user_store: UserStore | None = None
_session_store: SessionStore | None = None


def get_user_store() -> UserStore:
    global _user_store
    if _user_store is None:
        _user_store = UserStore()
    return _user_store


def get_session_store() -> SessionStore:
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store


# 数据认领要跨几类存储搬迁数据，因此这里也持有它们。
# 不 import app.server 的实例：server 反过来 import 本模块，会形成循环导入。
_training_log_store = None
_body_measure_store = None
_long_term_memory = None


def get_training_log_store():
    global _training_log_store
    if _training_log_store is None:
        from src.storage.training_log_store import TrainingLogStore
        _training_log_store = TrainingLogStore()
    return _training_log_store


def get_body_measure_store():
    global _body_measure_store
    if _body_measure_store is None:
        from src.storage.body_measure_store import BodyMeasureStore
        _body_measure_store = BodyMeasureStore()
    return _body_measure_store


def get_long_term_memory():
    global _long_term_memory
    if _long_term_memory is None:
        from src.memory.long_term import LongTermMemory
        _long_term_memory = LongTermMemory()
    return _long_term_memory


# ============================================================
# 身份依赖
# ============================================================
def get_current_principal(request: Request) -> Principal:
    """从 Cookie 解析当前身份。无有效会话时返回匿名——不抛 401。

    这里刻意不抛异常：绝大多数业务端点游客也能用（计划生成、问答、
    训练日志），身份只决定「数据算谁的」。需要强制登录的端点自己
    检查 ``principal.is_authenticated``。
    """
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return ANONYMOUS
    return get_session_store().resolve(token)


def require_authenticated(
    principal: Principal = Depends(get_current_principal),
) -> Principal:
    """要求已登录；未登录返回 401。"""
    if not principal.is_authenticated:
        raise HTTPException(status_code=401, detail={
            "error": "login_required",
            "message": "该操作需要登录。",
        })
    return principal


def require_owner_key(principal: Principal, client_key: str | None) -> str:
    """解析业务数据归属键，把格式错误转成 400。

    登录用户返回 ``u:{id}`` 并忽略 client_key——这一步是防越权的关键：
    athlete_key 存在前端 localStorage，攻击者可以自己编，若被采信，
    任何登录用户都能读写他人数据。
    """
    try:
        return resolve_owner_key(principal, client_key)
    except GuestKeyError as exc:
        raise HTTPException(status_code=400, detail={
            "error": "invalid_athlete_key",
            "message": str(exc),
        })


# ============================================================
# 请求体
# ============================================================
class Credentials(BaseModel):
    username: str = ""
    password: str = ""


class ClaimRequest(BaseModel):
    """认领请求：把本机游客身份下的数据挂到当前账号。"""

    guest_key: str = ""
    #: 本机存储的游客标识。只接受"看起来像游客"的值——账号键（u: 前缀）
    #: 与演示数据（demo-athlete）都会被拒，见下。

    include_profile_fingerprint: bool = True
    #: 是否顺带认领"身高体重指纹"名下的历史数据。默认开，但只能在
    #: 当前画像上算一次，因此只能救回最近一档体重的数据（见接口注释）。


# ============================================================
# Cookie 辅助
# ============================================================
def _set_session_cookie(response: Response, token: str, request: Request) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,          # JS 读不到，XSS 偷不走
        samesite="lax",         # 挡跨站 POST；同源跳转不受影响
        secure=request.url.scheme == "https",   # 本地 http 不加 Secure，否则收不到
        max_age=auth_mod.SESSION_TTL_SECONDS,
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=COOKIE_NAME, path="/")


def _user_payload(user: dict) -> dict:
    return {
        "authenticated": True,
        "user_id": user["id"],
        "username": user["username"],
        "owner_key": f"u:{user['id']}",
    }


def _unavailable() -> HTTPException:
    return HTTPException(status_code=503, detail={
        "error": "STORAGE_UNAVAILABLE",
        "message": "账号服务暂不可用（暂时连不上）。游客模式仍可使用。",
    })


# ============================================================
# 路由
# ============================================================
@router.post("/register")
def register(body: Credentials, request: Request, response: Response):
    """注册并直接登录。

    注册后立即签发会话而不是要求再登录一次：这是本地演示工具，
    多一步登录只增加摩擦，不带来安全收益。
    """
    store = get_user_store()
    if not store.enabled:
        raise _unavailable()
    username = body.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail={
            "error": "invalid_username", "message": "用户名不能为空。"})
    try:
        password_hash = auth_mod.hash_password(body.password)
    except auth_mod.AuthError as exc:
        raise HTTPException(status_code=400, detail={
            "error": "weak_password", "message": str(exc)})
    if not store.enabled:
        raise _unavailable()

    created = store.create(username, password_hash)
    if created is None:
        # create 返回 None 有两种可能：用户名已存在，或数据库在两次
        # 检查之间挂了。前者是用户错误、后者是服务错误，不能混为一谈，
        # 因此回查一次来区分。
        if store.find_by_username(username):
            raise HTTPException(status_code=409, detail={
                "error": "username_taken", "message": "该用户名已被使用。"})
        raise _unavailable()

    token = get_session_store().create(created["id"], created["username"])
    _set_session_cookie(response, token, request)
    return _user_payload(created)


@router.post("/login")
def login(body: Credentials, request: Request, response: Response):
    store = get_user_store()
    if not store.enabled:
        raise _unavailable()

    row = store.find_by_username(body.username)
    # 用户名不存在与密码错误返回同一个响应：区分开来等于提供一个
    # 用户名枚举接口，攻击者可据此筛出有效账号再做定向爆破。
    if not row or not auth_mod.verify_password(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail={
            "error": "invalid_credentials", "message": "用户名或密码不正确。"})
    if not row.get("is_active", True):
        raise HTTPException(status_code=403, detail={
            "error": "account_disabled", "message": "该账号已被停用。"})

    token = get_session_store().create(row["id"], row["username"])
    _set_session_cookie(response, token, request)
    return _user_payload(row)


@router.post("/logout")
def logout(request: Request, response: Response):
    get_session_store().revoke(request.cookies.get(COOKIE_NAME))
    _clear_session_cookie(response)
    return {"authenticated": False}


@router.get("/session")
def current_session(
    principal: Principal = Depends(get_current_principal),
):
    """查询当前登录态。前端用它决定进入账号模式还是游客模式。"""
    if not principal.is_authenticated:
        return {"authenticated": False, "user_id": None,
                "username": None, "owner_key": None}
    return {
        "authenticated": True,
        "user_id": principal.user_id,
        "username": principal.username,
        "owner_key": f"u:{principal.user_id}",
    }


# ============================================================
# 数据认领
# ============================================================
#: 演示数据的固定身份。它被所有点过"载入演示数据"的人共用，
#: 允许认领等于把共享的演示数据搬进个人账号——既污染账号，
#: 也会把别人的演示记录算到自己头上。
DEMO_ATHLETE_KEY = "demo-athlete"


def _is_claimable(guest_key: str, current_owner: str) -> tuple[bool, str]:
    """判断一个来源身份是否可以被认领。返回 (可否, 拒绝原因)。

    字符集校验不能省：认领会把 guest_key 拼进长期记忆的 SCAN 前缀
    （``LongTermMemory.rekey`` → ``{prefix}{key}:*``）。只挡 ``u:`` 前缀
    挡不住 ``*``——``guest_key="*"`` 会匹配到每一个用户的键，然后把它们
    搬到错位的新键名上并删掉原键，等于一次请求毁掉所有人的长期记忆。
    """
    key = (guest_key or "").strip()
    if not key:
        return False, "未提供 guest_key"
    if key.startswith("u:"):
        return False, "不能认领其他账号的数据"
    if not is_safe_guest_key(key):
        return False, "来源身份格式不合法"
    if key == current_owner:
        return False, "来源身份与当前账号相同"
    if key == DEMO_ATHLETE_KEY:
        return False, "演示数据为共享内容，不能认领到个人账号"
    return True, ""


@router.post("/claim")
def claim_guest_data(
    body: ClaimRequest,
    principal: Principal = Depends(require_authenticated),
):
    """把本机游客身份下的数据挂到当前账号名下。

    为什么需要它：用户完全可能先用了一阵子（生成计划、记训练日志），
    才决定注册账号。若不认领，注册反而成了"数据消失"的操作——
    与"登录是为了记住你"这件事直接矛盾。

    **能力边界必须说清楚**：只能认领游客 UUID 名下的数据。更早的历史
    数据挂在 ``make_user_key(profile)`` 生成的整数指纹下，那个指纹由
    身高体重目标伤病的哈希算出，**不可逆、也无法枚举**——只能拿用户
    *当前*的画像再算一次，因此最多救回"与当前体重相符的那一档"。
    不同体重时期的历史指纹在数学上无法找回，这一点在返回里如实标注，
    不做"已全部迁移"的暗示。
    """
    from src.core.plan_finalization import make_user_key

    owner = user_owner_key(principal.user_id)
    store = get_user_store()
    if not store.enabled:
        raise _unavailable()

    trained = get_training_log_store()
    measures = get_body_measure_store()
    memory = get_long_term_memory()

    result = {"owner_key": owner, "claimed": {}, "skipped": [], "notes": []}

    # ---- 1. 游客 UUID 名下的数据 ----
    ok, reason = _is_claimable(body.guest_key, owner)
    if ok:
        try:
            result["claimed"]["training_logs"] = trained.rekey(body.guest_key, owner)
        except Exception as exc:  # noqa: BLE001 - 单源失败不阻断其余来源
            logger.warning("[claim] 训练日志认领失败：%s", exc)
            result["claimed"]["training_logs_error"] = str(exc)

        try:
            measures.ensure_table()
            if measures.enabled:
                result["claimed"]["body_measurements"] = measures.rekey(
                    body.guest_key, owner)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[claim] 体测记录认领失败：%s", exc)
            result["claimed"]["body_measurements_error"] = str(exc)

        try:
            moved = memory.rekey(body.guest_key, owner)
            result["claimed"]["long_term"] = moved["moved"]
            if moved["skipped"]:
                result["notes"].append(
                    f"{moved['skipped']} 项本机记忆因账号下已有同名项而保留账号版本")
            if moved["failed"]:
                result["claimed"]["long_term_failed"] = moved["failed"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("[claim] 长期记忆认领失败：%s", exc)
            result["claimed"]["long_term_error"] = str(exc)
    elif body.guest_key:
        result["skipped"].append({"source": "guest_key", "reason": reason})

    # ---- 2. 画像指纹名下的数据（尽力而为）----
    if body.include_profile_fingerprint:
        profile = memory.get_preferences(owner).get("profile")
        if isinstance(profile, dict) and profile:
            fingerprint = str(make_user_key(profile))
            try:
                moved = memory.rekey(fingerprint, owner)
                if moved["moved"]:
                    result["claimed"]["long_term_from_profile"] = moved["moved"]
            except Exception as exc:  # noqa: BLE001
                logger.warning("[claim] 画像指纹记忆认领失败：%s", exc)
        result["notes"].append(
            "只认领了与当前画像相符的一档历史数据；更早时期因体重等变化产生的"
            "旧记录无法找回——它们由画像哈希生成，无法反推或枚举。")

    return result
