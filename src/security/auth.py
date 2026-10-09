"""本地账号的密码哈希与会话令牌。

【为什么用标准库而不是 passlib/bcrypt】
    项目刻意精简依赖（见 requirements.txt 的文件头注释）。``hashlib.scrypt``
    是标准库里的内存硬函数，OWASP 明确认可，无需引入任何第三方包；
    passlib 维护停滞且与新版 bcrypt 有兼容坑，为一个本地演示级的账号体系
    引入这类长期维护负担并不划算。

【为什么密码串要自描述】
    存成 ``scrypt$n$r$p$<salt>$<hash>`` 而不是裸哈希：强度参数迟早要调
    （机器变快、或发现当前参数偏弱），届时旧记录必须还能验证通过。
    自描述串让"用什么参数算的"随记录走，不必靠版本号去猜。

【为什么会话令牌只在 Redis 里存摘要】
    存明文的话，一次 Redis 备份泄露就等于把所有在线用户冒充一遍。
    客户端持有明文令牌，服务端只留 sha256 摘要——库被读走也换不出可用令牌。
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets

logger = logging.getLogger(__name__)

# scrypt 强度参数。n=2^16、r=8 约需 64MB，实测单次约 0.08s——
# 对登录体验无感，对离线爆破是有效成本。
#
# 必须显式传 maxmem：OpenSSL 默认内存上限约 32MB，不传时 n=2^15 就会
# 抛 "memory limit exceeded"（本机实测确认）。显式声明既解除该限制，
# 也让参数不随运行环境的 OpenSSL 构建而变。
_SCRYPT_N = 2 ** 16
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 64
_SCRYPT_MAXMEM = 128 * 1024 * 1024  # 2 倍于理论所需的 64MB，留余量

_ALGO = "scrypt"

# 会话有效期：7 天。健身是低频使用场景（隔天或每周几次），
# 太短会让用户每次都要重新登录。
SESSION_TTL_SECONDS = 7 * 24 * 3600

# 会话令牌在 Redis 里的键前缀
_SESSION_PREFIX = "auth:session:"

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
# 上限存在的原因：scrypt 对超长输入照样要完整计算，不设上限时
# 一个几 MB 的密码就能让每次登录吃掉大量 CPU。


class AuthError(Exception):
    """认证相关的可预期失败（用户名重复、密码错误等）。"""


# ============================================================
# 密码哈希
# ============================================================
def hash_password(password: str) -> str:
    """把明文密码算成自描述哈希串。"""
    _validate_password(password)
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt,
        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN,
        maxmem=_SCRYPT_MAXMEM,
    )
    return (f"{_ALGO}${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}"
            f"${salt.hex()}${digest.hex()}")


def verify_password(password: str, stored: str) -> bool:
    """校验明文密码是否匹配哈希串。

    任何解析失败都返回 False 而不是抛异常：调用方只关心"能不能登录"，
    而一条格式损坏的历史记录不应该让整个登录接口 500。
    """
    if not isinstance(password, str) or not isinstance(stored, str):
        return False
    if len(password) > MAX_PASSWORD_LENGTH:
        return False
    try:
        algo, n, r, p, salt_hex, hash_hex = stored.split("$")
        if algo != _ALGO:
            logger.warning("[auth] 未知的密码哈希算法: %s", algo)
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
            n=int(n), r=int(r), p=int(p), dklen=len(bytes.fromhex(hash_hex)),
            maxmem=_SCRYPT_MAXMEM,
        )
    except (ValueError, TypeError) as exc:
        logger.warning("[auth] 密码哈希串解析失败，按验证不通过处理: %s", exc)
        return False
    # 定时安全比较：普通 == 会在首个不同字节处提前返回，泄漏前缀匹配长度。
    return hmac.compare_digest(digest.hex(), hash_hex)


def _validate_password(password: str) -> None:
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"密码至少需要 {MIN_PASSWORD_LENGTH} 位")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise AuthError(f"密码不能超过 {MAX_PASSWORD_LENGTH} 位")


# ============================================================
# 会话令牌
# ============================================================
def new_session_token() -> str:
    """签发一个新的会话令牌（明文，只回给客户端一次）。"""
    return secrets.token_urlsafe(32)


def token_fingerprint(token: str) -> str:
    """令牌在服务端的存储形态（sha256 摘要）。"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def session_key(token: str) -> str:
    """令牌对应的 Redis 键。"""
    return f"{_SESSION_PREFIX}{token_fingerprint(token)}"
