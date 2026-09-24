"""按自然日累计 LLM 成本并设硬上限。

【要解决的问题】
    `cost_tracker` 是**观测**模块——内存累加 + 打日志，没有任何上限。
    本地 demo 无所谓，但对外开放后，一次脚本刷量、或某个用户反复点"重新生成"，
    成本就会无界增长，而这是要真金白银付的。上线前必须有刹车。

【为什么存 Redis 而不是内存】
    内存态计数在进程重启后归零，等于重启一次就重置预算——限制形同虚设。
    而重启在真实部署里很常见（部署、崩溃恢复、容器调度）。

【为什么在 API 入口检查，而不是在 LLM 调用层抛异常】
    在 LLM 层抛异常会中断一条已经跑了十几秒的流水线，用户等了半天才失败。
    入口检查是快速失败：超限直接返回 429 与可读原因，不浪费任何一次生成。

【Redis 不可用时放行，而不是拒绝】
    这是成本保护，不是安全边界。Redis 挂掉就让全站不可用，代价远大于超支风险；
    何况还有 `RateLimitMiddleware`（内存态、不依赖 Redis）作为兜底。
    但降级会记 warning，以便从日志发现"预算保护当前失效"。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

#: 东八区。预算按用户的自然日切分，用 UTC 会让"今天"在北京时间早上 8 点重置。
_CST = timezone(timedelta(hours=8))

_KEY_PREFIX = "cost:guard:daily"


def _today_key(now: datetime | None = None) -> str:
    """当日计数键，形如 cost:guard:daily:2026-09-24。"""
    moment = now or datetime.now(_CST)
    return f"{_KEY_PREFIX}:{moment.astimezone(_CST).strftime('%Y-%m-%d')}"


class CostGuard:
    """按天累计 LLM 成本，超过上限即拒绝后续请求。

    典型接线：把 ``record`` 注册进 ``cost_tracker`` 的 sink，
    这样 LLMProvider 不必知道本模块的存在；在 API 入口调 ``check``。
    """

    def __init__(self, redis_client=None, daily_limit: float = 10.0):
        """
        Args:
            redis_client: 具备 get/set 接口的客户端；为 None 时本守卫退化为
                "只记日志不拦截"——本地开发与单元测试常这样构造。
            daily_limit: 当日成本上限（元）。<=0 表示不限制。
        """
        self.redis = redis_client
        self.daily_limit = float(daily_limit or 0)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def spent_today(self) -> float:
        """当日已花费（元）。取不到时返回 0.0（配合下面的降级策略）。"""
        if self.redis is None:
            return 0.0
        try:
            raw = self.redis.get(_today_key())
            return float(raw) if raw is not None else 0.0
        except Exception as exc:  # noqa: BLE001 - 降级见模块 docstring
            logger.warning("[cost_guard] 读取当日成本失败，按 0 处理：%s", exc)
            return 0.0

    def check(self) -> tuple[bool, float]:
        """返回 (是否放行, 当日已花费)。

        未配置限额或未接 Redis 时一律放行。
        """
        if self.daily_limit <= 0 or self.redis is None:
            return True, 0.0
        spent = self.spent_today()
        return spent < self.daily_limit, spent

    # ------------------------------------------------------------------
    # 记账
    # ------------------------------------------------------------------

    def record(self, cost: float) -> None:
        """累加一次调用的成本。作为 ``cost_tracker`` 的 sink 被调用。

        自身出错不得影响主流程——记账失败顶多是预算不准，
        让一次正常的 LLM 调用因此失败才是更糟的取舍。
        """
        if self.redis is None or not cost or cost <= 0:
            return
        try:
            key = _today_key()
            current = self.redis.get(key)
            total = (float(current) if current is not None else 0.0) + float(cost)
            # 保留两天，便于跨日排查；比"永不过期"更省 Redis 空间
            self.redis.set(key, str(total), ex=2 * 24 * 3600)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[cost_guard] 记成本失败：%s", exc)

    # ------------------------------------------------------------------
    # 展示
    # ------------------------------------------------------------------

    def describe_limit(self) -> str:
        if self.daily_limit <= 0:
            return "未设置当日成本上限"
        return (
            f"当日 LLM 成本上限 ¥{self.daily_limit:.2f}，"
            f"已用 ¥{self.spent_today():.2f}"
        )
