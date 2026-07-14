"""
============================================================================
cost_tracker.py — LLM 调用成本追踪器（开发侧观测，不暴露给前端）
============================================================================
【角色】开发工具模块，追踪每一次 LLM 调用的 token 消耗和人民币成本。
【被调用者】LLMProvider（自动记录每次 chat/chat_stream 调用）
【调用者】  无外部依赖，纯内存累加 + Python logging 输出
【输出】    结构化日志（INFO 级别），不暴露给用户 API

【面试价值】
  面试时可以说："我为项目加了成本观测，每次 LLM 调用的 token 消耗和
  人民币成本都记录在开发日志里。优化前后对比，单次计划生成成本从 X 降到 Y。
  按日均 Z 次调用算，缓存命中每年可省约 W 元。"

【使用方式】
  from src.llm.cost_tracker import cost_tracker
  cost_tracker.record(model="deepseek-chat", tokens=1500)
  cost_tracker.log_summary()  # 手动输出累计摘要
  stats = cost_tracker.get_stats()  # 获取统计数据字典

【输出示例】
  [Cost] deepseek-chat | 1500 tokens | ¥0.0030 | 累计: 45 次 ¥0.1350
============================================================================
"""

from __future__ import annotations
import atexit
import logging
import time
from collections import defaultdict
from typing import Optional

from src.config import MODEL_PRICE_PER_1K

logger = logging.getLogger("cost")
# 设置为 INFO 级别，确保成本日志可见（开发环境）
# 生产环境可通过 logging config 调整为 WARNING 来屏蔽


class CostTracker:
    """LLM 调用成本追踪器（单例）。

    设计决策：
      - 纯内存累加，不写数据库。简单、零依赖、足够用。
      - 每次调用记录一条 INFO 日志，方便 grep/tail 实时查看。
      - 每 N 次调用自动输出累计摘要，避免刷屏。
      - atexit 注册 shutdown hook，进程退出时自动输出总汇总。
      - 线程不安全（FastAPI 单线程异步模型足够）。如果未来需要多线程，
        只需给 record() 加 threading.Lock。
    """

    _SUMMARY_INTERVAL = 25  # 每 25 次调用输出一次累计摘要

    def __init__(self):
        # per-model 累计
        self._tokens: dict[str, int] = defaultdict(int)
        self._cost: dict[str, float] = defaultdict(float)
        self._calls: dict[str, int] = defaultdict(int)
        self._total_calls: int = 0
        self._start_time: float = time.time()
        self._enabled: bool = True

    # ------------------------------------------------------------------
    # 核心 API
    # ------------------------------------------------------------------

    def record(self, model: str, tokens: int = 0, extra: str = ""):
        """记录一次 LLM 调用。

        Args:
            model:  模型名（如 "deepseek-chat"）
            tokens: 本次调用消耗的 token 数
            extra:  附加信息（如 "stream" / "json_mode" / "rewrite"），
                    会显示在日志中帮助区分调用类型
        """
        if not self._enabled:
            return

        price = MODEL_PRICE_PER_1K.get(model, MODEL_PRICE_PER_1K.get("deepseek-chat", 0.002))
        cost = tokens * price / 1000

        self._tokens[model] += tokens
        self._cost[model] += cost
        self._calls[model] += 1
        self._total_calls += 1

        # 单次调用日志：精确到小数点后 4 位
        tag = f"[{extra}]" if extra else ""
        logger.info(
            f"{tag} {model} | {tokens} tokens | "
            f"¥{cost:.4f} | 该模型累计 {self._calls[model]} 次 ¥{self._cost[model]:.4f}"
        )

        # 每 N 次输出一次汇总
        if self._total_calls % self._SUMMARY_INTERVAL == 0:
            self.log_summary()

    def record_stream(self, model: str, tokens: int = 0):
        """记录流式调用（语法糖，自动标注 stream）。"""
        self.record(model, tokens, extra="stream")

    def record_embedding(self, tokens: int = 0, count: int = 1):
        """记录 embedding 调用。"""
        model = "text-embedding-v4"
        # embedding 按条数计费，不是按 token
        self.record(model, tokens or count * 100, extra="embed")

    # ------------------------------------------------------------------
    # 统计输出
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """返回当前统计数据字典，供程序内部使用。"""
        elapsed = time.time() - self._start_time
        total_cost = sum(self._cost.values())
        total_tokens = sum(self._tokens.values())
        return {
            "total_calls": self._total_calls,
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 4),
            "elapsed_seconds": round(elapsed, 0),
            "per_model": {
                m: {
                    "calls": self._calls[m],
                    "tokens": self._tokens[m],
                    "cost": round(self._cost[m], 4),
                }
                for m in sorted(self._calls.keys())
            },
            # 面试用数据
            "avg_cost_per_call": round(total_cost / self._total_calls, 6) if self._total_calls else 0,
            "avg_tokens_per_call": round(total_tokens / self._total_calls, 0) if self._total_calls else 0,
        }

    def log_summary(self):
        """输出累计摘要到日志（INFO 级别）。"""
        stats = self.get_stats()
        if stats["total_calls"] == 0:
            return

        elapsed_m = stats["elapsed_seconds"] / 60
        logger.info(
            "=" * 55
        )
        logger.info(
            f"[Cost Summary] 总调用 {stats['total_calls']} 次 | "
            f"总 tokens {stats['total_tokens']:,} | "
            f"总成本 ¥{stats['total_cost']:.4f} | "
            f"运行 {elapsed_m:.0f} min"
        )
        logger.info(
            f"  平均每次: {stats['avg_tokens_per_call']:.0f} tokens | "
            f"¥{stats['avg_cost_per_call']:.6f}"
        )
        for m, d in stats["per_model"].items():
            logger.info(
                f"  {m}: {d['calls']} 次 | "
                f"{d['tokens']:,} tokens | "
                f"¥{d['cost']:.4f}"
            )
        # 预估：如果这是 10 分钟的数据，一天的成本大约是多少
        if elapsed_m > 0:
            daily_est = stats["total_cost"] * (1440 / elapsed_m)  # 24h
            logger.info(f"  按此速率预估日均成本: ¥{daily_est:.4f}")
        logger.info(
            "=" * 55
        )

    def _shutdown(self):
        """进程退出时自动输出总汇总。由 atexit 注册。"""
        if self._total_calls > 0:
            logger.info("[Cost Tracker] 进程退出，最终汇总：")
            self.log_summary()

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False


# 全局单例，所有模块 import 同一个实例
cost_tracker = CostTracker()
atexit.register(cost_tracker._shutdown)
