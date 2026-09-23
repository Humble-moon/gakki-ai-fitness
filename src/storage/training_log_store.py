"""
训练日志存储层 — 管理 training_logs + training_log_entries 的读写

职责：记录用户实际完成的训练（重量/组数/次数/RPE），供统计聚合与
      自适应调整建议消费。

与 DocumentStore 的关系：同一套主从写入范式（主表 + 明细表共享一个事务），
不同之处在于本模块额外承担「幂等按需建表」——因为训练日志表不在 init_db()
的调用路径上（init_db 只在 seed() 和知识库摄入时触发，纯 API 启动不会调）。
"""

import logging
from datetime import date, datetime, timedelta

from src.storage.pg import PGClient

logger = logging.getLogger(__name__)

# athlete_key 长度上限（前端生成的是 UUID4 字符串，36 字符；留出前缀余量）
MAX_ATHLETE_KEY_LEN = 64

# 单次查询返回的日志条数上限，防止一次拉爆内存
MAX_LOGS_PER_QUERY = 200


class TrainingLogStore:
    """训练日志的存储与读取。

    职责：
    - ensure_table(): 幂等按需建表（失败则整体降级，不阻断主链路）
    - save_log(): 保存一次训练（主表 + 明细同事务）
    - get_logs(): 按运动员取历史日志（含明细）
    - delete_logs(): 清理某运动员的全部日志（供 seed --reset 使用）

    降级语义：与 SparseRetriever 一致——存储不可用时 enabled=False，
    所有读写方法静默返回空值，由调用方（API 层）转成 503，而不是抛异常
    把计划生成等无关链路一起带崩。
    """

    def __init__(self):
        self.db = PGClient()
        self.enabled = True          # 建表失败 → 训练闭环功能整体降级
        self._table_ready = False

    # ------------------------------------------------------------------
    # 建表
    # ------------------------------------------------------------------

    def ensure_table(self) -> None:
        """按需建表。幂等；失败只记日志并降级，不影响主链路。

        用 Base.metadata.create_all(tables=[...]) 而不是手写 DDL：
        这两张表本身就在 ORM 单点真相里，由 ORM 生成 DDL 可以根治
        「ORM 改了、建表语句忘了改」的漂移——项目里 knowledge_chunks.parent_id
        就是这样漂移掉的（代码在用、ORM 没定义）。手写 DDL 只适用于不在 ORM
        里的旁路表（如 knowledge_chunks_sparse）。

        checkfirst=True（默认）保证幂等，等价于 CREATE TABLE IF NOT EXISTS。
        """
        if self._table_ready:
            return
        try:
            from src.models.db_models import Base, TrainingLog, TrainingLogEntry
            Base.metadata.create_all(
                self.db.engine,
                tables=[TrainingLog.__table__, TrainingLogEntry.__table__],
                checkfirst=True,
            )
            self._table_ready = True
        except Exception as exc:
            logger.warning("训练日志表创建失败，训练闭环功能降级禁用: %s", exc)
            self.enabled = False

    def _ready(self) -> bool:
        """确保表已就绪，返回存储是否可用。"""
        if not self._table_ready and self.enabled:
            self.ensure_table()
        return self._table_ready and self.enabled

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def save_log(self, athlete_key: str, entries: list[dict],
                 log_date: str | date | None = None, day_index: int = 0,
                 focus: str = "", plan_id: str = "", session_rpe: float | None = None,
                 duration_min: int = 0, body_weight: float | None = None,
                 notes: str = "") -> int | None:
        """保存一次训练日志（主表 + 明细同事务写入）。返回 log_id，不可用时返回 None。

        输入：
            athlete_key: str  — 稳定运动员标识
            entries: list[dict] — 动作明细，每项含
                exercise_name / sets / reps / weight / rpe / completed
            log_date: 训练日期，支持 "YYYY-MM-DD" 字符串或 date 对象；
                      缺省用**本地**日期（不是 UTC，避免东八区晚间记到次日）
            其余为可选元信息

        为什么主从必须同事务：明细写了一半失败会留下一条没有任何动作的空日志，
        统计时这条空日志会污染完成率（分母多了一次、分子是 0）。
        """
        if not self._ready():
            return None

        resolved_date = self._parse_date(log_date)

        with self.db.transaction() as tx:
            row = tx.fetch_one(
                """INSERT INTO training_logs
                   (athlete_key, log_date, day_index, focus, plan_id, session_rpe,
                    duration_min, body_weight, notes, created_at)
                   VALUES (:ak, :ld, :di, :fc, :pid, :srpe, :dur, :bw, :nt, :ts)
                   RETURNING id""",
                {"ak": athlete_key[:MAX_ATHLETE_KEY_LEN], "ld": resolved_date,
                 "di": day_index, "fc": focus or "", "pid": plan_id or "",
                 "srpe": session_rpe, "dur": duration_min or 0, "bw": body_weight,
                 "nt": (notes or "")[:512], "ts": datetime.utcnow()},
            )
            log_id = row.id

            for entry in entries or []:
                name = (entry.get("exercise_name") or "").strip()
                if not name:
                    continue                # 跳过没有动作名的空行，不写脏数据
                tx.execute(
                    """INSERT INTO training_log_entries
                       (log_id, athlete_key, exercise_name, sets, reps, weight,
                        rpe, completed, created_at)
                       VALUES (:lid, :ak, :en, :st, :rp, :wt, :rpe, :cp, :ts)""",
                    {"lid": log_id, "ak": athlete_key[:MAX_ATHLETE_KEY_LEN],
                     "en": name[:100], "st": int(entry.get("sets") or 0),
                     "rp": int(entry.get("reps") or 0),
                     "wt": float(entry.get("weight") or 0.0),
                     "rpe": entry.get("rpe"),
                     "cp": 1 if entry.get("completed", True) else 0,
                     "ts": datetime.utcnow()},
                )

            return log_id

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def get_logs(self, athlete_key: str, weeks: int = 12,
                 limit: int = MAX_LOGS_PER_QUERY, exercise: str | None = None) -> list[dict]:
        """取某运动员的历史训练日志（含动作明细），按日期倒序。

        输入：
            athlete_key: str — 稳定运动员标识
            weeks: int — 只取最近多少周的数据
            limit: int — 返回条数上限
            exercise: str | None — 只取包含该动作的日志（用于单动作进步曲线）

        实现说明：两条 SQL（主表 + 明细）后在 Python 侧按 log_id 分组，
        而不是 JOIN——明细行数是一次训练 × 动作数，JOIN 会把主表字段重复
        传输 N 遍；且分组逻辑只有一处，避免 SQL 变复杂。
        """
        if not self._ready():
            return []

        since = self._parse_date(None) - timedelta(weeks=weeks)

        if exercise:
            # 先筛出含该动作的日志 id，再取这些日志的完整明细
            id_rows = self.db.fetch_all(
                """SELECT DISTINCT log_id FROM training_log_entries
                   WHERE athlete_key = :ak AND exercise_name = :ex""",
                {"ak": athlete_key, "ex": exercise},
            )
            wanted = [r.log_id for r in id_rows]
            if not wanted:
                return []
            logs = self.db.fetch_all(
                """SELECT id, athlete_key, log_date, day_index, focus, plan_id,
                          session_rpe, duration_min, body_weight, notes, created_at
                   FROM training_logs
                   WHERE athlete_key = :ak AND log_date >= :since AND id = ANY(:ids)
                   ORDER BY log_date DESC, id DESC
                   LIMIT :lim""",
                {"ak": athlete_key, "since": since, "ids": wanted, "lim": limit},
            )
        else:
            logs = self.db.fetch_all(
                """SELECT id, athlete_key, log_date, day_index, focus, plan_id,
                          session_rpe, duration_min, body_weight, notes, created_at
                   FROM training_logs
                   WHERE athlete_key = :ak AND log_date >= :since
                   ORDER BY log_date DESC, id DESC
                   LIMIT :lim""",
                {"ak": athlete_key, "since": since, "lim": limit},
            )

        if not logs:
            return []

        log_ids = [r.id for r in logs]
        entries = self.db.fetch_all(
            """SELECT log_id, exercise_name, sets, reps, weight, rpe, completed
               FROM training_log_entries
               WHERE log_id = ANY(:ids)
               ORDER BY id ASC""",
            {"ids": log_ids},
        )

        by_log: dict[int, list[dict]] = {}
        for e in entries:
            by_log.setdefault(e.log_id, []).append({
                "exercise_name": e.exercise_name,
                "sets": e.sets, "reps": e.reps, "weight": float(e.weight or 0.0),
                "rpe": e.rpe, "completed": bool(e.completed),
            })

        return [
            {"log_id": r.id, "athlete_key": r.athlete_key,
             "log_date": r.log_date.isoformat() if r.log_date else "",
             "day_index": r.day_index, "focus": r.focus, "plan_id": r.plan_id,
             "session_rpe": r.session_rpe, "duration_min": r.duration_min,
             "body_weight": r.body_weight, "notes": r.notes,
             "entries": by_log.get(r.id, [])}
            for r in logs
        ]

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def delete_logs(self, athlete_key: str) -> int:
        """删除某运动员的全部训练日志（主从一起删）。返回删除的主表条数。

        供 seed 脚本 --reset 使用，保证重复灌演示数据不会叠加。
        """
        if not self._ready():
            return 0

        with self.db.transaction() as tx:
            row = tx.fetch_one(
                "SELECT COUNT(*) AS cnt FROM training_logs WHERE athlete_key = :ak",
                {"ak": athlete_key},
            )
            count = int(row.cnt) if row else 0
            tx.execute("DELETE FROM training_log_entries WHERE athlete_key = :ak",
                       {"ak": athlete_key})
            tx.execute("DELETE FROM training_logs WHERE athlete_key = :ak",
                       {"ak": athlete_key})
            return count

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(value: str | date | None) -> date:
        """把入参统一成 date。缺省用本地当天。

        注意用 datetime.now() 而非 utcnow()：log_date 的语义是"用户所在
        时区的日历日"，东八区晚上 8 点训练应当记今天，而不是 UTC 的次日。
        """
        if value is None:
            return datetime.now().date()
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        try:
            return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            logger.warning("无法解析训练日期 %r，回退为今天", value)
            return datetime.now().date()
