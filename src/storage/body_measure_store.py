"""
体测记录存储层 — 管理 body_measurements 的读写

职责：记录不定期的身体围度与成分快照（体重、体脂、肌肉量、各部位围度），
      供进度趋势展示消费。

与 TrainingLogStore 的关系：同一套「幂等按需建表 + 存储不可用即降级」范式。
不同之处在于本模块是单表、无明细，因此没有主从同事务的约束；但仍保留
rekey，因为游客注册后认领本机数据时，这批记录同样要跟着换身份——
漏掉的话，用户在体测页看到的是空的，而他明明量过。

关于表结构的来源：body_measurements 由 alembic 迁移创建（见 alembic/versions）。
这里的 ensure_table 是兜底——纯 API 启动不会跑迁移，若表尚未建好，
本模块独立降级而不是让接口 500。
"""

import logging
from datetime import date, datetime, timedelta

from src.storage.pg import PGClient

logger = logging.getLogger(__name__)

# athlete_key 长度上限，与 training_log_store 保持同一约定
MAX_ATHLETE_KEY_LEN = 64

# 单次查询返回条数上限。体测是低频记录（周级或月级），
# 200 条足够覆盖数年，不需要分页。
MAX_ROWS_PER_QUERY = 200


class BodyMeasureStore:
    """体测记录的存储与读取。

    职责：
    - ensure_table(): 幂等按需建表（失败则整体降级，不阻断主链路）
    - save(): 保存一条体测记录
    - get_measurements(): 按运动员取历史记录（按日期倒序）
    - delete_all(): 清理某运动员的全部记录（供 seed --reset 使用）
    - rekey(): 游客认领账号时把记录改挂到新身份

    降级语义：与 TrainingLogStore 一致——存储不可用时 enabled=False，
    所有读写方法静默返回空值，由 API 层转成 503。
    """

    def __init__(self):
        self.db = PGClient()
        self.enabled = True
        self._table_ready = False

    # ------------------------------------------------------------------
    # 建表
    # ------------------------------------------------------------------

    def ensure_table(self) -> None:
        """按需建表。幂等；失败只记日志并降级。

        走 Base.metadata 而不是手写 DDL：表本身在 ORM 单点真相里，
        让 ORM 生成 DDL 可以避免「模型改了、建表语句忘了改」的漂移。
        """
        if self._table_ready:
            return
        try:
            from src.models.db_models import Base, BodyMeasurement
            Base.metadata.create_all(
                self.db.engine,
                tables=[BodyMeasurement.__table__],
                checkfirst=True,
            )
            self._table_ready = True
        except Exception as exc:
            logger.warning("体测记录表创建失败，体测功能降级禁用: %s", exc)
            self.enabled = False

    def _ready(self) -> bool:
        """确保表已就绪，返回存储是否可用。"""
        if not self._table_ready and self.enabled:
            self.ensure_table()
        return self._table_ready and self.enabled

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def save(self, athlete_key: str, measure_date: str | date | None = None,
             **fields) -> int | None:
        """保存一条体测记录。返回记录 id，存储不可用时返回 None。

        入参 fields 只接受表中真实存在的列，其余静默忽略——调用方传多了
        不该让整条记录写不进去，而拼 SQL 时把未知键塞进去会直接报语法错。
        """
        if not self._ready():
            return None

        allowed = ("weight", "body_fat_pct", "muscle_mass",
                   "chest", "waist", "hip", "arm", "thigh")
        values = {k: fields.get(k) for k in allowed}
        notes = (fields.get("notes") or "")[:512]
        resolved_date = self._parse_date(measure_date)

        with self.db.transaction() as tx:
            row = tx.fetch_one(
                """INSERT INTO body_measurements
                   (athlete_key, measure_date, weight, body_fat_pct, muscle_mass,
                    chest, waist, hip, arm, thigh, notes, created_at)
                   VALUES (:ak, :md, :wt, :bf, :mm, :ch, :wa, :hp, :ar, :th, :nt, :ts)
                   RETURNING id""",
                {"ak": athlete_key[:MAX_ATHLETE_KEY_LEN], "md": resolved_date,
                 "wt": values["weight"], "bf": values["body_fat_pct"],
                 "mm": values["muscle_mass"], "ch": values["chest"],
                 "wa": values["waist"], "hp": values["hip"],
                 "ar": values["arm"], "th": values["thigh"],
                 "nt": notes, "ts": datetime.utcnow()},
            )
            return row.id if row else None

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def get_measurements(self, athlete_key: str, weeks: int | None = None,
                         limit: int = MAX_ROWS_PER_QUERY) -> list[dict]:
        """取某运动员的历史体测记录，按测量日期倒序。

        输入：
            athlete_key: str — 稳定运动员标识
            weeks: int | None — 只取最近多少周；None 表示不限时间
            limit: int — 返回条数上限
        """
        if not self._ready():
            return []

        params = {"ak": athlete_key[:MAX_ATHLETE_KEY_LEN], "lim": limit}
        where = "athlete_key = :ak"
        if weeks is not None:
            since = self._parse_date(None) - timedelta(weeks=weeks)
            where += " AND measure_date >= :since"
            params["since"] = since

        rows = self.db.fetch_all(
            f"""SELECT id, athlete_key, measure_date, weight, body_fat_pct,
                       muscle_mass, chest, waist, hip, arm, thigh, notes, created_at
                FROM body_measurements
                WHERE {where}
                ORDER BY measure_date DESC, id DESC
                LIMIT :lim""",
            params,
        )
        return [
            {
                "id": r.id,
                "measure_date": r.measure_date.isoformat() if r.measure_date else None,
                "weight": r.weight,
                "body_fat_pct": r.body_fat_pct,
                "muscle_mass": r.muscle_mass,
                "chest": r.chest,
                "waist": r.waist,
                "hip": r.hip,
                "arm": r.arm,
                "thigh": r.thigh,
                "notes": r.notes or "",
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # 清理与迁移
    # ------------------------------------------------------------------

    def delete_one(self, athlete_key: str, measurement_id: int) -> bool:
        """删除某运动员的一条记录。返回是否确实删掉了一行。

        带上 athlete_key 一起过滤：只按 id 删会让任何人凭一个自增整数
        就能抹掉别人的记录，而自增 id 是可枚举的。
        """
        if not self._ready():
            return False

        with self.db.transaction() as tx:
            row = tx.fetch_one(
                """DELETE FROM body_measurements
                   WHERE id = :mid AND athlete_key = :ak
                   RETURNING id""",
                {"mid": measurement_id, "ak": athlete_key[:MAX_ATHLETE_KEY_LEN]},
            )
            return row is not None

    def delete_all(self, athlete_key: str) -> int:
        """删除某运动员的全部体测记录。返回删除条数。"""
        if not self._ready():
            return 0

        key = athlete_key[:MAX_ATHLETE_KEY_LEN]
        with self.db.transaction() as tx:
            row = tx.fetch_one(
                "SELECT COUNT(*) AS cnt FROM body_measurements WHERE athlete_key = :ak",
                {"ak": key},
            )
            count = int(row.cnt) if row else 0
            if count:
                tx.execute("DELETE FROM body_measurements WHERE athlete_key = :ak",
                           {"ak": key})
            return count

    def rekey(self, old_key: str, new_key: str) -> int:
        """把旧身份名下的体测记录改挂到新身份。返回迁移条数。

        用于「游客注册后认领本机数据」。漏掉这一步的表现是：用户注册登录后
        打开体测页，此前量过的所有数据都不见了——而训练日志却还在，
        因为那条链路单独做了迁移。
        """
        if not self._ready():
            return 0

        old_key = (old_key or "")[:MAX_ATHLETE_KEY_LEN]
        new_key = (new_key or "")[:MAX_ATHLETE_KEY_LEN]

        with self.db.transaction() as tx:
            row = tx.fetch_one(
                "SELECT COUNT(*) AS cnt FROM body_measurements WHERE athlete_key = :ak",
                {"ak": old_key},
            )
            count = int(row.cnt) if row else 0
            if count == 0:
                return 0
            tx.execute(
                "UPDATE body_measurements SET athlete_key = :new WHERE athlete_key = :old",
                {"new": new_key, "old": old_key})
            return count

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(value: str | date | None) -> date:
        """把入参统一成 date。缺省用本地当天。

        用 datetime.now() 而非 utcnow()：measure_date 的语义是
        "用户所在时区的日历日"，理由同 TrainingLogStore._parse_date。
        """
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str) and value.strip():
            try:
                return datetime.strptime(value.strip()[:10], "%Y-%m-%d").date()
            except ValueError:
                logger.warning("无法解析体测日期 %r，回退为今天", value)
        return datetime.now().date()
