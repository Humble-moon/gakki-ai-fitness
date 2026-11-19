"""饮食记录的存储层 — diet_logs 表的读写（一餐一条，upsert 语义）。

与训练日志主从两表的偏离是有意为之（见 db_models.DietLog docstring）：
餐是原子单位，foods JSON 单列原子写入。重复记同一餐＝覆盖——非技术用户
的直觉是「改」不是「加」，UNIQUE(owner_key, log_date, meal_type) 承载它。

数字口径：catalog 来源的食物一律按 food_catalog 重算后入库（用户/模型报的
数字不被采信）；custom 来源（外卖/食堂手动估算）保留用户值但做防呆钳制，
并在条目上标 source="custom"，统计层单列估算占比。
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime

from src.rag import food_catalog
from src.storage.pg import PGClient

logger = logging.getLogger(__name__)

MAX_OWNER_KEY_LEN = 64
MAX_LOGS_PER_QUERY = 500
MEAL_TYPES = ("早餐", "午餐", "晚餐", "加餐")

# custom 防呆钳制：单食物热量上限与单餐合计上限。手动估算错一个数量级
# （把 kcal 写成 kJ、把整份写成每 100g）不该污染整周统计。
MAX_CUSTOM_FOOD_KCAL = 2000
MAX_MEAL_KCAL = 3000


class DietLogStore:
    """diet_logs 表的读写。降级语义与 PlanStore 一致。"""

    def __init__(self):
        self.db = PGClient()
        self.enabled = True
        self._table_ready = False

    def ensure_table(self) -> None:
        if self._table_ready:
            return
        try:
            from src.models.db_models import Base, DietLog
            Base.metadata.create_all(self.db.engine, tables=[DietLog.__table__],
                                     checkfirst=True)
            self._table_ready = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("饮食记录表创建失败，饮食记录功能降级禁用: %s", exc)
            self.enabled = False

    def _ready(self) -> bool:
        if not self._table_ready and self.enabled:
            self.ensure_table()
        return self._table_ready and self.enabled

    # ------------------------------------------------------------------
    # 写入（upsert）
    # ------------------------------------------------------------------
    def upsert_meal(self, owner_key: str, log_date, meal_type: str,
                    foods: list[dict], *, plan_id: int | None = None,
                    notes: str = "") -> int | None:
        """记一餐；同身份同日同餐已存在则覆盖。返回行 id。

        ``foods`` 条目 ``{name, grams}``（catalog）或
        ``{name, grams, kcal, protein_g, source:"custom"}``（手动）。
        catalog 条目营养由服务端重算；库外名字拒收（返回 None 交 API 报 400），
        绝不入库一行没有营养依据的记录。
        """
        if not self._ready():
            return None
        owner = (owner_key or "")[:MAX_OWNER_KEY_LEN]
        if not owner or meal_type not in MEAL_TYPES:
            return None
        if isinstance(log_date, str):
            try:
                log_date = date.fromisoformat(log_date)
            except ValueError:
                return None
        if not isinstance(log_date, date):
            return None

        cleaned = self._clean_foods(foods)
        if cleaned is None:
            return None
        with self.db.transaction() as tx:
            row = tx.fetch_one(
                """INSERT INTO diet_logs
                   (owner_key, log_date, meal_type, foods, plan_id, notes,
                    created_at, updated_at)
                   VALUES (:ok, :d, :m, :foods, :pid, :notes, :ts, :ts)
                   ON CONFLICT ON CONSTRAINT uq_diet_logs_owner_date_meal
                   DO UPDATE SET foods = :foods, plan_id = :pid, notes = :notes,
                                 updated_at = :ts
                   RETURNING id""",
                {"ok": owner, "d": log_date, "m": meal_type,
                 "foods": json.dumps(cleaned, ensure_ascii=False),
                 "pid": plan_id, "notes": (notes or "")[:512],
                 "ts": datetime.utcnow()},
            )
            return int(row.id) if row else None

    def _clean_foods(self, foods) -> list[dict] | None:
        if not isinstance(foods, list) or not foods:
            return None
        cleaned = []
        total_kcal = 0.0
        for item in foods:
            if not isinstance(item, dict):
                return None
            name = str(item.get("name") or "").strip()
            try:
                grams = float(item.get("grams") or 0)
            except (TypeError, ValueError):
                return None
            if not name or grams <= 0 or grams > 2000:
                return None
            if item.get("source") == "custom":
                try:
                    kcal = float(item.get("kcal") or 0)
                    protein = float(item.get("protein_g") or 0)
                except (TypeError, ValueError):
                    return None
                if kcal < 0 or protein < 0:
                    return None
                kcal = min(kcal, MAX_CUSTOM_FOOD_KCAL)
                entry = {"name": name[:60], "grams": round(grams, 1),
                         "kcal": round(kcal), "protein_g": round(protein, 1),
                         "fat_g": 0.0, "carbs_g": 0.0, "source": "custom"}
            else:
                nutrients = food_catalog.food_nutrients(name, grams)
                if nutrients is None:
                    return None  # 库外拒收：不入库无营养依据的记录
                entry = dict(nutrients, source="catalog")
            total_kcal += entry["kcal"]
            cleaned.append(entry)
        if total_kcal > MAX_MEAL_KCAL:
            return None
        return cleaned

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def get_logs(self, owner_key: str, since=None, limit: int = 200) -> list[dict]:
        if not self._ready():
            return []
        params = {"ok": (owner_key or "")[:MAX_OWNER_KEY_LEN],
                  "lim": max(1, min(int(limit), MAX_LOGS_PER_QUERY))}
        where = "owner_key = :ok"
        if since is not None:
            where += " AND log_date >= :since"
            params["since"] = since
        rows = self.db.fetch_all(
            f"""SELECT id, log_date, meal_type, foods, plan_id, notes, created_at, updated_at
                FROM diet_logs WHERE {where}
                ORDER BY log_date DESC, meal_type LIMIT :lim""",
            params,
        )
        out = []
        for r in rows:
            foods = r.foods if isinstance(r.foods, list) else json.loads(r.foods or "[]")
            out.append({
                "log_id": r.id,
                "log_date": r.log_date.isoformat() if r.log_date else "",
                "meal_type": r.meal_type,
                "foods": foods,
                "plan_id": r.plan_id,
                "notes": r.notes or "",
                "kcal": round(sum(f.get("kcal") or 0 for f in foods)),
                "protein_g": round(sum(f.get("protein_g") or 0 for f in foods), 1),
                "custom_kcal": round(sum(f.get("kcal") or 0 for f in foods
                                         if f.get("source") == "custom")),
                "updated_at": r.updated_at.isoformat() if r.updated_at else "",
            })
        return out

    def delete_log(self, owner_key: str, log_id: int) -> bool:
        if not self._ready():
            return False
        owner = (owner_key or "")[:MAX_OWNER_KEY_LEN]
        with self.db.transaction() as tx:
            row = tx.fetch_one(
                "SELECT COUNT(*) AS c FROM diet_logs WHERE id = :lid AND owner_key = :ok",
                {"lid": int(log_id), "ok": owner})
            if not row or int(row.c) == 0:
                return False
            tx.execute("DELETE FROM diet_logs WHERE id = :lid AND owner_key = :ok",
                       {"lid": int(log_id), "ok": owner})
            return True

    def delete_all(self, owner_key: str) -> int:
        if not self._ready():
            return 0
        owner = (owner_key or "")[:MAX_OWNER_KEY_LEN]
        if not owner:
            return 0
        with self.db.transaction() as tx:
            row = tx.fetch_one("SELECT COUNT(*) AS c FROM diet_logs WHERE owner_key = :ok",
                               {"ok": owner})
            count = int(row.c) if row else 0
            if count == 0:
                return 0
            tx.execute("DELETE FROM diet_logs WHERE owner_key = :ok", {"ok": owner})
            return count

    def rekey(self, old_key: str, new_key: str) -> int:
        """游客注册认领。UNIQUE(owner,date,meal) 存在：新身份已有同餐记录时
        保留新身份自己的那份（NOT EXISTS 守卫），不让认领覆盖已有数据。"""
        if not self._ready():
            return 0
        old_key = (old_key or "")[:MAX_OWNER_KEY_LEN]
        new_key = (new_key or "")[:MAX_OWNER_KEY_LEN]
        if not old_key or not new_key or old_key == new_key:
            return 0
        guard = """NOT EXISTS (SELECT 1 FROM diet_logs d2
                   WHERE d2.owner_key = :new AND d2.log_date = diet_logs.log_date
                     AND d2.meal_type = diet_logs.meal_type)"""
        with self.db.transaction() as tx:
            row = tx.fetch_one(
                f"SELECT COUNT(*) AS c FROM diet_logs WHERE owner_key = :old AND {guard}",
                {"old": old_key, "new": new_key})
            count = int(row.c) if row else 0
            if count:
                tx.execute(
                    f"UPDATE diet_logs SET owner_key = :new WHERE owner_key = :old AND {guard}",
                    {"new": new_key, "old": old_key})
            return count