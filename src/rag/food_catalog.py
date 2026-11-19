"""食物营养目录 —— 饮食计划与饮食记录里一切数字的唯一来源。

立项原则与 ``src/core/nutrition.py`` 同源：**数字结论不交给模型**。LLM 只负责
「选哪样食物、吃多少克」，每食物/每餐/每日的热量与宏量一律在这里按克数确定性
重算。模型报来的任何热量数字都会被丢弃。

数据源 ``data/seed_foods.json``（公开成分表估算值，按可食部每 100g）。与动作库
不同，食物没有现成的 PG 表，所以 v1 以 seed 为单源；将来若建 ``foods`` 表，照
``exercise_catalog`` 的 PG 优先层加一个 ``_foods_from_pg`` 即可，接口不变。

三个必须知道的口径：

1. **生熟状态是数字的一部分**。米饭（蒸）116 kcal/100g 与大米（生）346 差三倍，
   所以 name 带状态、``state`` 字段独立成列；克数一律按条目状态的可食部计。
2. **纤维参与能量核对**。成分表的能量含纤维发酵产能，Atwater 核对用
   ``4P + 9F + 4C - 2×fiber``（纤维按 2 kcal/g），CI 测试逐条锁 ≤10% 偏差。
3. **缓存对象不可就地修改**。``load_foods()`` 返回 lru_cache 的同一个 tuple；
   对外取单条一律走 ``get_food`` / ``find_substitutes``（返回副本）。
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_SEED_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "seed_foods.json"

# 封闭过敏原词表：seed 与用户输入都只允许这十个标签，新增标签必须先改这里，
# 否则过敏冲突表会出现「词表里有、食物上没标」的静默漏拦。
FOOD_ALLERGENS = frozenset(
    {"麸质", "花生", "坚果", "大豆", "蛋", "乳", "鱼", "虾蟹", "贝类", "芝麻"}
)
FOOD_CATEGORIES = frozenset({"主食", "蛋白", "蔬菜", "水果", "脂肪", "乳品"})
FOOD_FLAGS = ("high_gi", "high_purine", "high_sodium", "raw_or_high_mercury")
_FOOD_STATES = frozenset({"生", "熟", "干"})

# 最后兜底：seed 文件也读不到时（打包遗漏/损坏）仍能让链路跑通的最小集合。
_BUILTIN_FALLBACK = [
    {"name": "米饭（蒸）", "state": "熟", "category": "主食", "kcal_per_100g": 116,
     "protein_g": 2.6, "fat_g": 0.3, "carbs_g": 25.9, "fiber_g": 0.3,
     "high_gi": True, "default_serving_g": 150},
    {"name": "鸡胸肉（煮）", "state": "熟", "category": "蛋白", "kcal_per_100g": 150,
     "protein_g": 28.0, "fat_g": 3.5, "carbs_g": 0.0, "default_serving_g": 120},
    {"name": "鸡蛋（煮）", "state": "熟", "category": "蛋白", "kcal_per_100g": 144,
     "protein_g": 13.3, "fat_g": 8.8, "carbs_g": 2.8, "allergens": ["蛋"],
     "default_serving_g": 50},
    {"name": "西兰花", "state": "生", "category": "蔬菜", "kcal_per_100g": 36,
     "protein_g": 3.6, "fat_g": 0.6, "carbs_g": 4.3, "fiber_g": 1.6,
     "default_serving_g": 150},
    {"name": "苹果", "state": "生", "category": "水果", "kcal_per_100g": 54,
     "protein_g": 0.2, "fat_g": 0.2, "carbs_g": 13.5, "fiber_g": 1.2,
     "default_serving_g": 200},
    {"name": "牛奶（全脂）", "state": "熟", "category": "乳品", "kcal_per_100g": 54,
     "protein_g": 3.0, "fat_g": 3.2, "carbs_g": 3.4, "allergens": ["乳"],
     "default_serving_g": 250},
    {"name": "橄榄油", "state": "生", "category": "脂肪", "kcal_per_100g": 884,
     "protein_g": 0.0, "fat_g": 98.2, "carbs_g": 0.0, "default_serving_g": 10},
]


def _clean_entry(item: dict) -> dict | None:
    """补齐缺省字段；名字或热量缺失的条目直接丢弃（宁缺毋错）。"""
    name = (item.get("name") or "").strip()
    kcal = item.get("kcal_per_100g")
    if not name or not isinstance(kcal, (int, float)) or kcal <= 0:
        return None
    entry = {
        "name": name,
        "aliases": [a.strip() for a in (item.get("aliases") or []) if a and a.strip()],
        "state": item.get("state") or "熟",
        "category": item.get("category") or "主食",
        "kcal_per_100g": float(kcal),
        "protein_g": float(item.get("protein_g") or 0.0),
        "fat_g": float(item.get("fat_g") or 0.0),
        "carbs_g": float(item.get("carbs_g") or 0.0),
        "fiber_g": float(item.get("fiber_g") or 0.0),
        "allergens": [a for a in (item.get("allergens") or []) if a in FOOD_ALLERGENS],
        "default_serving_g": float(item.get("default_serving_g") or 100),
    }
    for flag in FOOD_FLAGS:
        entry[flag] = bool(item.get(flag))
    return entry


def _foods_from_seed() -> list[dict]:
    try:
        payload = json.loads(_SEED_PATH.read_text(encoding="utf-8"))
    except Exception as e:  # seed 不可读 → 内置兜底，不让饮食链路整体瘫痪
        logger.debug(f"food catalog: seed file unreadable ({e})")
        return []
    items = payload.get("foods") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    cleaned = [_clean_entry(it) for it in items if isinstance(it, dict)]
    return [e for e in cleaned if e]


@lru_cache(maxsize=1)
def load_foods() -> tuple[dict, ...]:
    """全部食物条目（按名字去重，保留首次出现）。

    返回值**不要就地修改**——lru_cache 缓存的同一个对象。取单条请用
    ``get_food``（返回副本）。
    """
    foods = _foods_from_seed() or [
        e for e in (_clean_entry(it) for it in _BUILTIN_FALLBACK) if e
    ]
    seen: set[str] = set()
    unique = []
    for entry in foods:
        if entry["name"] in seen:
            continue
        seen.add(entry["name"])
        unique.append(entry)
    return tuple(unique)


@lru_cache(maxsize=1)
def _name_index() -> dict[str, dict]:
    return {e["name"]: e for e in load_foods()}


@lru_cache(maxsize=1)
def _alias_map() -> dict[str, str]:
    """别名 → 规范名。别名不允许遮蔽另一个规范名（测试锁死）。"""
    mapping = {e["name"]: e["name"] for e in load_foods()}
    for entry in load_foods():
        for alias in entry["aliases"]:
            mapping.setdefault(alias, entry["name"])
    return mapping


def load_food_names() -> tuple[str, ...]:
    """全部规范名，最长优先（「鸡胸肉（煮）」先于「鸡胸」匹配）。"""
    return tuple(sorted((e["name"] for e in load_foods()), key=len, reverse=True))


def normalize_food_name(name: str) -> str:
    """去空白 + 别名归一；认不出时原样返回（由 get_food 判 None）。"""
    cleaned = (name or "").strip()
    return _alias_map().get(cleaned, cleaned)


def get_food(name: str) -> dict | None:
    """按名字取食物条目（副本）。

    精确/别名优先；否则做前缀匹配（「鸡胸」→「鸡胸肉（煮）」）。前缀命中多条时
    **优先熟态**——人吃的是做熟的食物，生态条目留给备餐计算；再按名字排序保证
    确定性。认不出返回 None，调用方据此走「库外食物」处置。
    """
    cleaned = normalize_food_name(name)
    if not cleaned:
        return None
    index = _name_index()
    hit = index.get(cleaned)
    if hit is not None:
        return dict(hit)
    candidates = [e for n, e in index.items() if n.startswith(cleaned) or cleaned.startswith(n)]
    if not candidates:
        return None
    candidates.sort(key=lambda e: (e["state"] != "熟", e["name"]))
    return dict(candidates[0])


def food_nutrients(name: str, grams: float) -> dict | None:
    """按克数确定性重算一份食物的能量与宏量。认不出食物返回 None。

    这是「数字不交给模型」的落点：writer 与饮食记录写入都经此函数，
    模型或用户报来的热量数字一律不被采信。
    """
    entry = get_food(name)
    if entry is None or grams is None:
        return None
    try:
        grams = float(grams)
    except (TypeError, ValueError):
        return None
    if grams < 0:
        return None
    factor = grams / 100.0
    return {
        "name": entry["name"],
        "grams": round(grams, 1),
        "kcal": round(entry["kcal_per_100g"] * factor),
        "protein_g": round(entry["protein_g"] * factor, 1),
        "fat_g": round(entry["fat_g"] * factor, 1),
        "carbs_g": round(entry["carbs_g"] * factor, 1),
    }


def find_substitutes(
    name: str,
    *,
    exclude_allergens: tuple[str, ...] | list[str] = (),
    exclude_flags: tuple[str, ...] | list[str] = (),
    limit: int = 5,
) -> list[dict]:
    """同类食物里找不含指定过敏原/标记的替代候选（副本，按宏量接近度排序）。

    过敏冲突的确定性替换走这里：先换菜再考虑 LLM 重写，多数过敏场景因此
    只需要单轮 LLM。源食物认不出时返回空列表（调用方另行处置库外食物）。
    """
    src = get_food(name)
    if src is None:
        return []
    banned_allergens = set(exclude_allergens)
    banned_flags = set(exclude_flags)
    primary = "protein_g" if src["category"] in ("蛋白", "乳品") else (
        "carbs_g" if src["category"] == "主食" else "kcal_per_100g"
    )

    def distance(entry: dict) -> tuple[float, str]:
        return (
            abs(entry[primary] - src[primary]) + 0.1 * abs(entry["kcal_per_100g"] - src["kcal_per_100g"]),
            entry["name"],
        )

    candidates = [
        e for e in load_foods()
        if e["name"] != src["name"]
        and e["category"] == src["category"]
        and not (set(e["allergens"]) & banned_allergens)
        and not any(e.get(flag) for flag in banned_flags)
    ]
    candidates.sort(key=distance)
    return [dict(e) for e in candidates[: max(0, int(limit))]]