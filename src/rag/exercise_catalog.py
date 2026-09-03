"""Exercise name catalog — data-driven replacement for hardcoded name lists.

The QA chain needs to anchor a user question to a concrete exercise (e.g. for
GraphRAG pain reasoning). The original implementation matched against ~40
hardcoded names; anything outside that list was invisible. This module loads
the full name list from the real exercise library instead:

1. PostgreSQL ``exercises`` table when reachable (the live library, 338+ names);
2. otherwise ``data/seed_exercises.json`` (the seed corpus shipped in the repo);
3. otherwise a small built-in fallback list.

Names are deduplicated and ordered longest-first so that "杠铃深蹲" matches
before "深蹲". The loaded list is cached for the process lifetime.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_SEED_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "seed_exercises.json"

# Last-resort list — only used when both PG and the seed file are unavailable.
_BUILTIN_FALLBACK = [
    "深蹲", "硬拉", "卧推", "推举", "划船", "弯举", "引体向上", "高位下拉",
    "保加利亚分腿蹲", "罗马尼亚硬拉", "平板支撑", "箭步蹲",
]


@lru_cache(maxsize=1)
def load_exercise_names() -> tuple[str, ...]:
    """Load all exercise names, longest-first, deduplicated."""
    names = _names_from_pg() or _names_from_seed() or list(_BUILTIN_FALLBACK)
    ordered = sorted({n for n in names if n}, key=len, reverse=True)
    return tuple(ordered)


def _names_from_pg() -> list[str]:
    try:
        from src.storage.pg import PGClient
        rows = PGClient().fetch_all("SELECT name FROM exercises")
        names = [r[0] for r in rows if r and r[0]]
        if names:
            return names
    except Exception as e:  # PG offline → fall through to the seed file
        logger.debug(f"exercise catalog: PG unavailable ({e}), trying seed file")
    return []


def _names_from_seed() -> list[str]:
    try:
        data = json.loads(_SEED_PATH.read_text(encoding="utf-8"))
        return [item.get("name", "") for item in data if item.get("name")]
    except Exception as e:
        logger.debug(f"exercise catalog: seed file unreadable ({e})")
        return []


def extract_exercise_name(text: str) -> str | None:
    """Return the longest exercise name appearing in ``text``, or None."""
    for name in load_exercise_names():
        if name in text:
            return name
    return None
