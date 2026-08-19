"""Deterministic, local-only routing for user fitness queries."""

from __future__ import annotations

import json
import re
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path


class QueryRoute(Enum):
    EXACT_ACTION = auto()
    KNOWLEDGE = auto()
    GRAPH = auto()
    INJURY_SENSITIVE = auto()
    FALLBACK = auto()


_MAX_QUERY_LENGTH = 1000
_INJURY_TERMS = (
    "疼",
    "痛",
    "受伤",
    "伤病",
    "损伤",
    "拉伤",
    "扭伤",
    "炎",
    "麻",
    "刺痛",
    "康复",
    "旧伤",
)
_RELATION_TERMS = (
    "哪些",
    "同时",
    "并且",
    "以及",
    "适合",
    "关联",
    "关系",
    "和.*的",
    ".*与.*",
)
_KNOWLEDGE_TERMS = (
    "如何",
    "怎么安排",
    "计划",
    "训练量",
    "频率",
    "原理",
    "区别",
    "营养",
    "恢复",
    "热身",
)


@lru_cache(maxsize=1)
def _exercise_names() -> frozenset[str]:
    fixture = Path(__file__).resolve().parents[2] / "data" / "seed_exercises.json"
    try:
        with fixture.open(encoding="utf-8") as stream:
            records = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return frozenset()
    return frozenset(
        name.strip() for record in records if isinstance(record, dict)
        for name in (record.get("name"),) if isinstance(name, str) and name.strip()
    )


def _contains_action(query: str) -> bool:
    normalized = re.sub(r"\s+", "", query).casefold()
    return any(name.casefold() in normalized for name in _exercise_names())


def classify_query(query: str, profile: dict | None = None) -> QueryRoute:
    """Classify a query without LLM, database, filesystem writes, or network calls."""
    del profile  # Reserved for future deterministic, user-profile-aware rules.
    if not isinstance(query, str):
        return QueryRoute.FALLBACK
    query = query.strip()
    if not query or len(query) > _MAX_QUERY_LENGTH:
        return QueryRoute.FALLBACK
    if any(term in query for term in _INJURY_TERMS):
        return QueryRoute.INJURY_SENSITIVE
    if _contains_action(query) and not any(
        re.search(term, query) for term in _RELATION_TERMS
    ):
        return QueryRoute.EXACT_ACTION
    if any(re.search(term, query) for term in _RELATION_TERMS):
        return QueryRoute.GRAPH
    if any(term in query for term in _KNOWLEDGE_TERMS):
        return QueryRoute.KNOWLEDGE
    return QueryRoute.FALLBACK
