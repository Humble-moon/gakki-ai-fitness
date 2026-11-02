"""启动期数据库迁移。

应用自己跑 ``alembic upgrade head``，而不是依赖部署脚本里的一句手工命令：
迁移与代码是同一个版本单元，漏跑一次就会让新代码撞上旧表，而报错往往
发生在业务路径深处，很难一眼看出是"忘了迁移"。

失败时抛异常、拒绝启动——带着不匹配的 schema 提供服务，比启动失败更糟。
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def upgrade_to_head() -> None:
    from alembic import command
    from alembic.config import Config as AlembicConfig

    root = Path(__file__).resolve().parents[1]
    cfg = AlembicConfig(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    logger.info("[migrate] upgrading database to head")
    command.upgrade(cfg, "head")
    logger.info("[migrate] database is at head")


__all__ = ["upgrade_to_head"]
