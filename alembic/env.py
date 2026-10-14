"""Alembic 迁移环境配置。

数据库连接从 `src.config` 读取（即 `.env`），不写进 `alembic.ini`——
那个文件会进版本控制，把口令写进去等于把凭据提交到仓库。

用法：
    alembic upgrade head          # 全新库：建出全部表
    alembic stamp head            # 已有数据的库：只登记版本，不执行 DDL
    alembic revision --autogenerate -m "描述"   # 改完模型后生成增量迁移

`stamp` 这一步容易被漏掉：既有部署的库是靠 `init_db()` 的 create_all 建起来的，
里面没有 alembic_version 表。此时直接 upgrade 会从零开始建表并因"表已存在"失败。
"""
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import create_engine, pool

from alembic import context

# 保证能 import src.*。alembic 通常在项目根执行，但显式声明不依赖 cwd。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATABASE_URL  # noqa: E402
from src.models.db_models import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _include_object(obj, name, type_, reflected, compare_to):
    """把 alembic 自己的版本表排除在比对之外。

    不排除的话，autogenerate 会把 alembic_version 当成"模型里没有的表"，
    于是每次生成迁移都附赠一条 drop_table。
    """
    if type_ == "table" and name == "alembic_version":
        return False
    return True


def run_migrations_offline() -> None:
    """离线模式：只输出 SQL，不连数据库。"""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=_include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：连库执行迁移。"""
    connectable = create_engine(DATABASE_URL, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=_include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
