"""enforce not null on training_plans owner_key

修正一处「模型与库不一致」的漂移。

来由：ORM 里 `TrainingPlan.owner_key` 声明为 `nullable=False`，但既有库
里该列是可空的。原因是建表走的是 `init_db()` 的 `Base.metadata.create_all`——
它只创建不存在的表，**不会修改已有表的约束**，所以模型后来收紧的那次改动
从未落到库上。这类漂移不会报错，只会在将来某次 autogenerate 里冒出来，
表现为一条与本次改动毫不相干的 alter。

安全性：执行前已核对生产库 21 行 training_plans 中 owner_key 无一为空，
加约束不会失败。若在别的环境上该列存在空值，此迁移会报错中止——
这是有意的：与其静默跳过，不如暴露出来，因为空 owner_key 意味着
一行计划不属于任何人，本身就需要先澄清。

Revision ID: abfe607e0be0
Revises: ee68ef26c164
Create Date: 2026-10-14 12:16:00.700805

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'abfe607e0be0'
down_revision: Union[str, Sequence[str], None] = 'ee68ef26c164'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column('training_plans', 'owner_key',
                    existing_type=sa.VARCHAR(length=64),
                    nullable=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column('training_plans', 'owner_key',
                    existing_type=sa.VARCHAR(length=64),
                    nullable=True)
