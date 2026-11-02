"""add sets_detail to training log entries

Revision ID: c1d2e3f4a5b6
Revises: abfe607e0be0
Create Date: 2026-11-02

为什么加这一列：汇总字段（sets/reps/weight）只能表达"这个动作做了 4 组、
代表重量 50kg"，而递增组、递减组、组间重量变化都记不下来。逐组明细
放在 JSON 列里，不改变既有统计口径，也不需要为每组建一行。
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c1d2e3f4a5b6"
down_revision: Union[str, Sequence[str], None] = "abfe607e0be0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "training_log_entries",
        sa.Column("sets_detail", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("training_log_entries", "sets_detail")
