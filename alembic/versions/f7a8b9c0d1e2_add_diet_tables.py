"""add diet plans and diet logs tables

Revision ID: f7a8b9c0d1e2
Revises: c1d2e3f4a5b6
Create Date: 2026-11-19 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f7a8b9c0d1e2'
down_revision: Union[str, Sequence[str], None] = 'c1d2e3f4a5b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'diet_plans',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('owner_key', sa.String(length=64), nullable=False),
        sa.Column('goal', sa.String(length=20), nullable=True),
        sa.Column('plan_data', sa.JSON(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_diet_plans_owner_key', 'diet_plans', ['owner_key'], unique=False)
    op.create_table(
        'diet_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('owner_key', sa.String(length=64), nullable=False),
        sa.Column('log_date', sa.Date(), nullable=False),
        sa.Column('meal_type', sa.String(length=16), nullable=False),
        sa.Column('foods', sa.JSON(), nullable=False),
        sa.Column('plan_id', sa.Integer(), nullable=True),
        sa.Column('notes', sa.String(length=512), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('owner_key', 'log_date', 'meal_type',
                            name='uq_diet_logs_owner_date_meal'),
    )
    op.create_index('ix_diet_logs_owner_key', 'diet_logs', ['owner_key'], unique=False)
    op.create_index('ix_diet_logs_owner_date', 'diet_logs', ['owner_key', 'log_date'],
                    unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_diet_logs_owner_date', table_name='diet_logs')
    op.drop_index('ix_diet_logs_owner_key', table_name='diet_logs')
    op.drop_table('diet_logs')
    op.drop_index('ix_diet_plans_owner_key', table_name='diet_plans')
    op.drop_table('diet_plans')