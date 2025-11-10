"""Add indexes to user_memories for improved query performance

Revision ID: e8c9f1234abc
Revises: d3fb5a9677e5
Create Date: 2025-11-09 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e8c9f1234abc'
down_revision: Union[str, None] = 'd3fb5a9677e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add composite index on user_id and importance for efficient retrieval
    op.create_index(
        'idx_user_memories_user_importance',
        'user_memories',
        ['user_id', 'importance'],
        unique=False
    )

    # Add composite index on user_id and updated_at for recency-based queries
    op.create_index(
        'idx_user_memories_user_updated',
        'user_memories',
        ['user_id', 'updated_at'],
        unique=False
    )

    # Add index on qdrant_id for faster lookups when retrieving from vector store
    op.create_index(
        'idx_user_memories_qdrant_id',
        'user_memories',
        ['qdrant_id'],
        unique=False
    )


def downgrade() -> None:
    # Remove indexes in reverse order
    op.drop_index('idx_user_memories_qdrant_id', table_name='user_memories')
    op.drop_index('idx_user_memories_user_updated', table_name='user_memories')
    op.drop_index('idx_user_memories_user_importance', table_name='user_memories')
