"""Add word_count and page_count to documents

Revision ID: e8a7b2c4f1d9
Revises: d3fb5a9677e5
Create Date: 2025-11-07 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e8a7b2c4f1d9'
down_revision: Union[str, None] = 'd3fb5a9677e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add word_count and page_count columns to documents table
    op.add_column('documents', sa.Column('word_count', sa.Integer(), nullable=True))
    op.add_column('documents', sa.Column('page_count', sa.Integer(), nullable=True))


def downgrade() -> None:
    # Remove word_count and page_count columns from documents table
    op.drop_column('documents', 'page_count')
    op.drop_column('documents', 'word_count')
