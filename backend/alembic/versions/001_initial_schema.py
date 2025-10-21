"""initial_schema

Revision ID: 001
Revises:
Create Date: 2025-10-21

Initial database schema with all core tables:
- users
- conversations
- messages
- scratchpad_entries
- knowledge_pools
- documents
- user_memories
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables for the RAG system."""

    # Create enum types
    op.execute("CREATE TYPE message_role AS ENUM ('user', 'assistant', 'system')")
    op.execute("CREATE TYPE scratchpad_entry_type AS ENUM ('todo', 'note', 'journal')")
    op.execute("CREATE TYPE document_status AS ENUM ('pending', 'processing', 'completed', 'failed')")

    # Users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('email', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('username', sa.String(100), unique=True, nullable=False, index=True),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_superuser', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()'), onupdate=sa.text('NOW()')),
    )

    # Conversations table
    op.create_table(
        'conversations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('title', sa.String(500), nullable=True),
        sa.Column('use_rag', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('use_scratchpad', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
    )

    # Messages table
    op.create_table(
        'messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('conversation_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('role', sa.Enum('user', 'assistant', 'system', name='message_role'), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
    )

    # Create index on messages for efficient conversation history queries
    op.create_index('idx_messages_conversation_created', 'messages', ['conversation_id', 'created_at'])

    # Scratchpad entries table
    op.create_table(
        'scratchpad_entries',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('entry_type', sa.Enum('todo', 'note', 'journal', name='scratchpad_entry_type'), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('is_completed', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('entry_date', sa.DateTime(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
    )

    # Index for efficient scratchpad queries
    op.create_index('idx_scratchpad_user_type', 'scratchpad_entries', ['user_id', 'entry_type', 'created_at'])

    # Knowledge pools table
    op.create_table(
        'knowledge_pools',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('collection_name', sa.String(300), nullable=False, unique=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
    )

    # Documents table
    op.create_table(
        'documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('knowledge_pool_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('knowledge_pools.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('filename', sa.String(500), nullable=False),
        sa.Column('file_path', sa.String(1000), nullable=True),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('mime_type', sa.String(100), nullable=True),
        sa.Column('source_type', sa.String(50), nullable=False, server_default='upload'),
        sa.Column('source_url', sa.String(2000), nullable=True),
        sa.Column('status', sa.Enum('pending', 'processing', 'completed', 'failed', name='document_status'), nullable=False, server_default='pending'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('num_chunks', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('num_tokens', sa.Integer(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
    )

    # User memories table
    op.create_table(
        'user_memories',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('source_conversation_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('conversations.id', ondelete='SET NULL'), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('importance', sa.Float(), nullable=False, server_default='0.5'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
    )

    # Index for efficient memory retrieval sorted by importance
    op.create_index('idx_user_memories_importance', 'user_memories', ['user_id', 'importance'], postgresql_ops={'importance': 'DESC'})


def downgrade() -> None:
    """Drop all tables in reverse order."""

    # Drop tables in reverse order (respecting foreign key constraints)
    op.drop_table('user_memories')
    op.drop_table('documents')
    op.drop_table('knowledge_pools')
    op.drop_table('scratchpad_entries')
    op.drop_table('messages')
    op.drop_table('conversations')
    op.drop_table('users')

    # Drop enum types
    op.execute('DROP TYPE IF EXISTS document_status')
    op.execute('DROP TYPE IF EXISTS scratchpad_entry_type')
    op.execute('DROP TYPE IF EXISTS message_role')
