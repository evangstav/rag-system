"""
Database models package.

Exports all SQLAlchemy models and base class.
"""

from app.models.database import (
    Base,
    User,
    Conversation,
    Message,
    MessageRole,
    ScratchpadEntry,
    ScratchpadEntryType,
    KnowledgePool,
    Document,
    DocumentStatus,
    UserMemory,
)

__all__ = [
    "Base",
    "User",
    "Conversation",
    "Message",
    "MessageRole",
    "ScratchpadEntry",
    "ScratchpadEntryType",
    "KnowledgePool",
    "Document",
    "DocumentStatus",
    "UserMemory",
]
