"""
Pydantic schemas for request/response validation.

These are separate from SQLAlchemy models to provide clean API contracts.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field


# ============================================================================
# User Schemas
# ============================================================================

class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)


class UserCreate(UserBase):
    """Schema for creating a new user."""
    password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    """Schema for updating user information."""
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=100)
    password: Optional[str] = Field(None, min_length=8)


class UserResponse(UserBase):
    """Schema for user responses."""
    id: UUID
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ============================================================================
# Conversation Schemas
# ============================================================================

class ConversationBase(BaseModel):
    """Base conversation schema."""
    title: Optional[str] = None
    use_rag: bool = False
    use_scratchpad: bool = False


class ConversationCreate(ConversationBase):
    """Schema for creating a conversation."""
    pass


class ConversationUpdate(BaseModel):
    """Schema for updating a conversation."""
    title: Optional[str] = None
    use_rag: Optional[bool] = None
    use_scratchpad: Optional[bool] = None


class ConversationResponse(ConversationBase):
    """Schema for conversation responses."""
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ============================================================================
# Message Schemas
# ============================================================================

class MessageBase(BaseModel):
    """Base message schema."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class MessageCreate(MessageBase):
    """Schema for creating a message."""
    conversation_id: UUID
    metadata: Optional[dict] = None


class MessageResponse(MessageBase):
    """Schema for message responses."""
    id: UUID
    conversation_id: UUID
    metadata: Optional[dict] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ============================================================================
# Scratchpad Schemas
# ============================================================================

class ScratchpadEntryBase(BaseModel):
    """Base scratchpad entry schema."""
    entry_type: str = Field(..., pattern="^(todo|note|journal)$")
    content: str
    is_completed: bool = False
    entry_date: Optional[datetime] = None
    metadata: Optional[dict] = None


class ScratchpadEntryCreate(ScratchpadEntryBase):
    """Schema for creating a scratchpad entry."""
    pass


class ScratchpadEntryUpdate(BaseModel):
    """Schema for updating a scratchpad entry."""
    content: Optional[str] = None
    is_completed: Optional[bool] = None
    entry_date: Optional[datetime] = None
    metadata: Optional[dict] = None


class ScratchpadEntryResponse(ScratchpadEntryBase):
    """Schema for scratchpad entry responses."""
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# Legacy scratchpad schema for backward compatibility with current API
class TodoItem(BaseModel):
    """Individual todo item."""
    id: str
    text: str
    completed: bool


class ScratchpadData(BaseModel):
    """Complete scratchpad data structure (legacy format)."""
    todos: List[TodoItem] = []
    notes: str = ""
    journal: str = ""


# ============================================================================
# Knowledge Pool Schemas
# ============================================================================

class KnowledgePoolBase(BaseModel):
    """Base knowledge pool schema."""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None


class KnowledgePoolCreate(KnowledgePoolBase):
    """Schema for creating a knowledge pool."""
    pass


class KnowledgePoolUpdate(BaseModel):
    """Schema for updating a knowledge pool."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None


class KnowledgePoolResponse(KnowledgePoolBase):
    """Schema for knowledge pool responses."""
    id: UUID
    user_id: UUID
    collection_name: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ============================================================================
# Document Schemas
# ============================================================================

class DocumentBase(BaseModel):
    """Base document schema."""
    filename: str
    source_type: str = "upload"
    source_url: Optional[str] = None


class DocumentCreate(DocumentBase):
    """Schema for creating a document."""
    knowledge_pool_id: UUID
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    metadata: Optional[dict] = None


class DocumentUpdate(BaseModel):
    """Schema for updating a document."""
    filename: Optional[str] = None
    status: Optional[str] = Field(None, pattern="^(pending|processing|completed|failed)$")
    error_message: Optional[str] = None
    metadata: Optional[dict] = None


class DocumentResponse(DocumentBase):
    """Schema for document responses."""
    id: UUID
    knowledge_pool_id: UUID
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    num_chunks: int
    num_tokens: Optional[int] = None
    metadata: Optional[dict] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ============================================================================
# RAG Schemas
# ============================================================================

class RAGSearchRequest(BaseModel):
    """Schema for RAG search requests."""
    query: str = Field(..., min_length=1)
    knowledge_pool_ids: Optional[List[UUID]] = None
    limit: int = Field(default=5, ge=1, le=50)


class RAGDocument(BaseModel):
    """Schema for a retrieved RAG document chunk."""
    document_id: UUID
    filename: str
    content: str
    score: float
    metadata: Optional[dict] = None


class RAGSearchResponse(BaseModel):
    """Schema for RAG search responses."""
    query: str
    results: List[RAGDocument]
    num_results: int


class DocumentUploadResponse(BaseModel):
    """Schema for document upload responses."""
    document_id: UUID
    filename: str
    status: str
    message: str


# ============================================================================
# Chat Schemas
# ============================================================================

class ChatMessage(BaseModel):
    """Chat message format."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    """Schema for chat requests."""
    messages: List[ChatMessage]
    conversation_id: Optional[UUID] = None
    use_rag: bool = False
    use_scratchpad: bool = False
    knowledge_pool_ids: Optional[List[UUID]] = None
    stream: bool = True


class ChatResponse(BaseModel):
    """Schema for chat responses."""
    message: str
    conversation_id: UUID
    metadata: Optional[dict] = None


# ============================================================================
# Memory Schemas
# ============================================================================

class UserMemoryBase(BaseModel):
    """Base user memory schema."""
    content: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class UserMemoryCreate(UserMemoryBase):
    """Schema for creating a user memory."""
    source_conversation_id: Optional[UUID] = None


class UserMemoryResponse(UserMemoryBase):
    """Schema for user memory responses."""
    id: UUID
    user_id: UUID
    source_conversation_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
