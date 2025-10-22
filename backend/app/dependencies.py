"""
FastAPI dependency injection functions.

Provides reusable dependencies for database sessions, services, and authentication.
"""

from typing import AsyncGenerator, Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.database import get_session
from app.config import settings
from app.models.database import User
from app.auth import decode_token

# Type alias for database session dependency
DatabaseSession = Annotated[AsyncSession, Depends(get_session)]

# HTTP Bearer token scheme for JWT authentication
security = HTTPBearer()


# ============================================================================
# Database Dependencies
# ============================================================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session dependency.

    This is an alias for get_session() from database.py for backward compatibility.
    """
    async for session in get_session():
        yield session


# ============================================================================
# Authentication Dependencies
# ============================================================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get the current authenticated user from JWT token.

    Args:
        credentials: HTTP Bearer token credentials
        db: Database session

    Returns:
        Authenticated User object

    Raises:
        HTTPException 401: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Decode token
    token = credentials.credentials
    payload = decode_token(token)

    if not payload:
        raise credentials_exception

    # Get user ID from token
    user_id_str = payload.get("sub")
    if not user_id_str:
        raise credentials_exception

    try:
        user_id = UUID(user_id_str)
    except ValueError:
        raise credentials_exception

    # Fetch user from database
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise credentials_exception

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get the current active user (not disabled).

    Args:
        current_user: Current authenticated user

    Returns:
        Active User object

    Raises:
        HTTPException 400: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


# ============================================================================
# Service Dependencies
# ============================================================================

# These will be implemented after creating the service classes

# def get_llm_service() -> "LLMService":
#     """Get LLM service instance."""
#     from app.services.llm import LLMService
#     return LLMService(api_key=settings.openai_api_key)


# def get_embedding_provider() -> "EmbeddingProvider":
#     """Get embedding provider instance."""
#     from app.services.rag.embeddings import OpenAIEmbeddings
#     return OpenAIEmbeddings(
#         api_key=settings.openai_api_key,
#         model=settings.embedding_model
#     )


# def get_vector_store() -> "VectorStore":
#     """Get vector store instance."""
#     from app.services.rag.vector_store import QdrantVectorStore
#     return QdrantVectorStore(
#         url=settings.qdrant_url,
#         api_key=settings.qdrant_api_key
#     )


# def get_rag_service(
#     embedding_provider: "EmbeddingProvider" = Depends(get_embedding_provider),
#     vector_store: "VectorStore" = Depends(get_vector_store),
# ) -> "RAGService":
#     """Get RAG service instance."""
#     from app.services.rag_service import RAGService
#     return RAGService(
#         embedding_provider=embedding_provider,
#         vector_store=vector_store
#     )


# ============================================================================
# Utility Dependencies
# ============================================================================

def get_settings():
    """Get application settings."""
    return settings
