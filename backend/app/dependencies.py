"""
FastAPI dependency injection functions.

Provides reusable dependencies for database sessions, services, and authentication.
"""

from typing import AsyncGenerator, Annotated
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.config import settings

# Type alias for database session dependency
DatabaseSession = Annotated[AsyncSession, Depends(get_session)]


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
# Authentication Dependencies (to be implemented)
# ============================================================================

# async def get_current_user(
#     token: str = Depends(oauth2_scheme),
#     db: AsyncSession = Depends(get_db)
# ) -> User:
#     """Get the current authenticated user from JWT token."""
#     # TODO: Implement JWT token validation
#     # credentials_exception = HTTPException(
#     #     status_code=status.HTTP_401_UNAUTHORIZED,
#     #     detail="Could not validate credentials",
#     #     headers={"WWW-Authenticate": "Bearer"},
#     # )
#     # ...
#     pass


# async def get_current_active_user(
#     current_user: User = Depends(get_current_user),
# ) -> User:
#     """Get the current active user (not disabled)."""
#     # if not current_user.is_active:
#     #     raise HTTPException(status_code=400, detail="Inactive user")
#     # return current_user
#     pass


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
