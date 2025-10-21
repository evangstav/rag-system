"""
Application configuration using Pydantic Settings.

Loads configuration from environment variables and .env file.
"""

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://user:pass@localhost:5432/ragchat",
        alias="DATABASE_URL"
    )

    # OpenAI
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")

    # Redis (optional, for future use)
    redis_url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")

    # Application
    secret_key: str = Field(default="your-secret-key-here", alias="SECRET_KEY")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"],
        alias="CORS_ORIGINS"
    )

    # Environment
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")

    # JWT (for authentication, to be implemented)
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=30, alias="JWT_EXPIRE_MINUTES")

    # RAG settings
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1536, alias="EMBEDDING_DIMENSIONS")
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    max_rag_results: int = Field(default=5, alias="MAX_RAG_RESULTS")

    # LLM settings
    default_llm_model: str = Field(default="gpt-4-turbo-preview", alias="DEFAULT_LLM_MODEL")
    max_tokens: int = Field(default=4000, alias="MAX_TOKENS")
    temperature: float = Field(default=0.7, alias="TEMPERATURE")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @property
    def async_database_url(self) -> str:
        """Get async database URL for SQLAlchemy."""
        url = self.database_url
        # Convert postgres:// to postgresql+asyncpg://
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://") and "asyncpg" not in url:
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

    @property
    def sync_database_url(self) -> str:
        """Get sync database URL (for Alembic migrations)."""
        url = self.database_url
        # Remove asyncpg driver if present
        if "postgresql+asyncpg://" in url:
            url = url.replace("postgresql+asyncpg://", "postgresql://", 1)
        return url


# Global settings instance
settings = Settings()
