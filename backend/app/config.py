"""
Application configuration using Pydantic Settings.

Loads configuration from environment variables and .env file.
"""

from typing import List
from pydantic import Field, field_validator, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://user:pass@localhost:5432/ragchat",
        alias="DATABASE_URL",
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
        default=["http://localhost:3000"], alias="CORS_ORIGINS"
    )

    # Environment
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")

    # JWT (for authentication, to be implemented)
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=30, alias="JWT_EXPIRE_MINUTES")

    # RAG settings - Embedding
    embedding_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    embedding_dimensions: int = Field(default=1536, alias="EMBEDDING_DIMENSIONS")

    # RAG settings - Chunking (character-based, legacy)
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    # RAG settings - Chunking (token-based, recommended)
    chunk_size_tokens: int = Field(default=512, alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(default=64, alias="CHUNK_OVERLAP_TOKENS")
    tokenizer: str = Field(default="cl100k_base", alias="TOKENIZER")

    # RAG settings - Retrieval
    max_rag_results: int = Field(default=5, alias="MAX_RAG_RESULTS")

    # RAG settings - Hybrid Search
    enable_hybrid_search: bool = Field(default=False, alias="ENABLE_HYBRID_SEARCH")
    bm25_backend: str = Field(
        default="postgresql",
        alias="BM25_BACKEND",
        description="BM25 backend: 'postgresql' (persistent, scalable) or 'memory' (fast, volatile)"
    )
    hybrid_search_semantic_weight: float = Field(
        default=0.5, alias="HYBRID_SEARCH_SEMANTIC_WEIGHT"
    )
    hybrid_search_keyword_weight: float = Field(
        default=0.5, alias="HYBRID_SEARCH_KEYWORD_WEIGHT"
    )
    hybrid_search_rrf_k: int = Field(default=60, alias="HYBRID_SEARCH_RRF_K")
    hybrid_search_retrieval_k: int = Field(
        default=20, alias="HYBRID_SEARCH_RETRIEVAL_K"
    )

    # LLM settings
    default_llm_model: str = Field(
        default="gpt-4-turbo-preview", alias="DEFAULT_LLM_MODEL"
    )
    max_tokens: int = Field(default=4000, alias="MAX_TOKENS")
    temperature: float = Field(default=0.7, alias="TEMPERATURE")

    # Validators

    @field_validator("bm25_backend")
    @classmethod
    def validate_bm25_backend(cls, v: str) -> str:
        """Ensure BM25 backend is valid."""
        valid_backends = ["postgresql", "memory"]
        if v not in valid_backends:
            raise ValueError(
                f"bm25_backend must be one of {valid_backends}, got '{v}'"
            )
        return v

    @field_validator("hybrid_search_semantic_weight", "hybrid_search_keyword_weight")
    @classmethod
    def validate_weight_range(cls, v: float, info: ValidationInfo) -> float:
        """Ensure hybrid search weights are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(
                f"{info.field_name} must be between 0 and 1, got {v}"
            )
        return v

    @field_validator("hybrid_search_keyword_weight")
    @classmethod
    def validate_weights_sum(cls, v: float, info) -> float:
        """
        Warn if semantic + keyword weights don't sum to 1.0.

        Note: This is a soft warning - RRF will still work with any weights,
        but weights summing to 1.0 are easier to interpret.
        """
        # Only validate if semantic_weight has already been set
        if "hybrid_search_semantic_weight" in info.data:
            semantic_weight = info.data.get("hybrid_search_semantic_weight", 0.5)
            total = semantic_weight + v
            if not (0.99 <= total <= 1.01):  # Allow small floating point errors
                import warnings
                warnings.warn(
                    f"Hybrid search weights sum to {total:.2f} instead of 1.0. "
                    f"This may make weight interpretation less intuitive. "
                    f"(semantic={semantic_weight}, keyword={v})"
                )
        return v

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
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
