"""
Embedding provider implementations.

OpenAI embeddings using the text-embedding-3-small or text-embedding-3-large models.
"""

from typing import List
import asyncio
from openai import AsyncOpenAI

from app.config import settings


class OpenAIEmbeddings:
    """
    OpenAI embeddings provider using the embeddings API.

    Supports both text-embedding-3-small (1536 dims) and text-embedding-3-large (3072 dims).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ):
        """
        Initialize OpenAI embeddings provider.

        Args:
            api_key: OpenAI API key (defaults to settings.openai_api_key)
            model: Model name (defaults to settings.embedding_model)
            dimensions: Embedding dimensions (defaults to settings.embedding_dimensions)
        """
        self.client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        self._model = model or settings.embedding_model
        self._dimensions = dimensions or settings.embedding_dimensions

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        # OpenAI API requires non-empty input
        if not text.strip():
            text = " "  # Use single space for empty strings

        response = await self.client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=self._dimensions,
        )

        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a single batch.

        This is more efficient than calling embed_text() multiple times.

        Args:
            texts: List of texts to embed (max 2048 texts per batch for OpenAI)

        Returns:
            List of embedding vectors, one per input text
        """
        if not texts:
            return []

        # Replace empty strings with spaces
        processed_texts = [text.strip() or " " for text in texts]

        # OpenAI has a limit of 2048 texts per batch
        batch_size = 2048
        all_embeddings: List[List[float]] = []

        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i : i + batch_size]

            response = await self.client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
            )

            # Extract embeddings in the correct order
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embeddings."""
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        return self._model


class CachedEmbeddings:
    """
    Wrapper around an embedding provider that caches results.

    Useful for development/testing to avoid repeated API calls.
    """

    def __init__(self, provider: OpenAIEmbeddings):
        """
        Initialize cached embeddings wrapper.

        Args:
            provider: Underlying embedding provider
        """
        self.provider = provider
        self._cache: dict[str, List[float]] = {}

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text with caching.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if text in self._cache:
            return self._cache[text]

        embedding = await self.provider.embed_text(text)
        self._cache[text] = embedding
        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with caching.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors, one per input text
        """
        # Check which texts are already cached
        results: List[List[float] | None] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, text in enumerate(texts):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Fetch uncached embeddings
        if uncached_texts:
            new_embeddings = await self.provider.embed_batch(uncached_texts)

            # Cache and store results
            for idx, text, embedding in zip(uncached_indices, uncached_texts, new_embeddings):
                self._cache[text] = embedding
                results[idx] = embedding

        return [r for r in results if r is not None]  # Type checker satisfaction

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embeddings."""
        return self.provider.dimensions

    @property
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        return self.provider.model_name

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Return the number of cached embeddings."""
        return len(self._cache)
