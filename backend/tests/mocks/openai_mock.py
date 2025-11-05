"""
Mock implementation of OpenAI API client for testing.

Provides deterministic, fast responses without making actual API calls.
"""

from typing import List, AsyncIterator, Optional
import hashlib


class MockMessage:
    """Mock message object."""
    def __init__(self, content: str, role: str = "assistant"):
        self.content = content
        self.role = role


class MockChoice:
    """Mock choice object in completion response."""
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.message = MockMessage(content)
        self.finish_reason = finish_reason
        self.index = 0


class MockDelta:
    """Mock delta object for streaming responses."""
    def __init__(self, content: Optional[str] = None):
        self.content = content
        self.role = "assistant" if content else None


class MockStreamChoice:
    """Mock choice object in streaming response."""
    def __init__(self, content: Optional[str] = None, finish_reason: Optional[str] = None):
        self.delta = MockDelta(content)
        self.finish_reason = finish_reason
        self.index = 0


class MockChatCompletion:
    """Mock chat completion response."""
    def __init__(self, content: str, model: str = "gpt-4"):
        self.id = f"chatcmpl-mock-{hash(content) % 10000}"
        self.choices = [MockChoice(content)]
        self.model = model
        self.usage = type('Usage', (), {
            'prompt_tokens': len(content.split()) * 2,
            'completion_tokens': len(content.split()),
            'total_tokens': len(content.split()) * 3,
        })()
        self.created = 1234567890


class MockStreamChunk:
    """Mock streaming chunk."""
    def __init__(self, content: Optional[str] = None, finish_reason: Optional[str] = None):
        self.id = "chatcmpl-mock-stream"
        self.choices = [MockStreamChoice(content, finish_reason)]
        self.model = "gpt-4"
        self.created = 1234567890


class MockEmbeddingData:
    """Mock embedding data object."""
    def __init__(self, embedding: List[float], index: int = 0):
        self.embedding = embedding
        self.index = index
        self.object = "embedding"


class MockEmbeddingResponse:
    """Mock embedding API response."""
    def __init__(self, embeddings: List[List[float]], model: str = "text-embedding-ada-002"):
        self.data = [MockEmbeddingData(emb, i) for i, emb in enumerate(embeddings)]
        self.model = model
        self.usage = type('Usage', (), {
            'prompt_tokens': len(embeddings) * 10,
            'total_tokens': len(embeddings) * 10,
        })()
        self.object = "list"


def generate_deterministic_embedding(text: str, dimension: int = 1536) -> List[float]:
    """
    Generate a deterministic fake embedding based on input text.

    Uses MD5 hash of text to seed a pseudo-random embedding vector.
    This ensures the same text always produces the same embedding.

    Args:
        text: Input text to embed
        dimension: Embedding dimension (default: 1536 for OpenAI ada-002)

    Returns:
        List of floats representing the embedding vector
    """
    # Use MD5 hash to get deterministic seed
    hash_bytes = hashlib.md5(text.encode()).digest()
    hash_int = int.from_bytes(hash_bytes, byteorder='big')

    # Generate pseudo-random floats between -1 and 1
    embedding = []
    seed = hash_int
    for i in range(dimension):
        # Linear congruential generator for pseudo-random numbers
        seed = (seed * 1103515245 + 12345) % (2**31)
        value = (seed / (2**31)) * 2 - 1  # Normalize to [-1, 1]
        embedding.append(value)

    # Normalize to unit vector (cosine similarity friendly)
    magnitude = sum(x**2 for x in embedding) ** 0.5
    if magnitude > 0:
        embedding = [x / magnitude for x in embedding]

    return embedding


class MockOpenAIClient:
    """
    Mock OpenAI client for testing.

    Provides deterministic responses without making actual API calls.
    Supports both standard and streaming chat completions, plus embeddings.
    """

    def __init__(self, **kwargs):
        """Initialize mock client (ignores api_key and other params)."""
        self.chat = MockChatCompletions()
        self.embeddings = MockEmbeddings()


class MockChatCompletions:
    """Mock chat completions API."""

    async def create(
        self,
        model: str,
        messages: List[dict],
        stream: bool = False,
        **kwargs
    ):
        """
        Create a chat completion (standard or streaming).

        Args:
            model: Model name (ignored, always returns same mock)
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to return streaming response
            **kwargs: Additional parameters (ignored)

        Returns:
            MockChatCompletion or async iterator of MockStreamChunk
        """
        # Extract user query for deterministic response
        last_message = messages[-1] if messages else {"content": ""}
        query = last_message.get("content", "")

        # Generate deterministic response based on query
        response_content = self._generate_response(query)

        if stream:
            return self._create_stream(response_content)
        else:
            return MockChatCompletion(response_content, model)

    async def _create_stream(self, content: str) -> AsyncIterator[MockStreamChunk]:
        """Create streaming response."""
        # Split content into words for streaming
        words = content.split()

        for word in words:
            yield MockStreamChunk(content=word + " ")

        # Final chunk with finish_reason
        yield MockStreamChunk(finish_reason="stop")

    def _generate_response(self, query: str) -> str:
        """
        Generate deterministic response based on query.

        You can customize this method to return specific responses for testing.
        """
        query_lower = query.lower()

        # Provide specific responses for common test queries
        if "machine learning" in query_lower:
            return "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve their performance over time."

        elif "supervised learning" in query_lower:
            return "Supervised learning is a type of machine learning where the algorithm learns from labeled training data to make predictions on unseen data."

        elif "python" in query_lower:
            return "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in data science, web development, and automation."

        elif "rag" in query_lower or "retrieval" in query_lower:
            return "Retrieval-Augmented Generation (RAG) combines information retrieval with language generation to provide more accurate and contextual responses based on external knowledge."

        # Default response
        return f"This is a mock response to your query about: {query[:50]}..."


class MockEmbeddings:
    """Mock embeddings API."""

    async def create(
        self,
        model: str,
        input: str | List[str],
        **kwargs
    ) -> MockEmbeddingResponse:
        """
        Create embeddings for input text(s).

        Args:
            model: Model name (ignored)
            input: Single string or list of strings to embed
            **kwargs: Additional parameters (ignored)

        Returns:
            MockEmbeddingResponse with deterministic embeddings
        """
        # Normalize input to list
        texts = [input] if isinstance(input, str) else input

        # Generate embeddings
        embeddings = [generate_deterministic_embedding(text) for text in texts]

        return MockEmbeddingResponse(embeddings, model)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_mock_client(**kwargs) -> MockOpenAIClient:
    """Create a mock OpenAI client."""
    return MockOpenAIClient(**kwargs)


async def mock_chat_completion(query: str, model: str = "gpt-4") -> MockChatCompletion:
    """Quick helper to generate a mock chat completion."""
    client = MockOpenAIClient()
    messages = [{"role": "user", "content": query}]
    return await client.chat.create(model=model, messages=messages)


async def mock_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """Quick helper to generate a mock embedding."""
    client = MockOpenAIClient()
    response = await client.embeddings.create(model=model, input=text)
    return response.data[0].embedding
