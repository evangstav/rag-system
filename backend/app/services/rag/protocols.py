"""
Protocol interfaces for RAG components.

These protocols define the contracts that different RAG provider implementations
must follow, enabling easy swapping of implementations (e.g., OpenAI -> Cohere,
Qdrant -> Pinecone, etc.).
"""

from typing import Protocol, List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass
class DocumentChunk:
    """A chunk of text from a document with metadata."""

    content: str
    metadata: Dict[str, Any]
    chunk_index: int = 0

    def __post_init__(self):
        """Ensure required metadata fields exist."""
        if "document_id" not in self.metadata:
            self.metadata["document_id"] = None
        if "source" not in self.metadata:
            self.metadata["source"] = "unknown"


@dataclass
class SearchResult:
    """A search result from vector store with score."""

    content: str
    score: float
    metadata: Dict[str, Any]
    document_id: Optional[UUID] = None
    chunk_index: int = 0

    @property
    def filename(self) -> str:
        """Extract filename from metadata."""
        return self.metadata.get("filename", self.metadata.get("source", "unknown"))


@dataclass
class Document:
    """A complete document for processing."""

    content: str
    metadata: Dict[str, Any]
    document_id: Optional[UUID] = None

    @property
    def source(self) -> str:
        """Get document source from metadata."""
        return self.metadata.get("source", "unknown")

    @property
    def filename(self) -> str:
        """Get document filename from metadata."""
        return self.metadata.get("filename", "unknown")


class EmbeddingProvider(Protocol):
    """
    Protocol for embedding generation services.

    Implementations: OpenAIEmbeddings, CohereEmbeddings, HuggingFaceEmbeddings, etc.
    """

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a single batch.

        This is more efficient than calling embed_text() multiple times.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors, one per input text
        """
        ...

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embeddings."""
        ...

    @property
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        ...


class VectorStore(Protocol):
    """
    Protocol for vector database services.

    Implementations: QdrantVectorStore, PineconeVectorStore, ChromaVectorStore, etc.
    """

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "cosine",
    ) -> None:
        """
        Create a new collection for storing vectors.

        Args:
            collection_name: Name of the collection
            vector_size: Dimensionality of vectors
            distance: Distance metric ("cosine", "euclidean", "dot")
        """
        ...

    async def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection and all its vectors.

        Args:
            collection_name: Name of the collection to delete
        """
        ...

    async def upsert(
        self,
        collection_name: str,
        documents: List[DocumentChunk],
        vectors: List[List[float]],
    ) -> List[str]:
        """
        Insert or update document chunks with their vectors.

        Args:
            collection_name: Collection to insert into
            documents: List of document chunks
            vectors: List of embedding vectors (must match documents order)

        Returns:
            List of IDs for the inserted documents
        """
        ...

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors in a collection.

        Args:
            collection_name: Collection to search in
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0-1)
            filter_conditions: Optional metadata filters

        Returns:
            List of search results sorted by similarity (highest first)
        """
        ...

    async def delete_by_document_id(
        self,
        collection_name: str,
        document_id: UUID,
    ) -> int:
        """
        Delete all chunks belonging to a document.

        Args:
            collection_name: Collection to delete from
            document_id: ID of the document to delete

        Returns:
            Number of chunks deleted
        """
        ...

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a collection.

        Args:
            collection_name: Collection name

        Returns:
            Dict with stats like vector_count, dimensions, etc.
        """
        ...


class TextSplitter(Protocol):
    """
    Protocol for text splitting/chunking services.

    Implementations: SmartTextSplitter, RecursiveTextSplitter, etc.
    """

    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Split text into chunks.

        Args:
            text: The text to split
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of document chunks
        """
        ...

    def split_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """
        Split multiple documents into chunks.

        Args:
            documents: List of documents to split

        Returns:
            List of all chunks from all documents
        """
        ...

    @property
    def chunk_size(self) -> int:
        """Get the target chunk size in characters."""
        ...

    @property
    def chunk_overlap(self) -> int:
        """Get the overlap between chunks in characters."""
        ...


class DocumentLoader(Protocol):
    """
    Protocol for document loading services.

    Implementations: PDFLoader, DocxLoader, WebLoader, TextLoader, etc.
    """

    def supports(self, source: str) -> bool:
        """
        Check if this loader supports the given source.

        Args:
            source: File path, URL, or other source identifier

        Returns:
            True if this loader can handle the source
        """
        ...

    async def load(self, source: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """
        Load a document from a source.

        Args:
            source: File path, URL, or other source identifier
            metadata: Optional metadata to attach to the document

        Returns:
            Loaded document with content and metadata
        """
        ...

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        ...
