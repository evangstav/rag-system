"""
RAG service orchestration layer.

Coordinates embedding generation, vector storage, document loading, and search.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
import os

from app.services.rag.protocols import (
    Document,
    DocumentChunk,
    SearchResult,
    EmbeddingProvider,
    VectorStore,
    TextSplitter,
)
from app.services.rag.embeddings import OpenAIEmbeddings
from app.services.rag.vector_store import QdrantVectorStore
from app.services.rag.text_splitter import SmartTextSplitter
from app.services.rag.loaders import (
    BaseDocumentLoader,
    TextLoader,
    PDFLoader,
    WebLoader,
)

# Try to import DocxLoader
try:
    from app.services.rag.loaders import DocxLoader

    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class RAGService:
    """
    RAG service that orchestrates document processing and retrieval.

    Coordinates:
    - Document loading (PDF, DOCX, web, text)
    - Text splitting
    - Embedding generation
    - Vector storage
    - Similarity search
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None,
        text_splitter: Optional[TextSplitter] = None,
    ):
        """
        Initialize RAG service.

        Args:
            embedding_provider: Embedding provider (defaults to OpenAIEmbeddings)
            vector_store: Vector store (defaults to QdrantVectorStore)
            text_splitter: Text splitter (defaults to SmartTextSplitter)
        """
        self.embedding_provider = embedding_provider or OpenAIEmbeddings()
        self.vector_store = vector_store or QdrantVectorStore()
        self.text_splitter = text_splitter or SmartTextSplitter()

        # Initialize document loaders
        self.loaders: List[BaseDocumentLoader] = [
            TextLoader(),
            PDFLoader(),
            WebLoader(),
        ]

        if DOCX_AVAILABLE:
            self.loaders.append(DocxLoader())

    async def create_knowledge_pool(
        self,
        collection_name: str,
    ) -> None:
        """
        Create a new knowledge pool (vector collection).

        Args:
            collection_name: Name for the collection
        """
        await self.vector_store.create_collection(
            collection_name=collection_name,
            vector_size=self.embedding_provider.dimensions,
            distance="cosine",
        )

    async def delete_knowledge_pool(self, collection_name: str) -> None:
        """
        Delete a knowledge pool and all its documents.

        Args:
            collection_name: Name of the collection to delete
        """
        await self.vector_store.delete_collection(collection_name)

    async def load_document(
        self,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Load a document from any supported source.

        Args:
            source: File path or URL
            metadata: Optional metadata to attach

        Returns:
            Loaded document

        Raises:
            ValueError: If no loader supports this source
        """
        # Find appropriate loader
        loader = None
        for l in self.loaders:
            if l.supports(source):
                loader = l
                break

        if not loader:
            raise ValueError(
                f"No loader found for source: {source}. "
                f"Supported: PDF, DOCX, TXT, MD, web URLs"
            )

        # Load document
        return await loader.load(source, metadata)

    async def ingest_document(
        self,
        source: str,
        collection_name: str,
        document_id: UUID,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a document into a knowledge pool.

        Complete pipeline:
        1. Load document
        2. Split into chunks
        3. Generate embeddings
        4. Store in vector database

        Args:
            source: File path or URL
            collection_name: Knowledge pool to add to
            document_id: UUID of the document (from database)
            metadata: Optional metadata to attach

        Returns:
            Dict with stats (num_chunks, num_tokens, etc.)
        """
        # Add document_id to metadata
        metadata = metadata or {}
        metadata["document_id"] = str(document_id)

        # 1. Load document
        document = await self.load_document(source, metadata)

        # 2. Split into chunks
        chunks = self.text_splitter.split_text(document.content, document.metadata)

        if not chunks:
            return {
                "num_chunks": 0,
                "num_tokens": 0,
                "status": "empty",
            }

        # 3. Generate embeddings
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding_provider.embed_batch(chunk_texts)

        # 4. Store in vector database
        await self.vector_store.upsert(
            collection_name=collection_name,
            documents=chunks,
            vectors=embeddings,
        )

        # Calculate stats
        total_tokens = sum(len(text.split()) for text in chunk_texts)

        return {
            "num_chunks": len(chunks),
            "num_tokens": total_tokens,
            "status": "completed",
        }

    async def search(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            collection_name: Knowledge pool to search in
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0-1)
            filter_conditions: Optional metadata filters

        Returns:
            List of search results sorted by relevance
        """
        # Generate query embedding
        query_embedding = await self.embedding_provider.embed_text(query)

        # Search vector store
        results = await self.vector_store.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions,
        )

        return results

    async def search_multiple_pools(
        self,
        query: str,
        collection_names: List[str],
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search across multiple knowledge pools.

        Args:
            query: Search query
            collection_names: List of knowledge pools to search
            limit: Maximum number of results per pool
            score_threshold: Minimum similarity score

        Returns:
            Combined and sorted search results from all pools
        """
        # Generate query embedding once
        query_embedding = await self.embedding_provider.embed_text(query)

        # Search all collections in parallel
        import asyncio

        tasks = [
            self.vector_store.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
            )
            for collection_name in collection_names
        ]

        results_per_pool = await asyncio.gather(*tasks)

        # Combine and sort by score
        all_results = []
        for results in results_per_pool:
            all_results.extend(results)

        # Sort by score (highest first)
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Return top N overall results
        return all_results[:limit]

    async def delete_document(
        self,
        collection_name: str,
        document_id: UUID,
    ) -> int:
        """
        Delete all chunks of a document from a knowledge pool.

        Args:
            collection_name: Knowledge pool containing the document
            document_id: UUID of the document to delete

        Returns:
            Number of chunks deleted
        """
        return await self.vector_store.delete_by_document_id(
            collection_name=collection_name,
            document_id=document_id,
        )

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a knowledge pool.

        Args:
            collection_name: Knowledge pool name

        Returns:
            Dict with stats (vectors_count, dimensions, etc.)
        """
        return await self.vector_store.get_collection_stats(collection_name)

    def format_search_results_for_context(
        self,
        results: List[SearchResult],
        max_length: int = 4000,
    ) -> str:
        """
        Format search results into a context string for LLM.

        Args:
            results: Search results to format
            max_length: Maximum character length for context

        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant documents found."

        context_parts = []
        current_length = 0

        for i, result in enumerate(results, 1):
            # Format: [Source N] content
            part = f"[Source {i}: {result.filename}]\n{result.content}\n"

            # Check if adding this would exceed max length
            if current_length + len(part) > max_length:
                break

            context_parts.append(part)
            current_length += len(part)

        context = "\n".join(context_parts)

        # Add header
        header = f"Retrieved {len(context_parts)} relevant documents:\n\n"

        return header + context
