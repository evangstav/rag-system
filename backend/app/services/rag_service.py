"""
RAG service orchestration layer.

Coordinates embedding generation, vector storage, document loading, and search.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.services.rag.bm25_index import BM25Index
from app.services.rag.embeddings import OpenAIEmbeddings
from app.services.rag.hybrid_search import HybridSearchService
from app.services.rag.postgres_bm25 import PostgresBM25Index
from app.services.rag.loaders import (
    BaseDocumentLoader,
    PDFLoader,
    TextLoader,
    WebLoader,
)
from app.services.rag.protocols import (
    Document,
    EmbeddingProvider,
    SearchResult,
    TextSplitter,
    VectorStore,
)
from app.services.rag.text_splitter import SmartTextSplitter
from app.services.rag.vector_store import QdrantVectorStore

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
        enable_hybrid_search: Optional[bool] = None,
        bm25_index = None,
        db_session: Optional[AsyncSession] = None,
    ):
        """
        Initialize RAG service.

        Args:
            embedding_provider: Embedding provider (defaults to OpenAIEmbeddings)
            vector_store: Vector store (defaults to QdrantVectorStore)
            text_splitter: Text splitter (defaults to SmartTextSplitter)
            enable_hybrid_search: Enable hybrid search (defaults to config setting)
            bm25_index: BM25 index (PostgresBM25Index or BM25Index, defaults to in-memory)
            db_session: Database session for PostgresBM25Index (optional)
        """
        self.embedding_provider = embedding_provider or OpenAIEmbeddings()
        self.vector_store = vector_store or QdrantVectorStore()
        self.text_splitter = text_splitter or SmartTextSplitter()
        self.db_session = db_session

        # Initialize document loaders
        self.loaders: List[BaseDocumentLoader] = [
            TextLoader(),
            PDFLoader(),
            WebLoader(),
        ]

        if DOCX_AVAILABLE:
            self.loaders.append(DocxLoader())

        # Initialize hybrid search components if enabled
        self.enable_hybrid_search = (
            enable_hybrid_search
            if enable_hybrid_search is not None
            else settings.enable_hybrid_search
        )

        if self.enable_hybrid_search:
            # Use provided BM25 index or create default
            if bm25_index is not None:
                self.bm25_index = bm25_index
            elif db_session is not None:
                # Use PostgreSQL-backed BM25 if db session provided
                self.bm25_index = PostgresBM25Index(db_session)
            else:
                # Fall back to in-memory BM25
                self.bm25_index = BM25Index()

            self.hybrid_search = HybridSearchService(
                embedding_provider=self.embedding_provider,
                vector_store=self.vector_store,
                bm25_index=self.bm25_index,
            )
        else:
            self.bm25_index = None
            self.hybrid_search = None

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

        # Also delete BM25 index if hybrid search is enabled
        if self.enable_hybrid_search and self.bm25_index:
            await self.bm25_index.delete_collection(collection_name)

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

        # 5. Build BM25 index if hybrid search is enabled
        if self.enable_hybrid_search and self.bm25_index:
            await self.bm25_index.append_documents(
                collection_name=collection_name,
                documents=chunks,
            )

        # Calculate stats
        total_tokens = sum(len(text.split()) for text in chunk_texts)

        return {
            "num_chunks": len(chunks),
            "num_tokens": total_tokens,
            "status": "completed",
            "hybrid_search_enabled": self.enable_hybrid_search,
        }

    async def search(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        use_hybrid: Optional[bool] = None,
    ) -> List[SearchResult]:
        """
        Search for relevant documents.

        Uses hybrid search (semantic + keyword) if enabled, otherwise pure semantic.

        Args:
            query: Search query
            collection_name: Knowledge pool to search in
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0-1)
            filter_conditions: Optional metadata filters
            use_hybrid: Override to force hybrid/semantic search (defaults to service config)

        Returns:
            List of search results sorted by relevance
        """
        # Determine whether to use hybrid search
        should_use_hybrid = (
            use_hybrid if use_hybrid is not None else self.enable_hybrid_search
        )

        # Use hybrid search if enabled and available
        if should_use_hybrid and self.hybrid_search:
            return await self.hybrid_search.search(
                query=query,
                collection_name=collection_name,
                limit=limit,
                score_threshold=score_threshold,
                filter_conditions=filter_conditions,
            )

        # Fall back to pure semantic search
        query_embedding = await self.embedding_provider.embed_text(query)

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
        # Delete from vector store
        num_deleted = await self.vector_store.delete_by_document_id(
            collection_name=collection_name,
            document_id=document_id,
        )

        # Also delete from BM25 index if hybrid search is enabled
        if self.enable_hybrid_search and self.bm25_index:
            await self.bm25_index.delete_by_document_id(
                collection_name=collection_name,
                document_id=str(document_id),
            )

        return num_deleted

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a knowledge pool.

        Args:
            collection_name: Knowledge pool name

        Returns:
            Dict with stats (vectors_count, dimensions, hybrid_search, etc.)
        """
        stats = await self.vector_store.get_collection_stats(collection_name)

        # Add hybrid search stats if enabled
        if self.enable_hybrid_search and self.hybrid_search:
            hybrid_stats = await self.hybrid_search.get_stats(collection_name)
            stats["hybrid_search"] = hybrid_stats

        return stats

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
