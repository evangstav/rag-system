"""
PostgreSQL Full-Text Search implementation for BM25-style keyword search.

Uses PostgreSQL's native full-text search with ts_rank_cd for BM25-style ranking.
Provides persistent, scalable alternative to in-memory BM25 index.
"""

from typing import Any, Dict, List, Tuple
from sqlalchemy import select, delete, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.logging_config import get_logger
from app.models.database import BM25Document
from app.services.rag.protocols import DocumentChunk

logger = get_logger(__name__)


class PostgresBM25Index:
    """
    PostgreSQL-backed BM25 keyword index using Full-Text Search.

    Features:
    - Persistent storage (survives restarts)
    - Scalable to millions of documents
    - Native PostgreSQL indexing (GIN)
    - ts_rank_cd for BM25-style ranking
    - ACID guarantees

    Usage:
        async with get_session() as db:
            index = PostgresBM25Index(db)
            await index.index_documents("collection", chunks)
            results = await index.search("collection", "query", limit=10)
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize PostgreSQL BM25 index.

        Args:
            db: Async SQLAlchemy database session
        """
        self.db = db

    async def index_documents(
        self,
        collection_name: str,
        documents: List[DocumentChunk],
    ) -> None:
        """
        Index documents for full-text search.

        Args:
            collection_name: Name of the collection
            documents: List of document chunks to index
        """
        if not documents:
            logger.warning(
                "postgres_bm25_no_documents",
                collection_name=collection_name,
            )
            return

        # Build document objects
        bm25_docs = []
        for doc in documents:
            bm25_doc = BM25Document(
                collection_name=collection_name,
                document_id=str(doc.metadata.get("document_id", "unknown")),
                chunk_index=doc.chunk_index,
                content=doc.content,
                doc_metadata={k: v for k, v in doc.metadata.items() if k != "document_id"},
            )
            bm25_docs.append(bm25_doc)

        # Bulk insert
        # Note: content_tsv is automatically updated by PostgreSQL trigger
        self.db.add_all(bm25_docs)
        await self.db.commit()

        logger.info(
            "postgres_bm25_index_built",
            collection_name=collection_name,
            num_documents=len(documents),
        )

    async def append_documents(
        self,
        collection_name: str,
        documents: List[DocumentChunk],
    ) -> None:
        """
        Append new documents to existing index.

        Args:
            collection_name: Name of the collection
            documents: New documents to add
        """
        # PostgreSQL handles incremental updates efficiently
        await self.index_documents(collection_name, documents)

        logger.info(
            "postgres_bm25_documents_appended",
            collection_name=collection_name,
            num_documents=len(documents),
        )

    async def search(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search using PostgreSQL Full-Text Search with BM25-style ranking.

        Uses ts_rank_cd which considers:
        - Term frequency
        - Document length normalization
        - Proximity of terms

        Args:
            collection_name: Collection to search
            query: Search query
            limit: Maximum number of results

        Returns:
            List of tuples: (document_metadata, score)
            Sorted by score (highest first)
        """
        if not query or not query.strip():
            logger.warning("postgres_bm25_empty_query")
            return []

        # Use websearch_to_tsquery for user-friendly query syntax
        # Supports: "phrase search", AND, OR, NOT, etc.
        result = await self.db.execute(
            text("""
                SELECT
                    document_id,
                    chunk_index,
                    content,
                    doc_metadata,
                    ts_rank_cd(
                        content_tsv,
                        websearch_to_tsquery('english', :query),
                        32  -- Normalization flag: divide by doc length
                    ) as score
                FROM bm25_documents
                WHERE
                    collection_name = :collection_name
                    AND content_tsv @@ websearch_to_tsquery('english', :query)
                ORDER BY score DESC
                LIMIT :limit
            """),
            {"query": query, "collection_name": collection_name, "limit": limit},
        )

        results = []
        for row in result:
            metadata = {
                "document_id": row.document_id,
                "chunk_index": row.chunk_index,
                "content": row.content,
                **(row.doc_metadata or {}),
            }
            score = float(row.score)
            results.append((metadata, score))

        logger.debug(
            "postgres_bm25_search_results",
            collection_name=collection_name,
            query=query,
            num_results=len(results),
        )

        return results

    async def delete_collection(self, collection_name: str) -> None:
        """
        Delete all documents in a collection.

        Args:
            collection_name: Collection to delete
        """
        result = await self.db.execute(
            delete(BM25Document).where(BM25Document.collection_name == collection_name)
        )
        await self.db.commit()

        deleted_count = result.rowcount
        logger.info(
            "postgres_bm25_collection_deleted",
            collection_name=collection_name,
            deleted_count=deleted_count,
        )

    async def delete_by_document_id(
        self,
        collection_name: str,
        document_id: str,
    ) -> int:
        """
        Delete all chunks belonging to a document.

        Args:
            collection_name: Collection name
            document_id: Document ID to delete

        Returns:
            Number of chunks deleted
        """
        result = await self.db.execute(
            delete(BM25Document).where(
                BM25Document.collection_name == collection_name,
                BM25Document.document_id == document_id,
            )
        )
        await self.db.commit()

        deleted_count = result.rowcount
        logger.info(
            "postgres_bm25_document_deleted",
            deleted_count=deleted_count,
            document_id=document_id,
            collection_name=collection_name,
        )

        return deleted_count

    async def has_collection(self, collection_name: str) -> bool:
        """
        Check if collection has any documents indexed.

        Args:
            collection_name: Collection name

        Returns:
            True if collection exists and has documents
        """
        result = await self.db.execute(
            select(func.count(BM25Document.id)).where(
                BM25Document.collection_name == collection_name
            )
        )
        count = result.scalar_one()
        return count > 0

    async def get_collection_size(self, collection_name: str) -> int:
        """
        Get number of documents in collection.

        Args:
            collection_name: Collection name

        Returns:
            Number of documents
        """
        result = await self.db.execute(
            select(func.count(BM25Document.id)).where(
                BM25Document.collection_name == collection_name
            )
        )
        return result.scalar_one()

    async def get_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about the BM25 index for a collection.

        Args:
            collection_name: Collection name

        Returns:
            Dictionary with stats
        """
        result = await self.db.execute(
            text("""
                SELECT
                    COUNT(*) as document_count,
                    AVG(LENGTH(content)) as avg_content_length,
                    pg_size_pretty(pg_total_relation_size('bm25_documents')) as table_size
                FROM bm25_documents
                WHERE collection_name = :collection_name
            """),
            {"collection_name": collection_name},
        )

        row = result.fetchone()

        return {
            "collection_name": collection_name,
            "document_count": row.document_count if row else 0,
            "avg_content_length": float(row.avg_content_length)
            if row and row.avg_content_length
            else 0,
            "table_size": row.table_size if row else "0 bytes",
            "backend": "postgresql_fts",
        }
