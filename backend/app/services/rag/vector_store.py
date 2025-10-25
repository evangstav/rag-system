"""
Vector store implementations.

Qdrant vector database for storing and searching document embeddings.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
import uuid as uuid_lib
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.config import settings
from app.services.rag.protocols import DocumentChunk, SearchResult


class QdrantVectorStore:
    """
    Qdrant vector database implementation.

    Provides async vector storage and similarity search using Qdrant.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize Qdrant vector store.

        Args:
            url: Qdrant server URL (defaults to settings.qdrant_url)
            api_key: Qdrant API key (defaults to settings.qdrant_api_key, optional)
        """
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self.client = AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key if self.api_key else None,
        )

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
        # Map distance string to Qdrant Distance enum
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }

        qdrant_distance = distance_map.get(distance.lower(), Distance.COSINE)

        # Check if collection already exists
        collections = await self.client.get_collections()
        collection_names = [col.name for col in collections.collections]

        if collection_name not in collection_names:
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=qdrant_distance,
                ),
            )

    async def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection and all its vectors.

        Args:
            collection_name: Name of the collection to delete
        """
        await self.client.delete_collection(collection_name=collection_name)

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

        Raises:
            ValueError: If inputs are invalid (empty, mismatched sizes, invalid dimensions)
        """
        # Validate collection name
        if not collection_name or not collection_name.strip():
            raise ValueError("Collection name cannot be empty")

        # Validate documents and vectors are not empty
        if not documents:
            raise ValueError("Documents list cannot be empty")

        if not vectors:
            raise ValueError("Vectors list cannot be empty")

        # Validate counts match
        if len(documents) != len(vectors):
            raise ValueError(
                f"Number of documents ({len(documents)}) must match "
                f"number of vectors ({len(vectors)})"
            )

        # Validate vector dimensions are consistent
        if vectors:
            import math

            expected_dim = len(vectors[0])

            if expected_dim == 0:
                raise ValueError("Vectors cannot be empty (0 dimensions)")

            for i, vec in enumerate(vectors):
                if len(vec) != expected_dim:
                    raise ValueError(
                        f"Vector {i} has inconsistent dimension: "
                        f"{len(vec)} != {expected_dim}"
                    )

                # Check for NaN or Inf values
                if any(math.isnan(v) or math.isinf(v) for v in vec):
                    raise ValueError(
                        f"Vector {i} contains NaN or Inf values (document: "
                        f"{documents[i].content[:50]}...)"
                    )

        points = []
        point_ids = []

        for doc, vector in zip(documents, vectors):
            # Generate unique ID for this chunk
            point_id = str(uuid_lib.uuid4())
            point_ids.append(point_id)

            # Build payload with content and metadata
            payload = {
                "content": doc.content,
                "chunk_index": doc.chunk_index,
                **doc.metadata,  # Spread metadata fields
            }

            # Create point
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
            points.append(point)

        # Upsert all points
        await self.client.upsert(
            collection_name=collection_name,
            points=points,
        )

        return point_ids

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
            filter_conditions: Optional metadata filters (e.g., {"document_id": "uuid"})

        Returns:
            List of search results sorted by similarity (highest first)
        """
        # Build filter if provided
        qdrant_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                # Convert UUID to string if needed
                if isinstance(value, UUID):
                    value = str(value)
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
            qdrant_filter = Filter(must=conditions)

        # Perform search
        search_results = await self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
        )

        # Convert to SearchResult objects
        results = []
        for hit in search_results:
            payload = hit.payload or {}

            # Extract document_id if present
            doc_id = payload.get("document_id")
            if doc_id and isinstance(doc_id, str):
                try:
                    doc_id = UUID(doc_id)
                except ValueError:
                    doc_id = None

            # Extract content and chunk_index
            content = payload.get("content", "")
            chunk_index = payload.get("chunk_index", 0)

            # Remove content and chunk_index from metadata to avoid duplication
            metadata = {
                k: v for k, v in payload.items() if k not in ["content", "chunk_index"]
            }

            result = SearchResult(
                content=content,
                score=hit.score,
                metadata=metadata,
                document_id=doc_id,
                chunk_index=chunk_index,
            )
            results.append(result)

        return results

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
        # Build filter for document_id
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=str(document_id)),
                )
            ]
        )

        # Delete points matching the filter
        result = await self.client.delete(
            collection_name=collection_name,
            points_selector=filter_condition,
        )

        # Qdrant doesn't return count in result, so we'll return a placeholder
        # In production, you might want to search first to count
        return 0  # TODO: Implement proper count if needed

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a collection.

        Args:
            collection_name: Collection name

        Returns:
            Dict with stats like vector_count, dimensions, etc.
        """
        info = await self.client.get_collection(collection_name=collection_name)

        return {
            "name": collection_name,
            "vectors_count": info.vectors_count or 0,
            "points_count": info.points_count or 0,
            "dimensions": info.config.params.vectors.size
            if info.config.params.vectors
            else 0,
            "distance": info.config.params.vectors.distance.value
            if info.config.params.vectors
            else "unknown",
        }

    async def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Collection name to check

        Returns:
            True if collection exists, False otherwise
        """
        collections = await self.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        return collection_name in collection_names
