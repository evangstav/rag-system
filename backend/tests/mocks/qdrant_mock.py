"""
Mock implementation of Qdrant vector store client for testing.

Provides in-memory vector storage without requiring a running Qdrant instance.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from uuid import UUID, uuid4
import math


@dataclass
class PointStruct:
    """Mock Qdrant point structure."""
    id: str | int
    vector: List[float]
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredPoint:
    """Mock Qdrant scored point (search result)."""
    id: str | int
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None


@dataclass
class CollectionInfo:
    """Mock collection information."""
    status: str = "green"
    vectors_count: int = 0
    points_count: int = 0
    segments_count: int = 1


class MockQdrantClient:
    """
    Mock Qdrant client for testing.

    Provides in-memory vector storage and search capabilities.
    Implements the most commonly used Qdrant operations.
    """

    def __init__(self, **kwargs):
        """
        Initialize mock client.

        Args:
            **kwargs: Ignored (for compatibility with real client)
        """
        # In-memory storage: {collection_name: [points]}
        self.collections: Dict[str, List[PointStruct]] = {}
        self.collection_configs: Dict[str, Dict[str, Any]] = {}

    # =========================================================================
    # Collection Management
    # =========================================================================

    async def create_collection(
        self,
        collection_name: str,
        vectors_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """
        Create a new collection.

        Args:
            collection_name: Name of the collection
            vectors_config: Vector configuration (size, distance, etc.)
            **kwargs: Additional parameters (ignored)

        Returns:
            True if successful
        """
        if collection_name in self.collections:
            raise ValueError(f"Collection {collection_name} already exists")

        self.collections[collection_name] = []
        self.collection_configs[collection_name] = {
            "vectors_config": vectors_config or {},
            **kwargs
        }
        return True

    async def get_collection(self, collection_name: str) -> CollectionInfo:
        """
        Get collection information.

        Args:
            collection_name: Name of the collection

        Returns:
            CollectionInfo object

        Raises:
            ValueError: If collection doesn't exist
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")

        points_count = len(self.collections[collection_name])
        return CollectionInfo(
            status="green",
            vectors_count=points_count,
            points_count=points_count,
        )

    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            True if successful
        """
        if collection_name in self.collections:
            del self.collections[collection_name]
            del self.collection_configs[collection_name]
        return True

    async def collection_exists(self, collection_name: str) -> bool:
        """
        Check if collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists
        """
        return collection_name in self.collections

    # =========================================================================
    # Point Operations
    # =========================================================================

    async def upsert(
        self,
        collection_name: str,
        points: List[PointStruct] | List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Insert or update points in a collection.

        Args:
            collection_name: Name of the collection
            points: List of points to upsert
            **kwargs: Additional parameters (ignored)

        Returns:
            Status dict

        Raises:
            ValueError: If collection doesn't exist
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")

        # Normalize points to PointStruct
        normalized_points = []
        for point in points:
            if isinstance(point, dict):
                normalized_points.append(PointStruct(
                    id=point.get("id", str(uuid4())),
                    vector=point["vector"],
                    payload=point.get("payload", {})
                ))
            else:
                normalized_points.append(point)

        # Update or insert points
        collection = self.collections[collection_name]
        for new_point in normalized_points:
            # Remove existing point with same ID
            collection[:] = [p for p in collection if p.id != new_point.id]
            # Add new point
            collection.append(new_point)

        return {"status": "completed", "count": len(normalized_points)}

    async def retrieve(
        self,
        collection_name: str,
        ids: List[str | int],
        **kwargs
    ) -> List[PointStruct]:
        """
        Retrieve points by IDs.

        Args:
            collection_name: Name of the collection
            ids: List of point IDs to retrieve
            **kwargs: Additional parameters (ignored)

        Returns:
            List of points

        Raises:
            ValueError: If collection doesn't exist
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")

        collection = self.collections[collection_name]
        return [p for p in collection if p.id in ids]

    async def delete(
        self,
        collection_name: str,
        points_selector: Dict[str, Any] | List[str | int],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Delete points from a collection.

        Args:
            collection_name: Name of the collection
            points_selector: Point IDs or filter dict
            **kwargs: Additional parameters (ignored)

        Returns:
            Status dict

        Raises:
            ValueError: If collection doesn't exist
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")

        collection = self.collections[collection_name]

        # Handle different selector types
        if isinstance(points_selector, list):
            # Delete by IDs
            ids_to_delete = set(points_selector)
            initial_count = len(collection)
            collection[:] = [p for p in collection if p.id not in ids_to_delete]
            deleted_count = initial_count - len(collection)
        else:
            # Delete by filter (simplified implementation)
            # For testing, we'll just clear all if filter is provided
            deleted_count = len(collection)
            collection.clear()

        return {"status": "completed", "deleted": deleted_count}

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        query_filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        **kwargs
    ) -> List[ScoredPoint]:
        """
        Search for similar vectors.

        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            limit: Maximum number of results
            query_filter: Filter conditions (simplified implementation)
            score_threshold: Minimum score threshold
            with_payload: Include payload in results
            with_vectors: Include vectors in results
            **kwargs: Additional parameters (ignored)

        Returns:
            List of scored points

        Raises:
            ValueError: If collection doesn't exist
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")

        collection = self.collections[collection_name]

        # Apply filter if provided (simplified)
        if query_filter:
            filtered_points = self._apply_filter(collection, query_filter)
        else:
            filtered_points = collection

        # Calculate cosine similarity for each point
        scored_points = []
        for point in filtered_points:
            similarity = self._cosine_similarity(query_vector, point.vector)

            # Apply score threshold
            if score_threshold is not None and similarity < score_threshold:
                continue

            scored_point = ScoredPoint(
                id=point.id,
                score=similarity,
                payload=point.payload if with_payload else {},
                vector=point.vector if with_vectors else None,
            )
            scored_points.append(scored_point)

        # Sort by score descending and limit
        scored_points.sort(key=lambda x: x.score, reverse=True)
        return scored_points[:limit]

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same dimension")

        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        # Cosine similarity normalized to [0, 1]
        similarity = dot_product / (magnitude1 * magnitude2)
        # Normalize from [-1, 1] to [0, 1]
        return (similarity + 1) / 2

    def _apply_filter(
        self,
        points: List[PointStruct],
        filter_dict: Dict[str, Any]
    ) -> List[PointStruct]:
        """
        Apply filter to points (simplified implementation).

        Args:
            points: List of points to filter
            filter_dict: Filter conditions

        Returns:
            Filtered points
        """
        # Simplified filter implementation for testing
        # Supports basic equality checks on payload fields

        filtered = []
        for point in points:
            matches = True
            for key, value in filter_dict.items():
                if key not in point.payload or point.payload[key] != value:
                    matches = False
                    break
            if matches:
                filtered.append(point)

        return filtered

    # =========================================================================
    # Utility Methods for Testing
    # =========================================================================

    def get_points_count(self, collection_name: str) -> int:
        """Get number of points in collection (test helper)."""
        return len(self.collections.get(collection_name, []))

    def clear_collection(self, collection_name: str):
        """Clear all points from collection (test helper)."""
        if collection_name in self.collections:
            self.collections[collection_name].clear()

    def get_all_points(self, collection_name: str) -> List[PointStruct]:
        """Get all points in collection (test helper)."""
        return self.collections.get(collection_name, []).copy()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_mock_client(**kwargs) -> MockQdrantClient:
    """Create a mock Qdrant client."""
    return MockQdrantClient(**kwargs)


async def create_test_collection(
    client: MockQdrantClient,
    name: str = "test_collection",
    dimension: int = 1536
) -> str:
    """
    Helper to create a test collection.

    Args:
        client: Mock client instance
        name: Collection name
        dimension: Vector dimension

    Returns:
        Collection name
    """
    await client.create_collection(
        collection_name=name,
        vectors_config={
            "size": dimension,
            "distance": "Cosine"
        }
    )
    return name


async def add_test_points(
    client: MockQdrantClient,
    collection_name: str,
    count: int = 10,
    dimension: int = 1536
) -> List[PointStruct]:
    """
    Helper to add test points to a collection.

    Args:
        client: Mock client instance
        collection_name: Name of the collection
        count: Number of points to add
        dimension: Vector dimension

    Returns:
        List of added points
    """
    import random

    points = []
    for i in range(count):
        vector = [random.random() for _ in range(dimension)]
        point = PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload={
                "content": f"Test document {i}",
                "index": i,
                "metadata": {"source": "test"}
            }
        )
        points.append(point)

    await client.upsert(collection_name, points)
    return points
