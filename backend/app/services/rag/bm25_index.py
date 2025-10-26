"""
BM25 keyword-based indexing for hybrid search.

BM25 (Best Match 25) is a probabilistic ranking function used for keyword matching.
It complements semantic search by capturing exact term matches that embeddings might miss.
"""

from typing import List, Dict, Any, Optional
import re
from collections import defaultdict
from rank_bm25 import BM25Okapi
import logging

from app.services.rag.protocols import DocumentChunk

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 keyword index for efficient keyword-based retrieval.

    This index works alongside vector search to provide hybrid search capabilities.
    BM25 excels at finding exact term matches, product IDs, technical terms, etc.

    Usage:
        index = BM25Index()
        await index.index_documents(collection_name, chunks)
        results = await index.search(collection_name, "query text", limit=10)
    """

    def __init__(self):
        """Initialize BM25 index storage."""
        # Collection name -> BM25Okapi instance
        self._bm25_indices: Dict[str, BM25Okapi] = {}

        # Collection name -> List of document metadata
        self._document_metadata: Dict[str, List[Dict[str, Any]]] = {}

        # Collection name -> List of tokenized documents
        self._tokenized_docs: Dict[str, List[List[str]]] = {}

    async def index_documents(
        self,
        collection_name: str,
        documents: List[DocumentChunk],
    ) -> None:
        """
        Build or update BM25 index for a collection.

        Args:
            collection_name: Name of the collection
            documents: List of document chunks to index
        """
        if not documents:
            logger.warning(f"No documents provided for BM25 indexing: {collection_name}")
            return

        # Tokenize all documents
        tokenized_docs = [self._tokenize(doc.content) for doc in documents]

        # Extract metadata for retrieval
        metadata_list = [
            {
                "content": doc.content,
                "document_id": doc.metadata.get("document_id"),
                "chunk_index": doc.chunk_index,
                **doc.metadata,
            }
            for doc in documents
        ]

        # Build BM25 index
        bm25_index = BM25Okapi(tokenized_docs)

        # Store everything
        self._bm25_indices[collection_name] = bm25_index
        self._document_metadata[collection_name] = metadata_list
        self._tokenized_docs[collection_name] = tokenized_docs

        logger.info(
            f"BM25 index built for collection '{collection_name}' "
            f"with {len(documents)} documents"
        )

    async def append_documents(
        self,
        collection_name: str,
        documents: List[DocumentChunk],
    ) -> None:
        """
        Append new documents to existing BM25 index.

        Note: BM25Okapi doesn't support incremental updates, so we rebuild the entire index.

        Args:
            collection_name: Name of the collection
            documents: New documents to add
        """
        if collection_name not in self._bm25_indices:
            # No existing index, create new
            await self.index_documents(collection_name, documents)
            return

        # Get existing documents
        existing_metadata = self._document_metadata[collection_name]
        existing_tokenized = self._tokenized_docs[collection_name]

        # Tokenize new documents
        new_tokenized = [self._tokenize(doc.content) for doc in documents]
        new_metadata = [
            {
                "content": doc.content,
                "document_id": doc.metadata.get("document_id"),
                "chunk_index": doc.chunk_index,
                **doc.metadata,
            }
            for doc in documents
        ]

        # Combine with existing
        all_tokenized = existing_tokenized + new_tokenized
        all_metadata = existing_metadata + new_metadata

        # Rebuild index with all documents
        bm25_index = BM25Okapi(all_tokenized)

        # Update storage
        self._bm25_indices[collection_name] = bm25_index
        self._document_metadata[collection_name] = all_metadata
        self._tokenized_docs[collection_name] = all_tokenized

        logger.info(
            f"BM25 index updated for collection '{collection_name}' "
            f"(+{len(documents)} documents, total: {len(all_metadata)})"
        )

    async def search(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
    ) -> List[tuple[Dict[str, Any], float]]:
        """
        Search using BM25 keyword matching.

        Args:
            collection_name: Collection to search
            query: Search query
            limit: Maximum number of results

        Returns:
            List of tuples: (document_metadata, bm25_score)
            Sorted by score (highest first)
        """
        if collection_name not in self._bm25_indices:
            logger.warning(
                f"BM25 index not found for collection '{collection_name}'. "
                "Returning empty results."
            )
            return []

        # Get BM25 index and documents
        bm25 = self._bm25_indices[collection_name]
        documents = self._document_metadata[collection_name]

        # Tokenize query
        query_tokens = self._tokenize(query)

        if not query_tokens:
            logger.warning(f"Empty query after tokenization: '{query}'")
            return []

        # Get BM25 scores for all documents
        scores = bm25.get_scores(query_tokens)

        # Create (doc, score) pairs
        doc_score_pairs = list(zip(documents, scores))

        # Filter out zero scores and sort by score (descending)
        doc_score_pairs = [
            (doc, float(score)) for doc, score in doc_score_pairs if score > 0
        ]
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        return doc_score_pairs[:limit]

    async def delete_collection(self, collection_name: str) -> None:
        """
        Delete BM25 index for a collection.

        Args:
            collection_name: Collection to delete
        """
        if collection_name in self._bm25_indices:
            del self._bm25_indices[collection_name]
            del self._document_metadata[collection_name]
            del self._tokenized_docs[collection_name]
            logger.info(f"BM25 index deleted for collection '{collection_name}'")

    async def delete_by_document_id(
        self,
        collection_name: str,
        document_id: str,
    ) -> int:
        """
        Delete all chunks belonging to a document and rebuild index.

        Args:
            collection_name: Collection name
            document_id: Document ID to delete

        Returns:
            Number of chunks deleted
        """
        if collection_name not in self._document_metadata:
            return 0

        # Filter out chunks from the target document
        old_metadata = self._document_metadata[collection_name]
        old_tokenized = self._tokenized_docs[collection_name]

        new_metadata = []
        new_tokenized = []
        deleted_count = 0

        for meta, tokens in zip(old_metadata, old_tokenized):
            if meta.get("document_id") == document_id:
                deleted_count += 1
            else:
                new_metadata.append(meta)
                new_tokenized.append(tokens)

        if deleted_count == 0:
            return 0

        # Rebuild index
        if new_metadata:
            bm25_index = BM25Okapi(new_tokenized)
            self._bm25_indices[collection_name] = bm25_index
            self._document_metadata[collection_name] = new_metadata
            self._tokenized_docs[collection_name] = new_tokenized
        else:
            # No documents left, delete collection
            await self.delete_collection(collection_name)

        logger.info(
            f"Deleted {deleted_count} chunks from BM25 index "
            f"for document '{document_id}' in collection '{collection_name}'"
        )

        return deleted_count

    def has_collection(self, collection_name: str) -> bool:
        """Check if collection exists in BM25 index."""
        return collection_name in self._bm25_indices

    def get_collection_size(self, collection_name: str) -> int:
        """Get number of documents in collection."""
        if collection_name not in self._document_metadata:
            return 0
        return len(self._document_metadata[collection_name])

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.

        Simple tokenization: lowercase, split on non-alphanumeric,
        remove tokens shorter than 2 characters.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()

        # Split on non-alphanumeric characters (keep numbers and letters)
        tokens = re.findall(r'\w+', text)

        # Filter out very short tokens (single characters)
        tokens = [t for t in tokens if len(t) >= 2]

        return tokens
