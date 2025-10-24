"""
Test suite loader.

Loads test queries from JSON files.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from app.evaluation.protocols import TestQuery


class TestSuiteLoader:
    """Load test suites from JSON files."""

    @staticmethod
    def load_from_json(file_path: str | Path) -> List[TestQuery]:
        """
        Load test queries from a JSON file.

        Expected format:
        {
          "name": "My Test Suite",
          "description": "Tests for feature X",
          "queries": [
            {
              "id": "q1",
              "query": "How do I reset my password?",
              "relevant_document_ids": ["doc_123", "doc_456"],
              "collection_name": "support_docs",
              "query_type": "factual",
              "difficulty": "easy"
            },
            ...
          ]
        }
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Test suite file not found: {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        queries = data.get("queries", [])

        test_queries = []
        for q in queries:
            test_query = TestQuery(
                id=q["id"],
                query=q["query"],
                relevant_document_ids=q["relevant_document_ids"],
                collection_name=q["collection_name"],
                query_type=q.get("query_type", "general"),
                difficulty=q.get("difficulty", "medium"),
                description=q.get("description"),
                expected_answer_keywords=q.get("expected_answer_keywords"),
                metadata=q.get("metadata"),
            )
            test_queries.append(test_query)

        return test_queries

    @staticmethod
    def save_to_json(test_queries: List[TestQuery], file_path: str | Path, metadata: Dict[str, Any] = None):
        """
        Save test queries to a JSON file.

        Args:
            test_queries: List of test queries
            file_path: Path to save to
            metadata: Optional metadata (name, description, etc.)
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": metadata.get("name", "Test Suite") if metadata else "Test Suite",
            "description": metadata.get("description", "") if metadata else "",
            "queries": [
                {
                    "id": q.id,
                    "query": q.query,
                    "relevant_document_ids": q.relevant_document_ids,
                    "collection_name": q.collection_name,
                    "query_type": q.query_type,
                    "difficulty": q.difficulty,
                    "description": q.description,
                    "expected_answer_keywords": q.expected_answer_keywords,
                    "metadata": q.metadata,
                }
                for q in test_queries
            ]
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def create_example_suite(output_path: str | Path):
        """
        Create an example test suite file.

        Useful for getting started.
        """
        example_queries = [
            TestQuery(
                id="q1",
                query="How do I reset my password?",
                relevant_document_ids=["doc_password_reset", "doc_account_settings"],
                collection_name="support_docs",
                query_type="factual",
                difficulty="easy",
                description="Basic password reset question",
            ),
            TestQuery(
                id="q2",
                query="What are the differences between Pro and Enterprise plans?",
                relevant_document_ids=["doc_pricing"],
                collection_name="support_docs",
                query_type="comparison",
                difficulty="medium",
                description="Comparison question requiring multiple data points",
            ),
            TestQuery(
                id="q3",
                query="How does JWT authentication work in the API?",
                relevant_document_ids=["doc_api_auth", "doc_jwt_guide"],
                collection_name="technical_docs",
                query_type="technical",
                difficulty="hard",
                description="Complex technical question",
            ),
        ]

        metadata = {
            "name": "Example Test Suite",
            "description": "Sample test suite for RAG evaluation",
        }

        TestSuiteLoader.save_to_json(example_queries, output_path, metadata)
        print(f"Example test suite created at: {output_path}")
