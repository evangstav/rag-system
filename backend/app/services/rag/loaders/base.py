"""
Base document loader class.

Provides common functionality for all document loaders.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import re

from app.services.rag.protocols import Document


class BaseDocumentLoader(ABC):
    """
    Abstract base class for document loaders.

    All document loaders should inherit from this class and implement
    the supports() and load() methods.
    """

    @abstractmethod
    def supports(self, source: str) -> bool:
        """
        Check if this loader supports the given source.

        Args:
            source: File path, URL, or other source identifier

        Returns:
            True if this loader can handle the source
        """
        pass

    @abstractmethod
    async def load(
        self,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Load a document from a source.

        Args:
            source: File path, URL, or other source identifier
            metadata: Optional metadata to attach to the document

        Returns:
            Loaded document with content and metadata
        """
        pass

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Common cleaning operations:
        - Remove excessive whitespace
        - Normalize line endings
        - Remove non-printable characters
        - Fix common encoding issues

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        # Remove non-printable characters (except newlines and tabs)
        text = re.sub(r"[^\x20-\x7E\n\t]", "", text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive whitespace
        text = re.sub(r" +", " ", text)  # Multiple spaces to single space
        text = re.sub(r"\n\n+", "\n\n", text)  # Multiple newlines to double newline

        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def extract_filename(self, source: str) -> str:
        """
        Extract filename from source path or URL.

        Args:
            source: File path or URL

        Returns:
            Filename
        """
        # Handle URLs
        if source.startswith("http://") or source.startswith("https://"):
            from urllib.parse import urlparse
            parsed = urlparse(source)
            return parsed.path.split("/")[-1] or parsed.netloc

        # Handle file paths
        import os
        return os.path.basename(source)

    def build_metadata(
        self,
        source: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build metadata dictionary for a document.

        Args:
            source: Source path or URL
            extra_metadata: Additional metadata to include

        Returns:
            Complete metadata dictionary
        """
        metadata = {
            "source": source,
            "filename": self.extract_filename(source),
            "loader": self.__class__.__name__,
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        return metadata
