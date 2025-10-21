"""
Plain text document loader.

Supports .txt, .md, .markdown, .csv, and other text files.
"""

from typing import Optional, Dict, Any
import aiofiles
import os

from app.services.rag.loaders.base import BaseDocumentLoader
from app.services.rag.protocols import Document


class TextLoader(BaseDocumentLoader):
    """
    Loader for plain text files.

    Supports: .txt, .md, .markdown, .csv, .json, .xml, .html, .log, etc.
    """

    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".json",
        ".xml",
        ".html",
        ".htm",
        ".log",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
    }

    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize text loader.

        Args:
            encoding: Text encoding to use (default: utf-8)
        """
        self.encoding = encoding

    def supports(self, source: str) -> bool:
        """
        Check if this loader supports the given source.

        Args:
            source: File path

        Returns:
            True if file extension is supported
        """
        _, ext = os.path.splitext(source.lower())
        return ext in self.SUPPORTED_EXTENSIONS

    async def load(
        self,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Load a text document from file.

        Args:
            source: File path
            metadata: Optional metadata to attach

        Returns:
            Loaded document
        """
        # Read file content
        async with aiofiles.open(source, "r", encoding=self.encoding) as f:
            content = await f.read()

        # Clean text
        content = self.clean_text(content)

        # Build metadata
        full_metadata = self.build_metadata(source, metadata)

        # Add file size
        full_metadata["file_size"] = os.path.getsize(source)

        # Add MIME type based on extension
        _, ext = os.path.splitext(source.lower())
        mime_types = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".markdown": "text/markdown",
            ".csv": "text/csv",
            ".json": "application/json",
            ".xml": "application/xml",
            ".html": "text/html",
            ".htm": "text/html",
            ".yaml": "application/x-yaml",
            ".yml": "application/x-yaml",
        }
        full_metadata["mime_type"] = mime_types.get(ext, "text/plain")

        return Document(
            content=content,
            metadata=full_metadata,
        )
