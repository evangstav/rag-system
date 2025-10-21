"""
PDF document loader.

Extracts text from PDF files using PyPDF2.
"""

from typing import Optional, Dict, Any
import os
from PyPDF2 import PdfReader

from app.services.rag.loaders.base import BaseDocumentLoader
from app.services.rag.protocols import Document


class PDFLoader(BaseDocumentLoader):
    """
    Loader for PDF files.

    Uses PyPDF2 to extract text from PDF documents.
    """

    def supports(self, source: str) -> bool:
        """
        Check if this loader supports the given source.

        Args:
            source: File path

        Returns:
            True if file has .pdf extension
        """
        return source.lower().endswith(".pdf")

    async def load(
        self,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Load a PDF document from file.

        Args:
            source: File path to PDF
            metadata: Optional metadata to attach

        Returns:
            Loaded document with extracted text
        """
        # Read PDF
        reader = PdfReader(source)

        # Extract text from all pages
        pages_text = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                pages_text.append(text)

        # Combine all pages
        content = "\n\n".join(pages_text)

        # Clean text
        content = self.clean_text(content)

        # Build metadata
        full_metadata = self.build_metadata(source, metadata)

        # Add PDF-specific metadata
        full_metadata["file_size"] = os.path.getsize(source)
        full_metadata["mime_type"] = "application/pdf"
        full_metadata["num_pages"] = len(reader.pages)

        # Extract PDF metadata if available
        if reader.metadata:
            pdf_info = {}
            if reader.metadata.title:
                pdf_info["title"] = reader.metadata.title
            if reader.metadata.author:
                pdf_info["author"] = reader.metadata.author
            if reader.metadata.subject:
                pdf_info["subject"] = reader.metadata.subject
            if reader.metadata.creator:
                pdf_info["creator"] = reader.metadata.creator

            if pdf_info:
                full_metadata["pdf_info"] = pdf_info

        return Document(
            content=content,
            metadata=full_metadata,
        )
