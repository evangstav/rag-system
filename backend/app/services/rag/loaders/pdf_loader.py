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

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF is encrypted or contains no text
            RuntimeError: If PDF reading fails for other reasons
        """
        # Validate file exists
        if not os.path.exists(source):
            raise FileNotFoundError(f"PDF file not found: {source}")

        # Validate it's a file (not directory)
        if not os.path.isfile(source):
            raise ValueError(f"Path is not a file: {source}")

        try:
            # Read PDF
            reader = PdfReader(source)

            # Check if encrypted
            if reader.is_encrypted:
                raise ValueError(
                    f"PDF is encrypted and cannot be read without a password: {source}"
                )

            # Extract text from all pages
            pages_text = []
            failed_pages = []

            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        pages_text.append(text)
                except Exception as e:
                    # Log but continue with other pages
                    failed_pages.append((page_num, str(e)))
                    continue

            # Check if any text was extracted
            if not pages_text:
                error_msg = f"No text content extracted from PDF: {source}"
                if failed_pages:
                    error_msg += f" ({len(failed_pages)} pages failed to extract)"
                raise ValueError(error_msg)

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
            full_metadata["pages_extracted"] = len(pages_text)

            # Add warning if some pages failed
            if failed_pages:
                full_metadata["extraction_warnings"] = [
                    f"Page {page_num}: {error}" for page_num, error in failed_pages
                ]

            # Extract PDF metadata if available
            if reader.metadata:
                pdf_info = {}
                try:
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
                except Exception:
                    # Metadata extraction is optional, continue if it fails
                    pass

            return Document(
                content=content,
                metadata=full_metadata,
            )

        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as e:
            # Wrap any other exceptions in a RuntimeError
            raise RuntimeError(f"Failed to load PDF {source}: {str(e)}") from e
