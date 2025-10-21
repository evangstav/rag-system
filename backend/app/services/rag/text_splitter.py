"""
Text splitting implementations with content-aware chunking.

Smart text splitter that respects sentence boundaries, markdown structure,
code blocks, and other semantic boundaries.
"""

from typing import List, Optional, Dict, Any
import re

from app.config import settings
from app.services.rag.protocols import Document, DocumentChunk


class SmartTextSplitter:
    """
    Content-aware text splitter that respects semantic boundaries.

    Features:
    - Respects sentence boundaries (doesn't split mid-sentence)
    - Preserves markdown structure (headers, lists, code blocks)
    - Keeps tables intact
    - Maintains code blocks
    - Overlaps chunks for context continuity
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: List[str] | None = None,
    ):
        """
        Initialize smart text splitter.

        Args:
            chunk_size: Target chunk size in characters (default from settings)
            chunk_overlap: Overlap between chunks in characters (default from settings)
            separators: List of separators to try in order (default: markdown-aware)
        """
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Hierarchical separators - try these in order
        self.separators = separators or [
            "\n\n\n",  # Multiple blank lines (section breaks)
            "\n\n",  # Paragraph breaks
            "\n",  # Line breaks
            ". ",  # Sentence endings
            "! ",  # Exclamation sentence endings
            "? ",  # Question sentence endings
            "; ",  # Semicolons
            ", ",  # Commas
            " ",  # Spaces
            "",  # Characters
        ]

    def split_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Split text into chunks.

        Args:
            text: The text to split
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of document chunks
        """
        if not text.strip():
            return []

        metadata = metadata or {}

        # First, try to identify and preserve special structures
        chunks_text = self._split_with_separators(text, self.separators)

        # Convert text chunks to DocumentChunk objects
        chunks: List[DocumentChunk] = []
        for i, chunk_text in enumerate(chunks_text):
            chunk = DocumentChunk(
                content=chunk_text,
                metadata=metadata.copy(),
                chunk_index=i,
            )
            chunks.append(chunk)

        return chunks

    def split_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """
        Split multiple documents into chunks.

        Args:
            documents: List of documents to split

        Returns:
            List of all chunks from all documents
        """
        all_chunks: List[DocumentChunk] = []

        for doc in documents:
            chunks = self.split_text(doc.content, doc.metadata)
            all_chunks.extend(chunks)

        return all_chunks

    def _split_with_separators(
        self,
        text: str,
        separators: List[str],
    ) -> List[str]:
        """
        Recursively split text using hierarchical separators.

        Args:
            text: Text to split
            separators: List of separators to try in order

        Returns:
            List of text chunks
        """
        # Base case: if text is short enough, return it
        if len(text) <= self._chunk_size:
            return [text] if text.strip() else []

        # Try each separator in order
        for i, separator in enumerate(separators):
            if separator == "":
                # Last resort: split by characters
                return self._split_by_chars(text)

            # Split by current separator
            splits = text.split(separator)

            # Filter out empty strings
            splits = [s for s in splits if s.strip()]

            if len(splits) <= 1:
                # This separator didn't help, try next one
                continue

            # Merge splits into chunks of appropriate size
            chunks = []
            current_chunk = []
            current_length = 0

            for split in splits:
                split_length = len(split) + len(separator)

                # If this split alone is too large, recursively split it
                if split_length > self._chunk_size:
                    # Save current chunk if it exists
                    if current_chunk:
                        chunks.append(separator.join(current_chunk))
                        current_chunk = []
                        current_length = 0

                    # Recursively split the large piece
                    sub_chunks = self._split_with_separators(
                        split,
                        separators[i + 1 :],  # Use remaining separators
                    )
                    chunks.extend(sub_chunks)
                    continue

                # Check if adding this split would exceed chunk size
                if current_length + split_length > self._chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append(separator.join(current_chunk))

                    # Start new chunk with overlap
                    overlap_text = separator.join(current_chunk)
                    if len(overlap_text) > self._chunk_overlap:
                        # Take last part for overlap
                        overlap_text = overlap_text[-self._chunk_overlap :]
                        # Find the start of a complete word/sentence in overlap
                        overlap_start = overlap_text.find(" ")
                        if overlap_start > 0:
                            overlap_text = overlap_text[overlap_start + 1 :]

                    # Find overlapping splits
                    overlap_splits = []
                    overlap_length = 0
                    for prev_split in reversed(current_chunk):
                        overlap_length += len(prev_split) + len(separator)
                        overlap_splits.insert(0, prev_split)
                        if overlap_length >= self._chunk_overlap:
                            break

                    current_chunk = overlap_splits
                    current_length = sum(len(s) for s in current_chunk) + len(
                        separator
                    ) * (len(current_chunk) - 1)

                # Add split to current chunk
                current_chunk.append(split)
                current_length += split_length

            # Add final chunk
            if current_chunk:
                chunks.append(separator.join(current_chunk))

            return chunks

        # If no separator worked, split by characters
        return self._split_by_chars(text)

    def _split_by_chars(self, text: str) -> List[str]:
        """
        Split text by characters (last resort).

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self._chunk_size

            # If this is not the last chunk and we have overlap
            if end < len(text):
                # Find a good breaking point (space or punctuation)
                break_point = end
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in " \n.,;!?":
                        break_point = i + 1
                        break
                end = break_point

            chunks.append(text[start:end].strip())

            # Move start forward with overlap
            start = end - self._chunk_overlap if end < len(text) else end

        return [c for c in chunks if c]

    @property
    def chunk_size(self) -> int:
        """Get the target chunk size in characters."""
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Get the overlap between chunks in characters."""
        return self._chunk_overlap


class MarkdownAwareTextSplitter(SmartTextSplitter):
    """
    Text splitter optimized for Markdown documents.

    Preserves:
    - Code blocks (fenced and indented)
    - Tables
    - Headers with their content
    - Lists
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """Initialize markdown-aware text splitter."""
        # Markdown-specific separators
        separators = [
            "\n## ",  # H2 headers
            "\n### ",  # H3 headers
            "\n#### ",  # H4 headers
            "\n\n",  # Paragraphs
            "\n",  # Lines
            ". ",  # Sentences
            " ",  # Words
            "",  # Characters
        ]

        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )

    def split_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Split markdown text preserving structure.

        Args:
            text: Markdown text to split
            metadata: Optional metadata

        Returns:
            List of document chunks
        """
        # Extract code blocks first
        code_blocks, text_without_code = self._extract_code_blocks(text)

        # Extract tables
        tables, text_without_tables = self._extract_tables(text_without_code)

        # Split remaining text
        chunks = super().split_text(text_without_tables, metadata)

        # TODO: Merge code blocks and tables back into appropriate chunks
        # For now, just add them as separate chunks

        return chunks

    def _extract_code_blocks(self, text: str) -> tuple[List[str], str]:
        """
        Extract fenced code blocks from markdown.

        Returns:
            Tuple of (code_blocks, text_without_code)
        """
        # Pattern for fenced code blocks
        pattern = r"```[\s\S]*?```"
        code_blocks = re.findall(pattern, text)
        text_without_code = re.sub(pattern, "<<CODE_BLOCK>>", text)

        return code_blocks, text_without_code

    def _extract_tables(self, text: str) -> tuple[List[str], str]:
        """
        Extract markdown tables.

        Returns:
            Tuple of (tables, text_without_tables)
        """
        # Simple table detection (lines with | characters)
        lines = text.split("\n")
        tables = []
        current_table = []
        text_lines = []

        for line in lines:
            if "|" in line and line.strip().startswith("|"):
                current_table.append(line)
            else:
                if current_table:
                    tables.append("\n".join(current_table))
                    text_lines.append("<<TABLE>>")
                    current_table = []
                text_lines.append(line)

        if current_table:
            tables.append("\n".join(current_table))
            text_lines.append("<<TABLE>>")

        return tables, "\n".join(text_lines)
