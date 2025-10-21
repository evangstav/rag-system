"""
Web page loader.

Fetches and extracts text from web pages using aiohttp and BeautifulSoup.
"""

from typing import Optional, Dict, Any
import re

from app.services.rag.loaders.base import BaseDocumentLoader
from app.services.rag.protocols import Document


class WebLoader(BaseDocumentLoader):
    """
    Loader for web pages.

    Fetches HTML content and extracts readable text.
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize web loader.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout

    def supports(self, source: str) -> bool:
        """
        Check if this loader supports the given source.

        Args:
            source: URL

        Returns:
            True if source is a valid HTTP/HTTPS URL
        """
        return source.startswith("http://") or source.startswith("https://")

    async def load(
        self,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Load a web page.

        Args:
            source: URL to fetch
            metadata: Optional metadata to attach

        Returns:
            Loaded document with extracted text
        """
        import aiohttp
        from bs4 import BeautifulSoup

        # Fetch the page
        async with aiohttp.ClientSession() as session:
            async with session.get(source, timeout=self.timeout) as response:
                html = await response.text()
                content_type = response.headers.get("Content-Type", "")

        # Parse HTML
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Extract text
        text = soup.get_text()

        # Clean text
        text = self.clean_text(text)

        # Build metadata
        full_metadata = self.build_metadata(source, metadata)
        full_metadata["mime_type"] = content_type
        full_metadata["source_type"] = "web"

        # Extract title if available
        if soup.title:
            full_metadata["title"] = soup.title.string

        # Extract meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            full_metadata["description"] = meta_desc["content"]

        return Document(
            content=text,
            metadata=full_metadata,
        )
