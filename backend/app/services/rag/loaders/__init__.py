"""
Document loaders package.

Loaders for various document types: PDF, DOCX, web pages, plain text, etc.
"""

from app.services.rag.loaders.base import BaseDocumentLoader
from app.services.rag.loaders.text_loader import TextLoader
from app.services.rag.loaders.pdf_loader import PDFLoader
from app.services.rag.loaders.web_loader import WebLoader

# DOCX loader will be added when python-docx is installed
try:
    from app.services.rag.loaders.docx_loader import DocxLoader

    __all__ = [
        "BaseDocumentLoader",
        "TextLoader",
        "PDFLoader",
        "WebLoader",
        "DocxLoader",
    ]
except ImportError:
    __all__ = [
        "BaseDocumentLoader",
        "TextLoader",
        "PDFLoader",
        "WebLoader",
    ]
