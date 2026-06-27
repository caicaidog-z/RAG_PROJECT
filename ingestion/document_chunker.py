from typing import Iterable, List

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from config.settings import AppSettings, get_settings
from models.embedding_factory import get_chunking_embedding


class SemanticDocumentChunker:
    """Apply semantic chunking only when a document body is large enough."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        max_content_length: int = 5000,
    ):
        self.settings = settings or get_settings()
        self.max_content_length = max_content_length
        self.text_splitter = SemanticChunker(
            get_chunking_embedding(self.settings),
            breakpoint_threshold_type="percentile",
        )

    def chunk_documents(self, documents: Iterable[Document]) -> List[Document]:
        chunked_documents: List[Document] = []
        for document in documents:
            if len(document.page_content) > self.max_content_length:
                chunked_documents.extend(self.text_splitter.split_documents([document]))
                continue
            chunked_documents.append(document)
        return chunked_documents
