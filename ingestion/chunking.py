
from typing import List
import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:

    """
    Split extracted documents into smaller chunks.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Maximum characters in each chunk.
        chunk_overlap: Overlapping characters between chunks.

    Returns:
        List of chunked LangChain Document objects.
    """

    if not documents:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)

    logger.info("Generated %d chunks from %d documents", len(chunks), len(documents))

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    logger.info("Document chunking completed successfully.")

    return chunks
