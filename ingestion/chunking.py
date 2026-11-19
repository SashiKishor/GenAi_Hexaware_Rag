
import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> list[Document]:

    if not documents:
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

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

    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_index"] = index

        # Generate unique ID for the chunk
        unique_text = (
            f"{chunk.metadata.get('source', 'unknown')}_"
            f"{chunk.metadata.get('version', 'unknown')}_"
            f"{chunk.metadata.get('page', 'unknown')}_"
            f"{chunk.metadata.get('start_index', index)}_"
            f"{chunk.page_content}"
        )

        chunk_id = hashlib.sha256(
            unique_text.encode("utf-8")
        ).hexdigest()

        chunk.metadata["chunk_id"] = chunk_id

    return chunks
