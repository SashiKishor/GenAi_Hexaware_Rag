import logging
import re
import unicodedata
import html
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null characters
    text = text.replace("\x00", "")

    # Remove unwanted control characters
    text = "".join(
        char for char in text
        if char in "\n\t" or char.isprintable()
    )

    # Fix words split across PDF lines
    text = re.sub(
        r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)",
        "",
        text
    )

    # Normalize spaces
    text = text.replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)

    # Preserve Markdown headings and tables
    text = "\n".join(
        line.strip() for line in text.splitlines()
    )

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_documents(documents: list[Document]) -> list[Document]:

    cleaned_documents = []
    for document in documents:
        cleaned_content = clean_text(document.page_content)

        if not cleaned_content:
            logger.warning(
                "Skipping empty document: %s",
                document.metadata.get("source")
            )
            continue

        metadata = document.metadata.copy()
        metadata.update({
            "original_character_count": len(document.page_content),
            "cleaned_character_count": len(cleaned_content),
            "cleaning_applied": True
        })

        cleaned_documents.append(
            Document(
                page_content=cleaned_content,
                metadata=metadata
            )
        )

    if not cleaned_documents:
        raise ValueError("No usable documents after cleaning.")

    logger.info(
        "Document cleaning completed: %d documents",
        len(cleaned_documents)
    )

    return cleaned_documents



 
