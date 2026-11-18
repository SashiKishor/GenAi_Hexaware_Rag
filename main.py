from ingestion.pdf_loader import load_pdfs
from ingestion.text_extractor import extract_text
from ingestion.chunking import split_documents
from ingestion.document_cleaner import clean_documents
import logging
from colorlog import ColoredFormatter

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter(
    "%(log_color)s%(levelname)s: %(name)s: %(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    }
))

logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    markdown_files = load_pdfs()
    documents = extract_text(markdown_files)
    cleaned_documents = clean_documents(documents)

    chunks = split_documents(
        cleaned_documents,
        chunk_size=1000,
        chunk_overlap=150
    )

    print("Total cleaned documents:", len(cleaned_documents))
    print("Total chunks generated:", len(chunks))
    print("Processing completed successfully!")

    if chunks:
        print("\nFirst chunk:")
        print(chunks[0].page_content)
        print("\nChunk metadata:")
        print(chunks[0].metadata)