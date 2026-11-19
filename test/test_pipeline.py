
from ingestion.pdf_loader import load_pdfs
from ingestion.text_extractor import extract_text
from ingestion.document_cleaner import clean_documents
from ingestion.chunking import split_documents

from embeddings.embedding_model import EmbeddingModel

from vector_store.chroma_db import ChromaDB
from vector_store.retriever import Retriever


def run_pipeline():

    print("\n========== STEP 1: PDF LOADING ==========")

    markdown_files = load_pdfs()

    print("Markdown files:", len(markdown_files))

    for file in markdown_files:
        print("Loaded:", file.name)


    print("\n========== STEP 2: TEXT EXTRACTION ==========")

    documents = extract_text(markdown_files)

    print("Documents extracted:", len(documents))

    for document in documents:
        print("Source:", document.metadata["source"])
        print("Characters:", len(document.page_content))


    print("\n========== STEP 3: DOCUMENT CLEANING ==========")

    cleaned_documents = clean_documents(documents)

    print("Cleaned documents:", len(cleaned_documents))


    print("\n========== STEP 4: CHUNKING ==========")

    chunks = split_documents(
        cleaned_documents,
        chunk_size=1000,
        chunk_overlap=150
    )

    print("Total chunks:", len(chunks))

    for chunk in chunks[:3]:

        print("\nChunk ID:", chunk.metadata["chunk_id"])
        print("Chunk index:", chunk.metadata["chunk_index"])
        print("Source:", chunk.metadata.get("source"))
        print("Content:", chunk.page_content[:200])


    print("\n========== STEP 5: EMBEDDING MODEL ==========")

    embedding_model = EmbeddingModel().get_model()

    print("Embedding model loaded successfully!")


    print("\n========== STEP 6: CHROMADB STORAGE ==========")

    db = ChromaDB(embedding_model)

    db.add_chunks(chunks)

    print("Chunks stored successfully!")


    print("\n========== STEP 7: RETRIEVAL ==========")

    retriever = Retriever(db)

    query = "What is the monthly maintenance fee for Business Plus?"

    results = retriever.search(
        query=query,
        k=3
    )

    print("\nSearch query:", query)
    print("Retrieved chunks:", len(results))

    for index, result in enumerate(results, start=1):

        print(f"\n----- Result {index} -----")

        print("Content:", result.page_content[:300])

        print(
            "Source:",
            result.metadata.get("source")
        )

        print(
            "Version:",
            result.metadata.get("version")
        )

        print(
            "Page:",
            result.metadata.get("page", "Not available")
        )


    print("\n========== PIPELINE COMPLETED SUCCESSFULLY ==========")


if __name__ == "__main__":
    run_pipeline()
