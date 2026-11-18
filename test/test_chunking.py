
from langchain_core.documents import Document
from ingestion.chunking import split_documents


# Placeholder data (simulating extracted PDF content)
sample_documents = [
    Document(
        page_content="""
        Business Plus Account

        The monthly maintenance fee for the Business Plus
        account is Rs. 650.

        Customers can get a monthly fee waiver if they
        maintain an average monthly balance of Rs. 50,000.

        The account provides digital banking facilities,
        online transactions and account management services.

        Customers must complete the required KYC verification
        before the account can be activated.
        """,
        metadata={
            "source": "northbridge_2026.pdf",
            "version": "2026",
            "page": 3,
            "document_type": "product_terms",
            "effective_date": "2026-01-01",
            "status": "current"
        }
    )
]


# Call the chunking function
chunks = split_documents(
    sample_documents,
    chunk_size=100,
    chunk_overlap=20
)


# Display the results
print("Total chunks created:", len(chunks))

for chunk in chunks:
    print("\n================================")
    print("Chunk index:", chunk.metadata["chunk_index"])
    print("Chunk content:", chunk.page_content)
    print("Metadata:", chunk.metadata)
