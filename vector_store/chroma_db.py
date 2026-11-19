
from pathlib import Path

from langchain_chroma import Chroma

from embeddings.embedding_model import EmbeddingModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHROMA_DIR = (
    PROJECT_ROOT / "data" / "processed" / "chroma_db"
)


class ChromaDB:

    def __init__(self, embedding_model=None):

        if embedding_model is None:
            embedding_model = EmbeddingModel().get_model()

        CHROMA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        self.vector_store = Chroma(
            collection_name="northbridge_documents",
            embedding_function=embedding_model,
            persist_directory=str(CHROMA_DIR)
        )

    def add_chunks(self, chunks):

        if not chunks:
            raise ValueError(
                "No chunks provided to store."
            )

        ids = []

        for chunk in chunks:

            chunk_id = chunk.metadata.get("chunk_id")

            if not chunk_id:
                raise ValueError(
                    "A chunk is missing chunk_id metadata."
                )

            ids.append(chunk_id)

        self.vector_store.add_documents(
            documents=chunks,
            ids=ids
        )

    def get_vector_store(self):

        return self.vector_store
