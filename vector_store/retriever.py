
class Retriever:

    def __init__(self, db):

        self.vector_store = db.get_vector_store()

    def search(self, query: str, k: int = 3):

        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        return self.vector_store.similarity_search(
            query,
            k=k
        )
