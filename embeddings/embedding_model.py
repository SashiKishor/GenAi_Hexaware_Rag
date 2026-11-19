
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:

    def __init__(self):
        try:
            # Load local cached model directly without connecting to HF hub
            self.model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"local_files_only": True},
            )
        except Exception:
            # Fallback to online loading if model is not yet cached locally
            self.model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

    def get_model(self):
        return self.model
