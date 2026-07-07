# from langchain_ollama import OllamaEmbeddings
# from src.config import EMBEDDING_MODEL, OLLAMA_BASE_URL
#
#
# class Embedder:
#     def __init__(self):
#         self.embeddings = OllamaEmbeddings(
#             model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL
#         )
#
#     def get_embeddings(self):
#         return self.embeddings

from langchain_ollama import OllamaEmbeddings
from src.config import EMBEDDING_MODEL, OLLAMA_BASE_URL

class Embedder:
    def get_embeddings(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
