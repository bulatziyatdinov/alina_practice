import os
import pickle

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStore:
    def save(self, documents: list[Document], embeddings: Embeddings, index_dir: str):
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

        faiss_store = FAISS.from_documents(documents, embeddings)
        faiss_store.save_local(index_dir)

        bm25_retriever = BM25Retriever.from_documents(documents)
        with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
            pickle.dump(bm25_retriever, f)

    def load_retriever(self, embeddings: Embeddings, index_dir: str, faiss_k: int,
                       bm25_k: int, bm25_w: int, faiss_w: int) -> EnsembleRetriever:
        faiss_store = FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )
        faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": faiss_k})

        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            bm25_retriever = pickle.load(f)
        bm25_retriever.k = bm25_k

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[bm25_w, faiss_w],
        )
        return ensemble_retriever
