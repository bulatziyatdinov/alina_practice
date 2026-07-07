import os

from langchain_community.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
)

from src.chunker import Chunker
from src.config import DATA_DIR, INDEX_DIR
from src.embedder import Embedder
from src.vector_store import VectorStore

SUPPORTED_EXTS = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyMuPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".csv": CSVLoader,
}


class Indexer:
    def __init__(self):
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.vector_store = VectorStore()

    def run(self):
        documents = []
        for root, _, files in os.walk(DATA_DIR):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTS:
                    file_path = os.path.join(root, file)
                    loader_class = SUPPORTED_EXTS[ext]
                    loader = loader_class(file_path)
                    loaded_docs = loader.load()

                    for doc in loaded_docs:
                        doc.metadata["source_name"] = file

                    documents.extend(loaded_docs)

        if not documents:
            return

        chunks = self.chunker.split(documents)

        debug_path = os.path.join(DATA_DIR, "chunks_debug.txt")
        with open(debug_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"=== Чанк #{i + 1} ===\n")
                f.write(f"Источник: {chunk.metadata.get('source_name', 'unknown')}\n")
                f.write(f"Страница: {chunk.metadata.get('page', '')}\n")
                f.write("-" * 40 + "\n")
                f.write(chunk.page_content)
                f.write("\n\n" + "=" * 80 + "\n\n")

        embeddings = self.embedder.get_embeddings()
        self.vector_store.save(chunks, embeddings, INDEX_DIR)
