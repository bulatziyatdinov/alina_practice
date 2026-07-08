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


class Indexer:
    def __init__(self):
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.vector_store = VectorStore()

    def run(self) -> tuple[int, int]:
        documents = []
        num_docs = 0
        for root, _, files in os.walk(DATA_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[-1].lower()

                if ext in {".txt", ".md"}:
                    loader = TextLoader(file_path, encoding="utf-8")
                elif ext == ".pdf":
                    loader = PyMuPDFLoader(file_path)
                elif ext in {".docx", ".doc"}:
                    loader = UnstructuredWordDocumentLoader(file_path)
                elif ext in {".xlsx", ".xls"}:
                    loader = UnstructuredExcelLoader(file_path)
                elif ext == ".csv":
                    loader = CSVLoader(file_path)
                else:
                    continue

                loaded_docs = loader.load()
                num_docs += 1
                for doc in loaded_docs:
                    doc.metadata["source_name"] = file
                documents.extend(loaded_docs)

        if not documents:
            return 0, 0

        chunks = self.chunker.split(documents)

        os.makedirs(INDEX_DIR, exist_ok=True)

        debug_path = os.path.join(INDEX_DIR, "chunks_debug.txt")
        with open(debug_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f'=== Чанк #{i} ===\n')
                f.write(f'Источник: {chunk.metadata.get("source_name", "unknown")}\n')
                f.write(f'Страница: {chunk.metadata.get("page", "None")}\n')
                f.write('-' * 40 + '\n')
                f.write(chunk.page_content)
                f.write("\n\n" + "=" * 80 + "\n\n")


        embeddings = self.embedder.get_embeddings()
        self.vector_store.save(chunks, embeddings, INDEX_DIR)

        return num_docs, i
