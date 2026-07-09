from langchain_ollama import OllamaLLM

from src.config import (
    BM25_K,
    BM25_WEIGHT,
    CHAT_MESSAGES_LIMIT,
    CONTEXT_LENGTH,
    DB_PATH,
    FAISS_K,
    FAISS_WEIGHT,
    INDEX_DIR,
    LLM_MODEL,
    NUM_PREDICT,
    OLLAMA_BASE_URL,
    REASONING,
    SEED,
    SRC_DIR,
    TEMPERATURE,
)
from src.db import Database
from src.embedder import Embedder
from src.vector_store import VectorStore


class RAGAgent:
    def __init__(self):
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            reasoning=REASONING,
            num_ctx=CONTEXT_LENGTH,
            temperature=TEMPERATURE,
            seed=SEED,
            num_predict=NUM_PREDICT,
        )
        self.db = Database(DB_PATH)
        embeddings = Embedder().get_embeddings()
        self.retriever = VectorStore().load_retriever(
            embeddings=embeddings,
            index_dir=INDEX_DIR,
            faiss_k=FAISS_K,
            bm25_k=BM25_K,
            faiss_w=FAISS_WEIGHT,
            bm25_w=BM25_WEIGHT,
        )
        self.base_prompt = self._get_prompt()

    def _get_prompt(self) -> str:
         with open(SRC_DIR + "/prompt.txt", "r", encoding="utf-8") as f:
             return f.read()

    def ask(self, session_id: str, query: str) -> str:
        history = self.db.get_messages(session_id, CHAT_MESSAGES_LIMIT)
        history_queries = ". ".join(
            [msg["content"] for msg in history if msg["role"] == "user"][-3:]
        )
        history_text = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in history]
        )

        query = query.strip()

        docs = self.retriever.invoke(f"{history_queries}. {query}")

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = (self.base_prompt +
                  f"Контекст: {context}\n"
                  f"История диалога (если есть): {history_text}\n"
                  f"Вопрос пользователя: {query}\n"
                  f"Напиши только ответ на вопрос пользователя")

        response = self.llm.invoke(prompt)

        self.db.add_message(session_id, "user", query)
        self.db.add_message(session_id, "assistant", response)

        return response

    def clear_history(self, session_id: str) -> None:
        self.db.delete_messages(session_id)
