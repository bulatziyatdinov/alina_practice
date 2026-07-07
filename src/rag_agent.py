from langchain_ollama import OllamaLLM
from src.config import (LLM_MODEL, OLLAMA_BASE_URL, INDEX_DIR, DB_PATH, FAISS_K, BM25_K,
                        REASONING, CONTEXT_LENGTH, CHAT_MESSAGES_LIMIT, TEMPERATURE, SEED)
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.db import Database


class RAGAgent:
    def __init__(self):
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            reasoning=REASONING,
            num_ctx=CONTEXT_LENGTH,
            temperature=TEMPERATURE,
            seed=SEED,
        )
        self.db = Database(DB_PATH)
        embeddings = Embedder().get_embeddings()
        self.retriever = VectorStore().load_retriever(
            embeddings=embeddings, index_dir=INDEX_DIR, faiss_k=FAISS_K, bm25_k=BM25_K
        )

    def ask(self, session_id: str, query: str) -> str:
        docs = self.retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        history = self.db.get_messages(session_id, CHAT_MESSAGES_LIMIT)
        history_text = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in history]
        )

        prompt = f"""
Ты — эксперт по имитационному моделированию в среде GPSS Studio. Твоя задача — давать 
точные, конструктивные и проверенные ответы на вопросы пользователей, опираясь 
исключительно на предоставленный контекст (фрагменты официального руководства пользователя).  

Правила работы:
**Ссылайся на источники** — всегда указывай раздел и номер страницы (если они есть в метаданных фрагмента). Например: «Согласно разделу 3.7.2 «Вкладка GPSS модель» (стр. 91) ...»
**Структурируй ответ** — разбивай сложные вопросы на логические части, используй списки, выделяй ключевые термины.
**Приводи примеры кода** — если вопрос касается реализации на GPSS или PLUS, покажи фрагмент кода с пояснениями. Пример должен быть рабочим и соответствовать синтаксису, описанному в руководстве.
**Будь краток и по делу** — не перегружай ответ лишней информацией, но дай достаточно деталей для понимания.
**Уточняй, если нужно** — если вопрос неоднозначен, задай уточняющий вопрос перед ответом.
В ответе запрещено использовать markdown форматирование. Опираться на контекст, если он есть.
Убери всякие оформления и декорирования текста. Желательно приводить источник информации,
например страницу и название раздела. Ответ обязательно на русском языке.

Контекст:
{context}

История диалога (если есть):
{history}

Вопрос пользователя:
{query}

Ответ:"""

        response = self.llm.invoke(prompt)

        self.db.add_message(session_id, "user", query)
        self.db.add_message(session_id, "assistant", response)

        return response