import os

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
INDEX_DIR: str = os.path.join(BASE_DIR, "index")
DB_PATH: str = os.path.join(BASE_DIR, "chat_history.db")

EMBEDDING_MODEL: str = "nomic-embed-text"
LLM_MODEL: str = "qwen2.5:1.5b"
OLLAMA_BASE_URL: str = "http://localhost:11434"
REASONING: bool = False
CONTEXT_LENGTH: int = 8192
CHAT_MESSAGES_LIMIT: int = 10
TEMPERATURE: float = 0.35
SEED: int = 42

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

FAISS_K: int = 10
BM25_K: int = 10
