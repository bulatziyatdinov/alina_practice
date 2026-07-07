import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "index")
DB_PATH = os.path.join(BASE_DIR, "chat_history.db")

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://localhost:11434"
REASONING = False
CONTEXT_LENGTH = 4096
CHAT_MESSAGES_LIMIT = 10

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Настройки гибридного поиска (количество возвращаемых документов)
FAISS_K = 5
BM25_K = 5