import os

# Настройки путей проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "index")
DB_PATH = os.path.join(BASE_DIR, "chat_history.db")

# Настройки моделей Ollama
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:1.5b"
LLM_MODEL = "qwen3.5:2b-q4_K_M"
OLLAMA_BASE_URL = "http://localhost:11434"
REASONING = False
CONTEXT_LENGTH = 4096
CHAT_MESSAGES_LIMIT = 10

# Настройки разбиения текста (чанкинга)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Настройки гибридного поиска (количество возвращаемых документов)
FAISS_K = 3
BM25_K = 3