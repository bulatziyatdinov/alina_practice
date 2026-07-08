import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR: str = os.path.join(BASE_DIR, "src")
DATA_DIR: str = os.path.join(BASE_DIR, "data")
INDEX_DIR: str = os.path.join(BASE_DIR, "index")
DB_PATH: str = os.path.join(BASE_DIR, "chat.db")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

REASONING = os.getenv("REASONING", "False").lower() == "true"
CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", "8192"))
CHAT_MESSAGES_LIMIT = int(os.getenv("CHAT_MESSAGES_LIMIT", "10"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
SEED = int(os.getenv("SEED", "42"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "2048"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

FAISS_K = int(os.getenv("FAISS_K", "8"))
BM25_K = int(os.getenv("BM25_K", "8"))

LOGGER_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"