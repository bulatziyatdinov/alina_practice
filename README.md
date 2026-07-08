## Требования

- Python 3.12+
- [Ollama](https://ollama.com/)

## Запуск

1. Запустите Ollama

```bash
ollama serve
```

2. Скачайте модели

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b
```

3. Скачайте проект

```bash
git clone https://github.com/bulatziyatdinov/alina_practice.git
cd alina_practice
```

4. Установите uv и зависимости

```bash
pip install uv
uv sync --no-dev
```


5. Запустите индексацию

```bash
uv run index.py
```

6. Запустите RAG

```bash
uv run main.py
```

---

## Конфигурация (файл `.env`)

```env
# Модели
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=qwen2.5:1.5b
OLLAMA_BASE_URL=http://localhost:11434

# LLM
TEMPERATURE=0.2
CONTEXT_LENGTH=8192
NUM_PREDICT=2048

# Чанкинг
CHUNK_SIZE=1500
CHUNK_OVERLAP=200

# Поиск
FAISS_K=8
BM25_K=8
```

