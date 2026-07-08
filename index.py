from src.indexer import Indexer


def main():
    import os

    os.environ['OLLAMA_HOST'] = 'http://127.0.0.1:11434'
    print("[INFO] Индексирование запущено")
    indexer = Indexer()
    num_docs, num_chunks = indexer.run()
    if not (num_docs and num_chunks):
        print("[WARNING] Проблемы с индексацией файлов или их чанкингом")
    print(f"[INFO] Индексирование закончено. Файлов: {num_docs}, чанков: {num_chunks}.")


if __name__ == "__main__":
    main()