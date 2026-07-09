import time
import uuid

from rich import print as rprint
from rich.markdown import Markdown

from src.rag_agent import RAGAgent


def main():
    agent = RAGAgent()
    session_id = str(uuid.uuid4())

    print(
        "[INFO] Приложение запущено\n"
        "Команды:\n"
        "-Выход: exit, quit, выход, выйти\n"
        "-Очистка памяти: clear, очистка, очистить\n"
        "-Помощь: info, help, информация, помощь"
    )

    while True:
        try:
            print("=" * 40)
            query = input("[ВОПРОС] ").strip()

            query_processed = query.lower().strip("\\/")
            if query_processed in {"exit", "quit", "выход", "выйти"}:
                print('[INFO] Приложение закрывается')
                break
            elif query_processed in {"clear", "очистка", "очистить"}:
                agent.clear_history(session_id)
                print("Контекст очищен")
                continue
            elif query_processed in {"info", "help", "информация", "помощь"}:
                print("Команды:"
                      "\n  Выход: exit, quit, выход, выйти"
                      "\n  Очистка памяти: clear, очистка, очистить"
                      "\n  Помощь: info, help, инфо, информация, помощь"
                )
                continue

            start_time = time.time()
            try:
                response = agent.ask(session_id, query)
                rprint(Markdown("[ОТВЕТ] " + response))
            except Exception:
                print("[ОШИБКА] Нет связи с моделями Ollama")
            elapsed = time.time() - start_time
            print(f"Время ответа: {elapsed:.3f} секунд")


        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
