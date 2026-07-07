import uuid
from src.rag_agent import RAGAgent


def main():
    agent = RAGAgent()
    session_id = str(uuid.uuid4())

    while True:
        try:
            print('='*40)
            query = input("Вопрос: ")
            if query.lower() in {"exit", "quit", "выход", "выйти"}:
                break

            response = agent.ask(session_id, query)
            print(f"\nОтвет: {response}\n")

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
