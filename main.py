import uuid
from src.rag_agent import RAGAgent
from rich import print as rprint
from rich.markdown import Markdown

def main():
    agent = RAGAgent()
    session_id = str(uuid.uuid4())

    while True:
        try:
            print('='*40)
            query = input("Вопрос: ")
            # todo: clear command

            if query.lower() in {"exit", "quit", "выход", "выйти"}:
                break

            response = agent.ask(session_id, query)
            rprint(Markdown('Ответ:' + response))

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
