import csv
import uuid

from src.rag_agent import RAGAgent


def main():
    agent = RAGAgent()
    session_id = 'test_' + str(uuid.uuid4())

    with open('test.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='~')
        rows = list(reader)
        print(rows)

    with open('result.txt', 'w', encoding='utf-8') as out:
        for i, row in enumerate(rows, start=1):
            question = row['Вопрос'].strip()
            agent.clear_history(session_id)
            answer = agent.ask(session_id, question)
            out.write(f'{i}\n{answer}\n')


if __name__ == "__main__":
    main()