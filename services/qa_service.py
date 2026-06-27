from functools import lru_cache

from workflows.rag_graph import get_graph


class QAService:
    def __init__(self):
        self.graph = get_graph()

    def stream_question(self, question: str):
        return self.graph.stream({"question": question})

    def answer_question(self, question: str) -> str:
        final_state = None
        for output in self.stream_question(question):
            for value in output.values():
                final_state = value
        if not final_state:
            return ""
        return final_state.get("generation", "")


@lru_cache(maxsize=1)
def get_qa_service() -> QAService:
    return QAService()


def stream_question(question: str):
    return get_qa_service().stream_question(question)


def answer_question(question: str) -> str:
    return get_qa_service().answer_question(question)
