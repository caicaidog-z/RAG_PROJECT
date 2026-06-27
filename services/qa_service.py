from workflows.rag_graph import graph


def stream_question(question: str):
    """流式执行主 RAG 图。"""
    return graph.stream({"question": question})


def answer_question(question: str) -> str:
    """执行主 RAG 图并返回最终回答。"""
    final_state = None
    for output in stream_question(question):
        for value in output.values():
            final_state = value
    if not final_state:
        return ""
    return final_state.get("generation", "")
