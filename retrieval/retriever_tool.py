from functools import lru_cache

from langchain_core.tools import create_retriever_tool

from retrieval.retriever_factory import get_retriever


def build_retriever_tool():
    retriever = get_retriever()
    return create_retriever_tool(
        retriever,
        "rag_retriever",
        "搜索并返回关于‘半导体和芯片’的信息，内容涵盖：半导体和芯片的封装、测试、光刻胶等。",
    )


@lru_cache(maxsize=1)
def get_retriever_tool():
    return build_retriever_tool()
