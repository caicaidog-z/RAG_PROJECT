"""问答业务逻辑。

流式问答路径：自建异步 RAG chain（retriever.ainvoke + llm.astream），
逐 token 推送，跳过 LangGraph 的幻觉打分/重试循环以换取实时流式体验。

prompt 与 generate_node2.py 保持一致，保证回答风格不变。
"""

import json
from typing import AsyncIterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from api.deps import get_llm, get_retriever
from utils.log_utils import log

# 与 graph2/generate_node2.py 同款提示模板，保证回答风格一致
RAG_PROMPT = PromptTemplate(
    template=(
        "你是一个问答任务助手。请根据以下检索到的上下文内容回答问题。"
        "如果不知道答案，请直接说明。回答保持简洁。\n"
        "问题：{question} \n上下文：{context} \n回答："
    ),
    input_variables=["question", "context"],
)


def _format_docs(docs) -> str:
    """把多个 Document 拼成上下文字符串（与 generate_node2 一致）。"""
    if isinstance(docs, list):
        return "\n\n".join(doc.page_content for doc in docs)
    return "\n\n" + docs.page_content


def _doc_to_source(doc):
    """提取文档来源元信息，供前端渲染引用。"""
    meta = getattr(doc, "metadata", {}) or {}
    return {
        "title": str(meta.get("title", "")),
        "source": str(meta.get("source", "")),
        "filename": str(meta.get("filename", "")),
        "category": str(meta.get("category", "")),
    }


async def stream_chat(question: str) -> AsyncIterator[str]:
    """流式问答生成器，逐条产出 SSE 事件行（已含 `data: ` 前缀与空行）。

    事件类型：
      - source: 检索到的文档来源（先于 token 发送）
      - token : LLM 生成的 token 片段
      - done  : 结束
      - error : 异常
    """
    try:
        retriever = get_retriever()
        llm = get_llm()

        log.info(f"[stream_chat] 开始检索: question={question!r}")
        documents = await retriever.ainvoke(question)
        log.info(f"[stream_chat] 检索到 {len(documents)} 条文档")

        # 1) 先推送来源
        sources = [_doc_to_source(d) for d in documents]
        yield _sse({"type": "source", "sources": sources})

        # 2) 流式生成 token
        context = _format_docs(documents)
        chain = RAG_PROMPT | llm | StrOutputParser()

        async for chunk in chain.astream({"context": context, "question": question}):
            if chunk:
                yield _sse({"type": "token", "content": chunk})

        # 3) 结束
        yield _sse({"type": "done"})
    except Exception as exc:  # noqa: BLE001 - SSE 需把错误推给前端
        log.exception(f"[stream_chat] 异常: {exc}")
        yield _sse({"type": "error", "content": str(exc)})


def chat_sync(question: str) -> dict:
    """同步问答：复用编译好的 LangGraph（含路由/幻觉打分/重试），返回完整答案。

    供 POST /api/chat 使用。graph.invoke 同步执行，可能较慢（含多次 LLM 调用）。
    """
    from api.deps import get_graph

    graph = get_graph()
    log.info(f"[chat_sync] 执行 graph: question={question!r}")
    result = graph.invoke({"question": question})

    generation = result.get("generation", "")
    documents = result.get("documents", []) or []
    sources = [_doc_to_source(d) for d in documents]
    return {"answer": generation, "sources": sources}


def _sse(payload: dict) -> str:
    """把 dict 序列化为一条 SSE 事件行。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
