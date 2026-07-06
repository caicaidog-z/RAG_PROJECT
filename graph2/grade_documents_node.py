from concurrent.futures import ThreadPoolExecutor

from graph2.grader_chain import retrieval_grader_chain
from utils.log_utils import log

# 文档评分是 I/O 密集（等 LLM 返回），用线程池并发打分，N 个文档从 N 次串行往返压成 1 次
_MAX_WORKERS = 8


def grade_documents(state):
    """
    评估检索到的文档与问题的相关性

    Args:
        state (dict): 当前图状态，包含问题和检索结果

    Returns:
        state (dict): 更新后的状态，documents字段仅保留相关文档
    """
    log.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")  # 打印当前阶段标识
    question = state["question"]  # 获取用户问题
    documents = state["documents"]  # 获取待评估文档

    def _grade_one(doc):
        """对单个文档评分，返回 (doc, 是否相关)"""
        try:
            score = retrieval_grader_chain.invoke(  # 调用评分器评估文档相关性
                {"question": question, "document": doc.page_content}
            )
            return doc, score.binary_score == "yes"
        except Exception as e:  # 单个文档评分失败不阻塞整体流程
            log.warning(f"文档评分异常，按不相关处理: {e}")
            return doc, False

    # 并发评分：所有文档同时打分，而非逐个等待
    filtered_docs = []
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, max(1, len(documents)))) as ex:
        for doc, is_relevant in ex.map(_grade_one, documents):
            if is_relevant:
                log.info("---GRADE: 打印相关标识---")
                filtered_docs.append(doc)
            else:
                log.info("---GRADE: 打印不相关标识,并丢掉doc---")
    return {"documents": filtered_docs, "question": question}  # 返回仅含相关文档的状态